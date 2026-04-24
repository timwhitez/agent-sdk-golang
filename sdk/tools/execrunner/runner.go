package execrunner

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"
	"sync"
	"time"
)

const (
	DefaultMaxOutputBytes = 100 * 1024
	DefaultKillGrace      = 250 * time.Millisecond
)

// Options controls command execution behavior.
type Options struct {
	Program        string
	Args           []string
	Dir            string
	Env            []string
	Timeout        time.Duration
	MaxOutputBytes int
	ArtifactPrefix string
	ArtifactDir    string
	KillGrace      time.Duration
	OnOutputChunk  func(OutputChunk)
}

// Result captures execution outcome and output accounting.
type Result struct {
	Output            string
	OutputBytes       int64
	CapturedBytes     int
	ExitCode          int
	TimedOut          bool
	OutputTruncated   bool
	OutputPath        string
	ArtifactBytes     int64
	OutputArtifactErr string
}

// OutputChunk is emitted for each captured stdout/stderr write.
type OutputChunk struct {
	Data       []byte
	TotalBytes int64
}

func Run(ctx context.Context, opts Options) (Result, error) {
	res := Result{ExitCode: -1}
	if opts.Program == "" {
		return res, fmt.Errorf("missing program")
	}
	if opts.MaxOutputBytes <= 0 {
		opts.MaxOutputBytes = DefaultMaxOutputBytes
	}
	if opts.KillGrace < 0 {
		opts.KillGrace = 0
	}
	if opts.KillGrace == 0 {
		opts.KillGrace = DefaultKillGrace
	}

	runCtx := ctx
	cancel := func() {}
	if opts.Timeout > 0 {
		runCtx, cancel = context.WithTimeout(ctx, opts.Timeout)
	}
	defer cancel()

	cmd := exec.Command(opts.Program, opts.Args...)
	cmd.Dir = opts.Dir
	if len(opts.Env) > 0 {
		cmd.Env = append([]string(nil), opts.Env...)
	}
	configureProcessGroup(cmd)

	collector := newOutputCollector(opts.MaxOutputBytes, opts.ArtifactDir, opts.ArtifactPrefix, opts.OnOutputChunk)
	cmd.Stdout = collector
	cmd.Stderr = collector

	if err := cmd.Start(); err != nil {
		collector.Close()
		return res, err
	}

	waitCh := make(chan error, 1)
	go func() {
		waitCh <- cmd.Wait()
	}()

	var waitErr error
	var rawWaitErr error
	select {
	case rawWaitErr = <-waitCh:
		waitErr = rawWaitErr
	case <-runCtx.Done():
		res.TimedOut = errors.Is(runCtx.Err(), context.DeadlineExceeded)
		_ = signalProcessGroupTerminate(cmd.Process)

		if opts.KillGrace > 0 {
			timer := time.NewTimer(opts.KillGrace)
			select {
			case rawWaitErr = <-waitCh:
				if !timer.Stop() {
					<-timer.C
				}
			case <-timer.C:
				_ = signalProcessGroupKill(cmd.Process)
				rawWaitErr = <-waitCh
			}
		} else {
			_ = signalProcessGroupKill(cmd.Process)
			rawWaitErr = <-waitCh
		}

		if res.TimedOut {
			waitErr = context.DeadlineExceeded
		} else {
			waitErr = runCtx.Err()
		}
	}

	collector.Close()
	snap := collector.snapshot()
	res.Output = snap.preview
	res.OutputBytes = snap.totalBytes
	res.CapturedBytes = len(res.Output)
	res.OutputTruncated = snap.truncated
	res.OutputPath = snap.outputPath
	res.ArtifactBytes = snap.artifactBytes
	res.OutputArtifactErr = snap.artifactErr
	res.ExitCode = exitCodeFromError(rawWaitErr)

	if waitErr != nil {
		return res, waitErr
	}
	return res, nil
}

func exitCodeFromError(err error) int {
	if err == nil {
		return 0
	}
	type exitCoder interface {
		ExitCode() int
	}
	var exitErr *exec.ExitError
	if errors.As(err, &exitErr) {
		return exitErr.ExitCode()
	}
	if ec, ok := err.(exitCoder); ok {
		return ec.ExitCode()
	}
	return -1
}

type outputSnapshot struct {
	preview       string
	totalBytes    int64
	truncated     bool
	outputPath    string
	artifactBytes int64
	artifactErr   string
}

type outputCollector struct {
	mu sync.Mutex

	limit int
	onOut func(OutputChunk)

	preview bytes.Buffer
	total   int64

	artifactDir    string
	artifactPrefix string
	artifact       *os.File
	artifactPath   string
	artifactBytes  int64
	artifactErr    string
	primed         bool
	truncated      bool
}

func newOutputCollector(limit int, artifactDir, artifactPrefix string, onOut func(OutputChunk)) *outputCollector {
	if strings.TrimSpace(artifactPrefix) == "" {
		artifactPrefix = "tool-output-*.log"
	}
	return &outputCollector{
		limit:          limit,
		onOut:          onOut,
		artifactDir:    artifactDir,
		artifactPrefix: artifactPrefix,
	}
}

func (c *outputCollector) Write(p []byte) (int, error) {
	chunk := append([]byte(nil), p...)
	c.mu.Lock()

	n := len(p)
	if n == 0 {
		c.mu.Unlock()
		return 0, nil
	}
	c.total += int64(n)
	totalBytes := c.total

	remaining := c.limit - c.preview.Len()
	if remaining > 0 {
		head := p
		if len(head) > remaining {
			head = head[:remaining]
		}
		_, _ = c.preview.Write(head)
		p = p[len(head):]
	}

	if len(p) > 0 {
		c.truncated = true
		if err := c.ensureArtifactLocked(); err != nil {
			if c.artifactErr == "" {
				c.artifactErr = err.Error()
			}
		} else if c.artifact != nil {
			written, err := c.artifact.Write(p)
			c.artifactBytes += int64(written)
			if err != nil && c.artifactErr == "" {
				c.artifactErr = err.Error()
			}
			if err == nil && written < len(p) && c.artifactErr == "" {
				c.artifactErr = io.ErrShortWrite.Error()
			}
		}
	}
	callback := c.onOut
	c.mu.Unlock()
	if callback != nil {
		callback(OutputChunk{
			Data:       chunk,
			TotalBytes: totalBytes,
		})
	}
	return n, nil
}

func (c *outputCollector) Close() {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.artifact == nil {
		return
	}
	if err := c.artifact.Close(); err != nil && c.artifactErr == "" {
		c.artifactErr = err.Error()
	}
	c.artifact = nil
}

func (c *outputCollector) snapshot() outputSnapshot {
	c.mu.Lock()
	defer c.mu.Unlock()

	truncated := c.truncated || c.total > int64(c.preview.Len())
	return outputSnapshot{
		preview:       c.preview.String(),
		totalBytes:    c.total,
		truncated:     truncated,
		outputPath:    c.artifactPath,
		artifactBytes: c.artifactBytes,
		artifactErr:   c.artifactErr,
	}
}

func (c *outputCollector) ensureArtifactLocked() error {
	if c.artifact != nil {
		return nil
	}
	f, err := os.CreateTemp(c.artifactDir, c.artifactPrefix)
	if err != nil {
		return err
	}
	if err := f.Chmod(0o600); err != nil {
		_ = f.Close()
		_ = os.Remove(f.Name())
		return err
	}
	c.artifact = f
	c.artifactPath = f.Name()

	if c.primed {
		return nil
	}
	c.primed = true
	if c.preview.Len() == 0 {
		return nil
	}
	written, werr := c.artifact.Write(c.preview.Bytes())
	c.artifactBytes += int64(written)
	if werr != nil {
		return werr
	}
	if written < c.preview.Len() {
		return io.ErrShortWrite
	}
	return nil
}
