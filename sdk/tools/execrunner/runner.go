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
	"sync/atomic"
	"time"
	"unicode/utf8"

	"github.com/timwhitez/agent-sdk-golang/sdk/artifact"
)

const (
	DefaultMaxOutputBytes = 100 * 1024
	DefaultKillGrace      = 250 * time.Millisecond
	// DefaultKillWaitGrace bounds how long Run waits for cmd.Wait to return
	// after SIGKILL. Processes that escaped the group (setsid/daemons) can keep
	// inherited output descriptors open indefinitely, so waiting is best-effort.
	DefaultKillWaitGrace = 5 * time.Second
	// DefaultPipeDrainGrace bounds how long Run keeps reading the output pipes
	// after the foreground process has exited. A background child that inherited
	// stdout/stderr must not turn a finished command into a timeout.
	DefaultPipeDrainGrace = 2 * time.Second
	// invalidUTF8PreviewReplacement marks bytes that cannot be decoded in the
	// bounded preview. The byte-exact artifact is unaffected.
	invalidUTF8PreviewReplacement = "�"
)

// ErrProcessKillTimeout reports that the process group was killed but at least
// one process did not reap within the kill wait grace, so Run stopped waiting.
var ErrProcessKillTimeout = errors.New("execrunner: process did not exit after kill; abandoning wait")

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
	// ArtifactOwner, ArtifactStreamSink, and ArtifactResolverCapability enable
	// canonical, separately-owned stdout/stderr objects. ArtifactDir/Prefix are
	// retained only for the legacy combined-path fallback.
	ArtifactOwner              artifact.Owner
	ArtifactStreamSink         artifact.StreamSink
	ArtifactResolverCapability artifact.ResolverCapability
	KillGrace                  time.Duration
	// KillWaitGrace bounds the post-SIGKILL wait for the process to be reaped.
	// Defaults to DefaultKillWaitGrace.
	KillWaitGrace time.Duration
	// PipeDrainGrace bounds the wait for output pipes to reach EOF once the
	// process has exited. Defaults to DefaultPipeDrainGrace.
	PipeDrainGrace time.Duration
	OnOutputChunk  func(OutputChunk)
}

// Result captures execution outcome and output accounting.
type Result struct {
	Output        string
	OutputBytes   int64
	CapturedBytes int
	ExitCode      int
	TimedOut      bool
	// KillProcessTimedOut reports that the process group was killed but the
	// process was not reaped within KillWaitGrace, so Run abandoned the wait.
	KillProcessTimedOut bool
	// PipeDrainTimedOut reports that the process exited but a surviving
	// descendant still held stdout/stderr, so the pipes were closed after
	// PipeDrainGrace. Later output was not captured and the streams are not
	// committed as complete.
	PipeDrainTimedOut bool
	OutputTruncated   bool
	OutputPath        string
	ArtifactBytes     int64
	OutputArtifactErr string
	// OutputArtifacts contains only complete, validated canonical raw-stream
	// manifests. Failed streams are represented in OutputArtifactDiagnostics.
	OutputArtifacts           []artifact.Manifest
	OutputArtifactDiagnostics []artifact.Diagnostic
}

// OutputChunk is emitted for each captured stdout/stderr write.
type OutputChunk struct {
	Data       []byte
	TotalBytes int64
	Stream     string
}

// Run rejects an ended context after argument validation and immediately before
// starting the command. These checks are not atomic with OS process creation;
// cancellation after Start uses the existing bounded process-tree termination.
func Run(ctx context.Context, opts Options) (Result, error) {
	return run(ctx, opts, context.WithTimeout, nil, (*exec.Cmd).Start)
}

// run keeps timeout creation, preparation and Start injectable per call for deterministic tests.
func run(ctx context.Context, opts Options, withTimeout func(context.Context, time.Duration) (context.Context, context.CancelFunc), beforeStart func(context.Context), start func(*exec.Cmd) error) (Result, error) {
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
	if opts.KillWaitGrace <= 0 {
		opts.KillWaitGrace = DefaultKillWaitGrace
	}
	if opts.PipeDrainGrace <= 0 {
		opts.PipeDrainGrace = DefaultPipeDrainGrace
	}
	if err := ctx.Err(); err != nil {
		res.TimedOut = errors.Is(err, context.DeadlineExceeded)
		return res, err
	}

	runCtx := ctx
	cancel := func() {}
	if opts.Timeout > 0 {
		runCtx, cancel = withTimeout(ctx, opts.Timeout)
	}
	defer cancel()
	if err := runCtx.Err(); err != nil {
		res.TimedOut = errors.Is(err, context.DeadlineExceeded)
		return res, err
	}

	cmd := exec.Command(opts.Program, opts.Args...)
	cmd.Dir = opts.Dir
	if len(opts.Env) > 0 {
		cmd.Env = append([]string(nil), opts.Env...)
	}
	configureProcessGroup(cmd)

	collector := newOutputCollector(opts.MaxOutputBytes, opts.ArtifactDir, opts.ArtifactPrefix, opts.OnOutputChunk)
	canonical := newCanonicalProcessStreams(context.WithoutCancel(ctx), opts)
	if canonical.requested {
		collector.disableLegacyArtifact()
	}
	// Own the output pipes (instead of letting exec create them) so EOF is observed
	// directly: exec.Cmd.Wait reports ErrWaitDelay only for a zero exit status, so
	// it cannot tell whether a failed or killed command's streams were complete.
	streams := []*outputPump{
		{writer: processStreamWriter{stream: "stdout", combined: collector, canonical: canonical.stdout}},
		{writer: processStreamWriter{stream: "stderr", combined: collector, canonical: canonical.stderr}},
	}
	closePumpFiles := func() {
		for _, st := range streams {
			st.closeFiles()
		}
	}
	for i, st := range streams {
		if err := st.open(); err != nil {
			closePumpFiles()
			collector.Close()
			canonical.abortIncomplete(context.WithoutCancel(ctx), err)
			return res, err
		}
		if i == 0 {
			cmd.Stdout = st.w
		} else {
			cmd.Stderr = st.w
		}
	}
	abortBeforeStart := func(cause error) {
		closePumpFiles()
		collector.Close()
		// Writers currently Begin on their first write. Abort also covers any
		// writer acquired by future preparation; unexecuted output is never committed.
		canonical.abortIncomplete(context.WithoutCancel(ctx), cause)
	}

	if beforeStart != nil {
		beforeStart(runCtx)
	}
	if err := runCtx.Err(); err != nil {
		abortBeforeStart(err)
		res.TimedOut = errors.Is(err, context.DeadlineExceeded)
		return res, err
	}
	startErr := start(cmd)
	for _, st := range streams {
		st.closeWriteEnd() // the child owns its copy; EOF needs ours closed
	}
	if startErr != nil {
		abortBeforeStart(startErr)
		return res, startErr
	}
	processGroup, err := attachProcessGroup(cmd.Process)
	if err != nil {
		// Fail closed: a command that could not be placed inside the platform
		// process-tree boundary must not continue running unsupervised.
		_ = cmd.Process.Kill()
		_ = cmd.Wait()
		closePumpFiles()
		collector.Close()
		canonical.finish(context.WithoutCancel(ctx))
		return res, fmt.Errorf("establish process-tree boundary: %w", err)
	}
	defer processGroup.close()

	var pumps sync.WaitGroup
	for _, st := range streams {
		pumps.Add(1)
		go func() {
			defer pumps.Done()
			st.pump()
		}()
	}
	// waitCh closes once the process is reaped and both streams reached EOF. After
	// the process exits, a descendant may still hold the pipes: the streams then
	// get PipeDrainGrace (not counting time inside a slow consumer Write) before
	// the readers are closed. A cancelled or timed-out run skips that cut-off so
	// the timeout path below keeps waiting for EOF bounded by KillWaitGrace.
	var cmdWaitErr error
	var drainForced atomic.Bool
	waitCh := make(chan struct{})
	pumpsDone := make(chan struct{})
	go func() {
		pumps.Wait()
		close(pumpsDone)
	}()
	go func() {
		cmdWaitErr = cmd.Wait()
		if !waitForPumpsDrain(streams, pumpsDone, runCtx.Done(), opts.PipeDrainGrace, opts.PipeDrainGrace+opts.KillWaitGrace) {
			drainForced.Store(true)
			closePumpFiles()
		}
		<-pumpsDone
		closePumpFiles() // release the read ends deterministically, not via GC
		close(waitCh)
	}()

	var waitErr error
	var rawWaitErr error
	killWaitTimedOut := false
	pipeDrainTimedOut := false
	select {
	case <-waitCh:
		rawWaitErr = cmdWaitErr
		waitErr = rawWaitErr
	case <-runCtx.Done():
		res.TimedOut = errors.Is(runCtx.Err(), context.DeadlineExceeded)
		_ = processGroup.terminate()

		reaped := false
		if opts.KillGrace > 0 {
			timer := time.NewTimer(opts.KillGrace)
			select {
			case <-waitCh:
				rawWaitErr = cmdWaitErr
				reaped = true
				if !timer.Stop() {
					<-timer.C
				}
			case <-timer.C:
			}
		}
		if !reaped {
			_ = processGroup.kill()
			// SIGKILL cannot reach processes that escaped the group (setsid,
			// daemons), and such a process can hold the inherited output
			// descriptors open, which keeps cmd.Wait blocked forever. Bound the
			// post-kill wait so the caller always gets a result.
			killTimer := time.NewTimer(opts.KillWaitGrace)
			select {
			case <-waitCh:
				rawWaitErr = cmdWaitErr
				if !killTimer.Stop() {
					<-killTimer.C
				}
			case <-killTimer.C:
				killWaitTimedOut = true
			}
		}

		if res.TimedOut {
			waitErr = context.DeadlineExceeded
		} else {
			waitErr = runCtx.Err()
		}
	}

	if !killWaitTimedOut {
		// waitCh closed on every path, so this also covers a forced cut-off that
		// raced with a timeout or cancellation: it must never be committed complete.
		pipeDrainTimedOut = drainForced.Load()
	}
	if killWaitTimedOut {
		// cmd.Wait never returned, so the wait goroutine still owns the stream
		// writers and the legacy artifact file must stay open for any straggler
		// writes. Hand the close off so the descriptor is released whenever the
		// straggler finally exits, instead of blocking this call.
		res.KillProcessTimedOut = true
		go func() {
			<-waitCh
			collector.Close()
		}()
	} else {
		collector.Close()
	}
	if killWaitTimedOut {
		canonical.abortIncomplete(context.WithoutCancel(ctx), fmt.Errorf("process wait exceeded %s after tree kill; output streams have not reached EOF", opts.KillWaitGrace))
	} else if pipeDrainTimedOut {
		canonical.abortIncomplete(context.WithoutCancel(ctx), fmt.Errorf("output streams did not reach EOF within %s after the process exited", opts.PipeDrainGrace))
	} else {
		canonical.finish(context.WithoutCancel(ctx))
	}
	snap := collector.snapshot()
	res.Output = snap.preview
	res.OutputBytes = snap.totalBytes
	res.CapturedBytes = len(res.Output)
	res.OutputTruncated = snap.truncated
	res.OutputPath = snap.outputPath
	res.ArtifactBytes = snap.artifactBytes
	res.OutputArtifactErr = snap.artifactErr
	res.OutputArtifacts = canonical.manifests()
	res.OutputArtifactDiagnostics = canonical.diagnostics()
	if canonicalBytes := canonical.artifactBytes(); canonicalBytes > 0 {
		res.ArtifactBytes = canonicalBytes
	}
	if diagnostics := formatArtifactDiagnostics(res.OutputArtifactDiagnostics); diagnostics != "" {
		if strings.TrimSpace(res.OutputArtifactErr) == "" {
			res.OutputArtifactErr = diagnostics
		} else {
			res.OutputArtifactErr = strings.TrimSpace(res.OutputArtifactErr) + "; " + diagnostics
		}
	}
	res.ExitCode = exitCodeFromError(rawWaitErr)
	if pipeDrainTimedOut {
		// The command's own status stands; only the stream capture is incomplete.
		res.PipeDrainTimedOut = true
		res.Output = strings.TrimRight(res.Output, "\n") + "\n[a background process still holds stdout; output after exit was not captured]"
	}
	if killWaitTimedOut {
		// cmd.Wait never returned, so rawWaitErr is nil and must not be read as
		// a clean exit.
		res.ExitCode = -1
		waitErr = fmt.Errorf("%w after %s: %w", ErrProcessKillTimeout, opts.KillWaitGrace, waitErr)
	}

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
	mu         sync.Mutex
	callbackMu sync.Mutex

	limit int
	onOut func(OutputChunk)

	preview bytes.Buffer
	total   int64

	artifactDir     string
	artifactPrefix  string
	artifact        *os.File
	artifactPath    string
	artifactBytes   int64
	artifactErr     string
	primed          bool
	truncated       bool
	disableArtifact bool
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
	return c.writeStream("combined", p)
}

func (c *outputCollector) writeStream(stream string, p []byte) (int, error) {
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
		if c.disableArtifact {
			// Canonical stdout/stderr sinks own complete bytes in this mode. The
			// legacy anonymous combined temp file would be a duplicate object.
		} else if err := c.ensureArtifactLocked(); err != nil {
			if c.artifactErr == "" {
				c.artifactErr = err.Error()
			}
		} else if c.artifact != nil {
			written, err := writeArtifactChunk(c.artifact, p)
			c.artifactBytes += int64(written)
			if err != nil && c.artifactErr == "" {
				c.artifactErr = err.Error()
			}
		}
	}
	callback := c.onOut
	c.mu.Unlock()
	if callback != nil {
		c.callbackMu.Lock()
		callback(OutputChunk{
			Data:       chunk,
			TotalBytes: totalBytes,
			Stream:     stream,
		})
		c.callbackMu.Unlock()
	}
	return n, nil
}

func (c *outputCollector) disableLegacyArtifact() {
	c.mu.Lock()
	c.disableArtifact = true
	c.mu.Unlock()
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
		preview:       boundedUTF8Preview(c.preview.Bytes(), c.limit),
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
	written, werr := writeArtifactChunk(c.artifact, c.preview.Bytes())
	c.artifactBytes += int64(written)
	if werr != nil {
		return werr
	}
	return nil
}

func writeArtifactChunk(w io.Writer, p []byte) (int, error) {
	written, err := w.Write(p)
	if err == nil && written < len(p) {
		err = io.ErrShortWrite
	}
	return written, err
}

func boundedUTF8Preview(raw []byte, limit int) string {
	if limit <= 0 || len(raw) == 0 {
		return ""
	}
	// Command streams are byte-oriented and a cap may split a multi-byte rune.
	// The artifact remains byte-exact; the preview marks undecodable bytes with
	// U+FFFD instead of deleting them, then backs off to a rune boundary so it is
	// always bounded, valid UTF-8 without silently swallowing characters.
	return truncateAtRuneBoundary(
		strings.ToValidUTF8(string(raw), invalidUTF8PreviewReplacement),
		limit,
	)
}

// truncateAtRuneBoundary caps text at maxBytes without splitting a multi-byte
// rune. text is expected to be valid UTF-8, so the backoff is bounded by
// utf8.UTFMax-1 bytes.
func truncateAtRuneBoundary(text string, maxBytes int) string {
	if maxBytes <= 0 {
		return ""
	}
	if len(text) <= maxBytes {
		return text
	}
	cut := maxBytes
	for cut > 0 && !utf8.RuneStart(text[cut]) {
		cut--
	}
	return text[:cut]
}

// outputPump copies one child output stream from an os.Pipe into writer.
type outputPump struct {
	writer io.Writer
	r, w   *os.File
	// writeStart is the UnixNano time the pump entered writer.Write, or 0 while
	// it is not inside the consumer.
	writeStart atomic.Int64
}

const pipeDrainTick = 10 * time.Millisecond

// waitForPumpsDrain is called once the process has exited. It returns true if
// the streams were not cut off: pumpsDone closed, or stop fired. It returns
// false when grace of wall time has passed since the exit, not counting time
// any pump spent inside writer.Write: a slow consumer must not lose output the
// process already wrote, but a descendant that keeps writing (or holds the pipe
// silently) cannot extend the wait. Excluding Write time cannot tell a slow
// consumer from a fast producer, so ceiling is a hard wall-clock limit since the
// exit that no excluded time can exceed.
func waitForPumpsDrain(streams []*outputPump, pumpsDone, stop <-chan struct{}, grace, ceiling time.Duration) bool {
	ticker := time.NewTicker(pipeDrainTick)
	defer ticker.Stop()
	begin := time.Now()
	last := begin
	var spent time.Duration
	for spent < grace && last.Sub(begin) < ceiling {
		select {
		case <-pumpsDone:
			return true
		case <-stop:
			return true
		case now := <-ticker.C:
			inWrite := false
			for _, st := range streams {
				inWrite = inWrite || st.writeStart.Load() != 0
			}
			if !inWrite {
				spent += now.Sub(last)
			}
			last = now
		}
	}
	return false
}

func (p *outputPump) open() (err error) {
	p.r, p.w, err = os.Pipe()
	return err
}

func (p *outputPump) closeWriteEnd() {
	if p.w != nil {
		_ = p.w.Close()
	}
}

func (p *outputPump) closeFiles() {
	p.closeWriteEnd()
	if p.r != nil {
		_ = p.r.Close()
	}
}

func (p *outputPump) pump() {
	buf := make([]byte, 32*1024)
	for {
		n, err := p.r.Read(buf)
		if n > 0 {
			p.writeStart.Store(time.Now().UnixNano())
			_, werr := p.writer.Write(buf[:n])
			p.writeStart.Store(0)
			if werr != nil {
				return
			}
		}
		if err != nil {
			return
		}
	}
}
