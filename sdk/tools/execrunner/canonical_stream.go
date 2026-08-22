package execrunner

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"hash"
	"io"
	"reflect"
	"strings"
	"sync"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/artifact"
)

type processStreamWriter struct {
	stream    string
	combined  *outputCollector
	canonical *canonicalStreamCapture
}

func (w processStreamWriter) Write(p []byte) (int, error) {
	n, err := w.combined.writeStream(w.stream, p)
	if n > 0 && w.canonical != nil {
		w.canonical.write(p[:n])
	}
	return n, err
}

type canonicalProcessStreams struct {
	requested   bool
	stdout      *canonicalStreamCapture
	stderr      *canonicalStreamCapture
	setupErrors []artifact.Diagnostic
}

func newCanonicalProcessStreams(ctx context.Context, opts Options) *canonicalProcessStreams {
	streams := &canonicalProcessStreams{requested: opts.ArtifactStreamSink != nil}
	if !streams.requested {
		return streams
	}
	if err := opts.ArtifactResolverCapability.Validate(); err != nil {
		streams.setupErrors = append(streams.setupErrors, streamDiagnostic(
			"artifact_stream_capability",
			"register_valid_resolver_capability",
			err,
		))
		return streams
	}
	if !opts.ArtifactResolverCapability.Registered {
		streams.setupErrors = append(streams.setupErrors, streamDiagnostic(
			"artifact_stream_capability",
			"register_resolver_capability",
			fmt.Errorf("host resolver capability is not registered"),
		))
		return streams
	}
	owner := opts.ArtifactOwner
	owner.Stream = ""
	owner.Part = ""
	if err := owner.Validate(); err != nil {
		streams.setupErrors = append(streams.setupErrors, streamDiagnostic(
			"artifact_stream_owner",
			"configure_artifact_owner",
			err,
		))
		return streams
	}
	createdAt := time.Now().UTC()
	streams.stdout = newCanonicalStreamCapture(ctx, opts.ArtifactStreamSink, owner, "stdout", "process_stdout", createdAt, opts.ArtifactResolverCapability.Recovery)
	streams.stderr = newCanonicalStreamCapture(ctx, opts.ArtifactStreamSink, owner, "stderr", "process_stderr", createdAt, opts.ArtifactResolverCapability.Recovery)
	return streams
}

func (s *canonicalProcessStreams) finish(ctx context.Context) {
	if s == nil {
		return
	}
	if s.stdout != nil {
		s.stdout.finish(ctx)
	}
	if s.stderr != nil {
		s.stderr.finish(ctx)
	}
}

func (s *canonicalProcessStreams) manifests() []artifact.Manifest {
	if s == nil {
		return nil
	}
	manifests := make([]artifact.Manifest, 0, 2)
	for _, stream := range []*canonicalStreamCapture{s.stdout, s.stderr} {
		if stream == nil {
			continue
		}
		if manifest, ok := stream.manifestSnapshot(); ok {
			manifests = append(manifests, manifest)
		}
	}
	return manifests
}

func (s *canonicalProcessStreams) diagnostics() []artifact.Diagnostic {
	if s == nil {
		return nil
	}
	diagnostics := append([]artifact.Diagnostic(nil), s.setupErrors...)
	for _, stream := range []*canonicalStreamCapture{s.stdout, s.stderr} {
		if stream != nil {
			diagnostics = append(diagnostics, stream.diagnosticsSnapshot()...)
		}
	}
	return diagnostics
}

func (s *canonicalProcessStreams) artifactBytes() int64 {
	var total int64
	for _, manifest := range s.manifests() {
		if manifest.ObjectMeasurement.Bytes != nil {
			total += *manifest.ObjectMeasurement.Bytes
		}
	}
	return total
}

type canonicalStreamCapture struct {
	mu sync.Mutex

	ctx     context.Context
	sink    artifact.StreamSink
	request artifact.StreamPutRequest

	writer      artifact.StreamObjectWriter
	hash        hash.Hash
	total       int64
	failed      bool
	finished    bool
	manifest    artifact.Manifest
	hasManifest bool
	diagnostics []artifact.Diagnostic
}

func newCanonicalStreamCapture(ctx context.Context, sink artifact.StreamSink, owner artifact.Owner, stream, part string, createdAt time.Time, recovery artifact.Recovery) *canonicalStreamCapture {
	owner.Stream = stream
	owner.Part = part
	return &canonicalStreamCapture{
		ctx:  ctx,
		sink: sink,
		request: artifact.StreamPutRequest{
			ObjectKind: artifact.ObjectKindRawStream,
			Owner:      owner,
			Retention: artifact.Retention{
				Class:     artifact.RetentionDurable,
				CreatedAt: createdAt,
			},
			ContentType: "application/octet-stream",
			Encoding:    "binary",
			Recovery:    cloneStreamRecovery(recovery),
		},
		hash: sha256.New(),
	}
}

func (c *canonicalStreamCapture) write(p []byte) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if len(p) == 0 || c.finished {
		return
	}
	_, _ = c.hash.Write(p)
	c.total += int64(len(p))
	if c.failed {
		return
	}
	if c.writer == nil {
		writer, err := c.sink.Begin(c.ctx, c.request)
		if err != nil {
			c.failLocked("artifact_stream_begin", "check_stream_store_and_retry", err)
			return
		}
		if writer == nil {
			c.failLocked("artifact_stream_begin", "repair_stream_sink_contract", fmt.Errorf("stream sink returned a nil writer"))
			return
		}
		c.writer = writer
	}
	written, err := c.writer.Write(p)
	if err == nil && written != len(p) {
		err = io.ErrShortWrite
	}
	if err != nil {
		c.failLocked("artifact_stream_write", "check_stream_store_and_retry", err)
	}
}

func (c *canonicalStreamCapture) finish(ctx context.Context) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.finished {
		return
	}
	c.finished = true
	if c.total == 0 || c.failed || c.writer == nil {
		return
	}
	manifest, err := c.writer.Commit(ctx)
	if err != nil {
		c.failLocked("artifact_stream_commit", "check_stream_store_and_retry", err)
		return
	}
	if err := validateCanonicalStreamManifest(manifest, c.request, c.total, hex.EncodeToString(c.hash.Sum(nil))); err != nil {
		c.failLocked("artifact_stream_manifest", "repair_stream_sink_manifest", err)
		return
	}
	c.manifest = manifest.Clone()
	c.hasManifest = true
	c.writer = nil
}

func (c *canonicalStreamCapture) failLocked(stage, action string, err error) {
	c.failed = true
	c.diagnostics = append(c.diagnostics, streamDiagnostic(stage, action, err))
	if c.writer != nil {
		if abortErr := c.writer.Abort(c.ctx); abortErr != nil {
			c.diagnostics = append(c.diagnostics, streamDiagnostic(
				"artifact_stream_abort",
				"inspect_stream_store_cleanup",
				abortErr,
			))
		}
		c.writer = nil
	}
}

func (c *canonicalStreamCapture) manifestSnapshot() (artifact.Manifest, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if !c.hasManifest {
		return artifact.Manifest{}, false
	}
	return c.manifest.Clone(), true
}

func (c *canonicalStreamCapture) diagnosticsSnapshot() []artifact.Diagnostic {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]artifact.Diagnostic(nil), c.diagnostics...)
}

func validateCanonicalStreamManifest(manifest artifact.Manifest, request artifact.StreamPutRequest, total int64, digest string) error {
	if err := manifest.Validate(); err != nil {
		return fmt.Errorf("stream sink returned invalid manifest: %w", err)
	}
	if !manifest.Complete || !manifest.Recoverable {
		return fmt.Errorf("stream sink manifest must be complete and recoverable")
	}
	if manifest.ObjectKind != request.ObjectKind {
		return fmt.Errorf("stream sink object_kind mismatch")
	}
	if !reflect.DeepEqual(manifest.Owner, request.Owner) {
		return fmt.Errorf("stream sink owner mismatch")
	}
	if manifest.ObjectMeasurement.Bytes == nil || *manifest.ObjectMeasurement.Bytes != total {
		return fmt.Errorf("stream sink byte count mismatch")
	}
	if manifest.ObjectMeasurement.SHA256 != digest {
		return fmt.Errorf("stream sink sha256 mismatch")
	}
	if manifest.Retention.Class != artifact.RetentionDurable || manifest.Retention.ExpiresAt != nil {
		return fmt.Errorf("stream sink retention must be durable without expires_at")
	}
	if !streamRecoveryEqual(manifest.Recovery, request.Recovery) {
		return fmt.Errorf("stream sink recovery contract mismatch")
	}
	return nil
}

func streamDiagnostic(stage, action string, err error) artifact.Diagnostic {
	message := "artifact stream persistence failed"
	if err != nil && strings.TrimSpace(err.Error()) != "" {
		message = strings.TrimSpace(err.Error())
	}
	return artifact.Diagnostic{
		Severity: "warning",
		Stage:    stage,
		Action:   action,
		Message:  message,
	}
}

func formatArtifactDiagnostics(diagnostics []artifact.Diagnostic) string {
	parts := make([]string, 0, len(diagnostics))
	for _, diagnostic := range diagnostics {
		parts = append(parts, fmt.Sprintf(
			"[WARN] stage=%s action=%s: %s",
			diagnostic.Stage,
			diagnostic.Action,
			diagnostic.Message,
		))
	}
	return strings.Join(parts, "; ")
}

func cloneStreamRecovery(recovery artifact.Recovery) artifact.Recovery {
	out := recovery
	out.AllowedRangeUnits = append([]artifact.RangeUnit(nil), recovery.AllowedRangeUnits...)
	return out
}

func streamRecoveryEqual(left, right artifact.Recovery) bool {
	return left.Capability == right.Capability &&
		left.Tool == right.Tool &&
		left.Instruction == right.Instruction &&
		reflect.DeepEqual(left.AllowedRangeUnits, right.AllowedRangeUnits)
}
