package compaction

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

const CompactionCheckpointSchemaVersion = 1

// CompactionCheckpoint is a runtime-neutral, replayable history checkpoint.
// Hosts persist it before the Agent replaces in-memory history.
type CompactionCheckpoint struct {
	SchemaVersion int           `json:"schema_version"`
	CheckpointID  string        `json:"checkpoint_id"`
	Messages      []llm.Message `json:"messages"`
	Result        Result        `json:"result"`
}

type CompactionCheckpointWriter interface {
	SaveCompactionCheckpoint(context.Context, CompactionCheckpoint) error
}

type CompactionCheckpointWriterFunc func(context.Context, CompactionCheckpoint) error

func (f CompactionCheckpointWriterFunc) SaveCompactionCheckpoint(ctx context.Context, checkpoint CompactionCheckpoint) error {
	if f == nil {
		return nil
	}
	return f(ctx, checkpoint)
}

func NewCompactionCheckpoint(messages []llm.Message, res Result) (CompactionCheckpoint, error) {
	checkpoint := CompactionCheckpoint{
		SchemaVersion: CompactionCheckpointSchemaVersion,
		Messages:      llm.CloneMessages(messages),
		Result:        cloneCheckpointResult(res),
	}
	checkpoint.Result.CheckpointID = ""
	checkpoint.Result.CheckpointMessages = len(checkpoint.Messages)
	id, err := compactionCheckpointID(checkpoint)
	if err != nil {
		return CompactionCheckpoint{}, err
	}
	checkpoint.CheckpointID = id
	checkpoint.Result.CheckpointID = id
	return checkpoint, nil
}

func (c CompactionCheckpoint) Validate() error {
	if c.SchemaVersion != CompactionCheckpointSchemaVersion {
		return fmt.Errorf("unsupported compaction checkpoint schema_version %d", c.SchemaVersion)
	}
	if strings.TrimSpace(c.CheckpointID) == "" {
		return fmt.Errorf("compaction checkpoint_id is required")
	}
	if len(c.Messages) == 0 {
		return fmt.Errorf("compaction checkpoint messages are required")
	}
	if !c.Result.Compacted {
		return fmt.Errorf("compaction checkpoint result must be compacted")
	}
	if resultID := strings.TrimSpace(c.Result.CheckpointID); resultID != "" && resultID != strings.TrimSpace(c.CheckpointID) {
		return fmt.Errorf("compaction checkpoint result checkpoint_id mismatch")
	}
	if c.Result.CheckpointMessages != len(c.Messages) {
		return fmt.Errorf("compaction checkpoint message count mismatch")
	}
	got, err := compactionCheckpointID(c)
	if err != nil {
		return err
	}
	if got != strings.TrimSpace(c.CheckpointID) {
		return fmt.Errorf("compaction checkpoint hash mismatch")
	}
	return nil
}

func compactionCheckpointID(checkpoint CompactionCheckpoint) (string, error) {
	checkpoint.CheckpointID = ""
	checkpoint.Result.CheckpointID = ""
	b, err := json.Marshal(checkpoint)
	if err != nil {
		return "", fmt.Errorf("encode compaction checkpoint identity: %w", err)
	}
	return ContentHash(string(b)), nil
}

func cloneCheckpointResult(res Result) Result {
	out := res
	out.Usage = cloneUsage(res.Usage)
	out.TiersApplied = append([]string(nil), res.TiersApplied...)
	out.Warnings = append([]string(nil), res.Warnings...)
	out.pendingLedger = nil
	out.previousLedger = nil
	return out
}

// CompactionCheckpointOutcome is the Agent's final report for one checkpoint
// its CompactionCheckpointWriter acknowledged (SaveCompactionCheckpoint
// returned nil). Exactly one outcome is reported per acknowledged checkpoint
// whose publication the Agent owns (every path except the checkpoint-only
// Agent.CommitCompactionCheckpoint, whose caller owns publication).
type CompactionCheckpointOutcome struct {
	// CheckpointID is the acknowledged checkpoint's content identity.
	CheckpointID string
	// Published reports that the Agent installed history whose first Messages
	// messages have exactly the checkpoint's JSON identity. It is reported
	// before the matching CompactionEvent, if any, is emitted.
	//
	// When false, the checkpoint was abandoned after the acknowledgement: no
	// history carrying it was installed, the live history is the one the
	// Agent had before, and no CompactionEvent follows for it. A later
	// compaction, if any, writes a new checkpoint.
	Published bool
	// Messages is the checkpoint's message count (Result.CheckpointMessages).
	Messages int
	// Reason is a diagnostic for an abandoned checkpoint; empty when published.
	Reason string
}

// CompactionCheckpointSettler is optionally implemented by a
// CompactionCheckpointWriter that must learn the outcome of a checkpoint it
// acknowledged, for example because the acknowledgement made the checkpoint
// durable and replayable. The Agent calls it synchronously on the goroutine
// that published or abandoned the checkpoint, without holding its history
// lock. An error is reported through the Agent's warning sink; it cannot undo
// the reported outcome, so the settler must fail closed on its own side.
type CompactionCheckpointSettler interface {
	SettleCompactionCheckpoint(context.Context, CompactionCheckpointOutcome) error
}
