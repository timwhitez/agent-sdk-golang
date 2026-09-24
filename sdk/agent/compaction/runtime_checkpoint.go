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
