package compaction

import (
	"context"
	"encoding/json"
	"errors"
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

// CheckpointOutcomeUnknown is implemented by a checkpoint writer's error when
// the checkpoint may already be durable although the write reported failure
// (for example an append whose state is indeterminate). The Agent then never
// rolls back the ledger, retries the checkpoint or publishes history for it.
type CheckpointOutcomeUnknown interface {
	CheckpointOutcomeUnknown() bool
}

// CheckpointOutcomeIsUnknown reports whether any error in err's tree (every
// error it wraps or joins) declares an unknown checkpoint outcome. A marker
// returning false speaks only for its own node and never hides a positive
// one below or beside it. A plain error does not declare one: the writer is
// the only party that knows whether its store may have been changed.
//
// The walk is bounded (maxCheckpointErrorNodes). A tree it cannot finish
// checking within the bound, including a cyclic one, is reported as unknown:
// only a completely checked tree without a positive marker is negative, so a
// positive can never be hidden by depth, width or join order.
func CheckpointOutcomeIsUnknown(err error) bool {
	pending := []error{err}
	for visited := 0; len(pending) > 0; visited++ {
		if visited >= maxCheckpointErrorNodes || len(pending) > maxCheckpointErrorNodes {
			return true // not completely checked: cannot confirm "not written"
		}
		node := pending[len(pending)-1]
		pending = pending[:len(pending)-1]
		if node == nil {
			continue
		}
		if marker, ok := node.(CheckpointOutcomeUnknown); ok && marker.CheckpointOutcomeUnknown() {
			return true
		}
		switch wrapped := node.(type) {
		case interface{ Unwrap() []error }:
			pending = append(pending, wrapped.Unwrap()...)
		case interface{ Unwrap() error }:
			pending = append(pending, wrapped.Unwrap())
		}
	}
	return false
}

// maxCheckpointErrorNodes bounds CheckpointOutcomeIsUnknown's walk.
const maxCheckpointErrorNodes = 256

// ErrCheckpointStoreQuarantined refuses a checkpoint write after an earlier
// checkpoint write of the Agent had an unknown outcome. No write was
// attempted. The host must reconcile its store and call
// Agent.CheckpointStoreReconciled before checkpoints are written again;
// configuration updates do not release it.
var ErrCheckpointStoreQuarantined = errors.New("compaction: checkpoint store quarantined after an unknown checkpoint outcome; reconcile the store, then release it explicitly")
