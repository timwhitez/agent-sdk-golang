package compaction

import "errors"

// CheckpointOutcomeUnknown is implemented by a checkpoint writer's error when
// its checkpoint may already be durable despite the reported failure.
// The Agent then retains the ledger, does not publish history, and quarantines
// further checkpoint writes until the host explicitly reconciles the store.
type CheckpointOutcomeUnknown interface {
	CheckpointOutcomeUnknown() bool
}

// maxCheckpointErrorNodes bounds both visited nodes and retained pending nodes.
const maxCheckpointErrorNodes = 256

// CheckpointOutcomeIsUnknown checks wrapped and joined errors conservatively.
// False markers cannot override a positive descendant. Only a completely
// checked tree without a positive marker is negative; exhausting either bound
// returns unknown. The SDK's traversal storage is fixed-size: it never copies
// an arbitrarily wide Unwrap slice before checking its remaining capacity.
// User-defined Error/Unwrap/marker implementations themselves are outside this
// storage bound, just as they are outside the standard errors package's control.
func CheckpointOutcomeIsUnknown(err error) bool {
	var pending [maxCheckpointErrorNodes]error
	pending[0] = err
	count, visited := 1, 0
	for count > 0 {
		if visited == maxCheckpointErrorNodes {
			return true
		}
		visited++
		count--
		node := pending[count]
		pending[count] = nil
		if node == nil {
			continue
		}
		if marker, ok := node.(CheckpointOutcomeUnknown); ok && marker.CheckpointOutcomeUnknown() {
			return true
		}
		switch wrapped := node.(type) {
		case interface{ Unwrap() []error }:
			children := wrapped.Unwrap()
			if len(children) > len(pending)-count {
				return true
			}
			count += copy(pending[count:], children)
		case interface{ Unwrap() error }:
			// Popping a node always leaves room for its one child.
			pending[count] = wrapped.Unwrap()
			count++
		}
	}
	return false
}

// ErrCheckpointStoreQuarantined refuses a new checkpoint after an earlier
// write had an unknown outcome. No new write was attempted. Configuration
// updates do not release quarantine; the host must reconcile the store and
// explicitly call Agent.CheckpointStoreReconciled.
var ErrCheckpointStoreQuarantined = errors.New("compaction: checkpoint store quarantined after an unknown checkpoint outcome; reconcile the store, then release it explicitly")
