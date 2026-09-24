package agent

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/agent/compaction"
)

func wrapN(err error, n int) error {
	for i := 0; i < n; i++ {
		err = fmt.Errorf("w%d: %w", i, err)
	}
	return err
}

func joinWith(first error, plain int, last error) error {
	errs := []error{}
	if first != nil {
		errs = append(errs, first)
	}
	for i := 0; i < plain; i++ {
		errs = append(errs, fmt.Errorf("plain %d", i))
	}
	if last != nil {
		errs = append(errs, last)
	}
	return errors.Join(errs...)
}

// selfCycle unwraps to itself: a pathological tree the walk cannot finish.
type selfCycle struct{}

func (e *selfCycle) Error() string { return "cycle" }
func (e *selfCycle) Unwrap() error { return e }

// #174: the walk stays bounded, but a tree it could not finish checking is
// never reported as "no unknown": only a completely checked tree without a
// positive marker is negative.
func TestQ04BoundedWalkNeverFailsOpen(t *testing.T) {
	positive := indeterminateAppend{}
	plain := errors.New("plain")
	for _, tc := range []struct {
		name string
		err  error
		want bool
	}{
		// Depth: n wraps + the leaf are n+1 nodes; the bound is 256 nodes.
		{"254 wraps over plain (255 nodes, complete)", wrapN(plain, 254), false},
		{"255 wraps over plain (256 nodes, complete)", wrapN(plain, 255), false},
		{"256 wraps over plain (257 nodes, unfinished)", wrapN(plain, 256), true},
		{"255 wraps over positive", wrapN(positive, 255), true},
		{"256 wraps over positive", wrapN(positive, 256), true},
		{"300 wraps over positive", wrapN(positive, 300), true},
		// Width: the join plus its children.
		{"join of 254 plain (255 nodes, complete)", joinWith(nil, 254, nil), false},
		{"join of 255 plain (256 nodes, complete)", joinWith(nil, 255, nil), false},
		{"join of 256 plain (257 nodes, unfinished)", joinWith(nil, 256, nil), true},
		{"join positive first, then 256 plain", joinWith(positive, 256, nil), true},
		{"join 256 plain, then positive", joinWith(nil, 256, positive), true},
		// Cycles never finish; a positive next to one is not hidden.
		{"cycle next to positive", errors.Join(positive, &selfCycle{}), true},
		{"positive after cycle", errors.Join(&selfCycle{}, positive), true},
		{"cycle alone", &selfCycle{}, true},
		// Small complete negatives stay compatible.
		{"plain", plain, false},
		{"wrapped plain", wrapN(plain, 3), false},
		{"nil", nil, false},
	} {
		if got := compaction.CheckpointOutcomeIsUnknown(tc.err); got != tc.want {
			t.Errorf("%s: got %v want %v", tc.name, got, tc.want)
		}
	}
}

// #174 through the real CommitCompactionHistory: a deep or wide error tree
// with a positive marker beyond the walk bound keeps the ledger, publishes
// no history and is not written again.
func TestQ04DeepAndWideUnknownThroughCommit(t *testing.T) {
	for name, fail := range map[string]error{
		"deep":       wrapN(indeterminateAppend{}, 300),
		"wide first": joinWith(indeterminateAppend{}, 300, nil),
		"wide last":  joinWith(nil, 300, indeterminateAppend{}),
	} {
		t.Run(name, func(t *testing.T) {
			writer := &outcomeWriter{fail: func(n int) error {
				if n == 1 {
					return fail
				}
				return nil
			}}
			store := &checkpointLedgerStore{}
			ag := outcomeAgent(t, writer, store, nil)
			source := ag.Messages()
			if _, out, err := commitCandidate(t, ag); err == nil || out.Compacted || !reflect.DeepEqual(source, ag.Messages()) {
				t.Fatalf("first commit: err=%v compacted=%v", err, out.Compacted)
			}
			if ledger, _ := store.snapshot(); ledger == nil || ledger.Summary == nil {
				t.Fatal("ledger rolled back")
			}
			if _, _, err := commitCandidate(t, ag); !errors.Is(err, compaction.ErrCheckpointStoreQuarantined) || writer.count() != 1 {
				t.Fatalf("second commit: err=%v writes=%d", err, writer.count())
			}
		})
	}
	_ = context.Background
}
