package compaction

import (
	"errors"
	"fmt"
	"testing"
)

type outcomeTestMarker struct {
	positive bool
	child    error
}

func (m outcomeTestMarker) Error() string                  { return "marker" }
func (m outcomeTestMarker) Unwrap() error                  { return m.child }
func (m outcomeTestMarker) CheckpointOutcomeUnknown() bool { return m.positive }

type outcomeTestFanout []error

func (m outcomeTestFanout) Error() string   { return "fanout" }
func (m outcomeTestFanout) Unwrap() []error { return m }

type outcomeTestCycle struct{}

func (m *outcomeTestCycle) Error() string { return "cycle" }
func (m *outcomeTestCycle) Unwrap() error { return m }

func TestCheckpointOutcomeBoundedTree(t *testing.T) {
	plain := errors.New("not written")
	positive := outcomeTestMarker{positive: true}
	wrap := func(n int, leaf error) error {
		for i := 0; i < n; i++ {
			leaf = fmt.Errorf("wrapped: %w", leaf)
		}
		return leaf
	}
	for _, tc := range []struct {
		name string
		err  error
		want bool
	}{
		{"nil", nil, false},
		{"plain", plain, false},
		{"positive", positive, true},
		{"false wrapping positive", outcomeTestMarker{child: positive}, true},
		{"joined positive first", errors.Join(positive, plain), true},
		{"joined positive last", errors.Join(plain, positive), true},
		{"255 nodes complete", wrap(254, plain), false},
		{"256 nodes complete", wrap(255, plain), false},
		{"257 nodes unknown", wrap(256, plain), true},
		{"positive beyond bound", wrap(300, positive), true},
		{"cycle", &outcomeTestCycle{}, true},
		{"cycle and positive", errors.Join(positive, &outcomeTestCycle{}), true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := CheckpointOutcomeIsUnknown(tc.err); got != tc.want {
				t.Fatalf("got %v, want %v", got, tc.want)
			}
		})
	}
	for _, n := range []int{254, 255, 256, 257, 100_000} {
		children := make(outcomeTestFanout, n)
		for i := range children {
			children[i] = plain
		}
		if got, want := CheckpointOutcomeIsUnknown(children), n+1 > maxCheckpointErrorNodes; got != want {
			t.Fatalf("width %d: got %v want %v", n, got, want)
		}
	}
}

// The preallocated error tree belongs to the writer. Traversal must not copy
// its 100,000 children into an SDK-owned slice before noticing the bound.
// The generous allocation ceiling allows a fixed stack to escape on a future
// compiler, while the former append-before-check implementation exceeds it.
func TestCheckpointOutcomeWideTreeHasBoundedAllocation(t *testing.T) {
	children := make(outcomeTestFanout, 100_000)
	for i := range children {
		children[i] = errors.New("plain")
	}
	var root error = children
	bench := testing.Benchmark(func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			if !CheckpointOutcomeIsUnknown(root) {
				b.Fatal("unfinished tree was treated as negative")
			}
		}
	})
	if bytes := bench.AllocedBytesPerOp(); bytes > 16<<10 {
		t.Fatalf("traversal allocated %d bytes/op, want a fixed <=16 KiB bound", bytes)
	}
}
