package agent

import (
	"context"
	"sync/atomic"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

// A Handler panic followed by a panicking OnPanic must still release the
// host stage exactly once, on both the sequential and the wave path.
func TestToolBlockDoublePanicStillReleasesStage(t *testing.T) {
	for _, parallel := range []bool{false, true} {
		name := map[bool]string{false: "sequential", true: "wave"}[parallel]
		t.Run(name, func(t *testing.T) {
			var releases atomic.Int32
			calls, a := sequentialFixture(2, func(context.Context, int) (any, error) { panic("handler boom") })
			admit := a.Admit
			a.Admit = func(ctx context.Context, i int) (BlockAdmission, error) {
				admission, err := admit(ctx, i)
				admission.Finish = func() bool { releases.Add(1); return false }
				return admission, err
			}
			a.OnPanic = func(int, context.Context, any) (llm.Content, error) { panic("on-panic boom") }
			if parallel {
				a.parallel = &blockParallelism{maxWorkers: 2, eligible: allEligible}
			}
			state, _ := newToolBlockState(calls)
			func() {
				defer func() { _ = recover() }()
				_, _ = runSequentialBlock(context.Background(), state, calls, a, false)
			}()
			want := int32(1)
			if parallel {
				want = 2 // both calls of the wave were admitted and started
			}
			if got := releases.Load(); got != want {
				t.Fatalf("stage releases=%d want %d", got, want)
			}
		})
	}
}
