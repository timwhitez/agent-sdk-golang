package tools_test

import (
	"context"
	"errors"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

type round2WaitContext struct {
	context.Context
	entered chan struct{}
	once    sync.Once
}

func (c *round2WaitContext) Done() <-chan struct{} {
	c.once.Do(func() { close(c.entered) })
	return c.Context.Done()
}

func TestRound2DependencyUnwindReleasesWaitersAndPreservesOverride(t *testing.T) {
	for _, mode := range []string{"body_goexit", "deferred_goexit", "deferred_panic"} {
		for _, override := range []bool{false, true} {
			name := mode
			if override {
				name += "_override"
			}
			t.Run(name, func(t *testing.T) {
				c := tools.NewContainer()
				key := tools.Dep[string]("exiting")
				started, release, ownerDone := make(chan struct{}), make(chan struct{}), make(chan any, 1)
				var oldCalls, newCalls atomic.Int32
				tools.Provide(c, key, func(context.Context) (string, error) {
					if oldCalls.Add(1) != 1 {
						return "retried", nil
					}
					close(started)
					<-release
					switch mode {
					case "deferred_panic":
						defer func() { panic("deferred provider panic") }()
						return "must not cache", nil
					case "deferred_goexit":
						defer runtime.Goexit()
						return "must not cache", nil
					}
					runtime.Goexit()
					return "unreachable", nil
				})
				go func() {
					defer func() { ownerDone <- recover() }()
					_, _ = tools.Get(c, t.Context(), key)
				}()
				<-started
				ctx, cancel := context.WithTimeout(t.Context(), time.Second)
				defer cancel()
				waiting := &round2WaitContext{Context: ctx, entered: make(chan struct{})}
				waiterDone := make(chan error, 1)
				go func() { _, err := tools.Get(c, waiting, key); waiterDone <- err }()
				<-waiting.entered
				if override {
					tools.Override(c, key, func(context.Context) (string, error) {
						newCalls.Add(1)
						return "override", nil
					})
					if value, err := tools.Get(c, t.Context(), key); err != nil || value != "override" {
						t.Fatalf("new binding value=%q err=%v", value, err)
					}
				}
				close(release)
				gotPanic := <-ownerDone
				if mode == "deferred_panic" && gotPanic != "deferred provider panic" || mode != "deferred_panic" && gotPanic != nil {
					t.Fatalf("owner control flow changed: panic=%v", gotPanic)
				}
				if err := <-waiterDone; err == nil || errors.Is(err, context.DeadlineExceeded) {
					t.Fatalf("waiter did not receive retired call failure: %v", err)
				}
				want := "retried"
				if override {
					want = "override"
				}
				if value, err := tools.Get(c, ctx, key); err != nil || value != want {
					t.Fatalf("post-Goexit value=%q err=%v; want %q", value, err, want)
				}
				if override && newCalls.Load() != 1 {
					t.Fatalf("detached unwind changed replacement cache: calls=%d", newCalls.Load())
				}
			})
		}
	}
}
