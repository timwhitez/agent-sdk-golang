package tools

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

type panicWaitContext struct {
	context.Context
	entered chan struct{}
	once    sync.Once
}

func (c *panicWaitContext) Done() <-chan struct{} {
	c.once.Do(func() { close(c.entered) })
	return c.Context.Done()
}

func TestAuditDependencyProviderPanicDoesNotPoisonResolution(t *testing.T) {
	c := NewContainer()
	key := Dep[string]("recovering")
	calls := 0
	Provide(c, key, func(context.Context) (string, error) {
		calls++
		if calls == 1 {
			panic("first attempt")
		}
		return "ready", nil
	})
	func() {
		defer func() {
			if got := recover(); got != "first attempt" {
				t.Fatalf("panic=%v", got)
			}
		}()
		_, _ = Get(c, t.Context(), key)
	}()
	ctx, cancel := context.WithTimeout(t.Context(), 100*time.Millisecond)
	defer cancel()
	got, err := Get(c, ctx, key)
	if err != nil || got != "ready" || calls != 2 {
		t.Fatalf("after recovered provider panic: value=%q err=%v provider calls=%d; want fresh successful resolution", got, err, calls)
	}
}

func TestDependencyProviderPanicReleasesWaitersWithoutChangingOverride(t *testing.T) {
	for _, override := range []bool{false, true} {
		t.Run(map[bool]string{false: "retry", true: "override"}[override], func(t *testing.T) {
			c := NewContainer()
			key := Dep[string]("service")
			started, release := make(chan struct{}), make(chan struct{})
			var oldCalls, newCalls atomic.Int32
			Provide(c, key, func(context.Context) (string, error) {
				if oldCalls.Add(1) == 1 {
					close(started)
					<-release
					panic("original provider panic")
				}
				return "retried", nil
			})
			ownerDone := make(chan any, 1)
			go func() {
				defer func() { ownerDone <- recover() }()
				_, _ = Get(c, t.Context(), key)
			}()
			<-started
			ctx, cancel := context.WithTimeout(t.Context(), time.Second)
			defer cancel()
			waitCtx := &panicWaitContext{Context: ctx, entered: make(chan struct{})}
			waiterDone := make(chan error, 1)
			go func() { _, err := Get(c, waitCtx, key); waiterDone <- err }()
			<-waitCtx.entered
			if override {
				Override(c, key, func(context.Context) (string, error) { newCalls.Add(1); return "override", nil })
				if got, err := Get(c, t.Context(), key); err != nil || got != "override" {
					t.Fatalf("override: %q %v", got, err)
				}
			}
			close(release)
			if got := <-ownerDone; got != "original provider panic" {
				t.Fatalf("owner panic changed: %v", got)
			}
			if err := <-waiterDone; err == nil || errors.Is(err, context.DeadlineExceeded) {
				t.Fatalf("waiter did not receive completed failure: %v", err)
			}
			want := "retried"
			if override {
				want = "override"
			}
			if got, err := Get(c, t.Context(), key); err != nil || got != want {
				t.Fatalf("next value=%q err=%v want=%q", got, err, want)
			}
			if override && newCalls.Load() != 1 {
				t.Fatalf("override cache lost: calls=%d", newCalls.Load())
			}
		})
	}
}
