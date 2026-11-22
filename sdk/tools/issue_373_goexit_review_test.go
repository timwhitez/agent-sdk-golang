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

type goexitWaitContext struct {
	context.Context
	entered chan struct{}
	once    sync.Once
}

func (c *goexitWaitContext) Done() <-chan struct{} {
	c.once.Do(func() { close(c.entered) })
	return c.Context.Done()
}

func TestDependencyGoexitReleasesOldWaitersAndPreservesPendingOverride(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()
	c := tools.NewContainer()
	key := tools.Dep[string]("goexit-with-pending-override")
	oldStarted, releaseOld := make(chan struct{}), make(chan struct{})
	var releaseOldOnce sync.Once
	defer releaseOldOnce.Do(func() { close(releaseOld) })
	tools.Provide(c, key, func(context.Context) (string, error) {
		close(oldStarted)
		<-releaseOld
		runtime.Goexit()
		return "unreachable", nil
	})
	ownerDone := make(chan struct{})
	var ownerReturned atomic.Bool
	go func() {
		defer close(ownerDone)
		_, _ = tools.Get(c, ctx, key)
		ownerReturned.Store(true)
	}()
	select {
	case <-oldStarted:
	case <-ctx.Done():
		t.Fatal("old provider did not start")
	}
	oldWaitCtx := &goexitWaitContext{Context: ctx, entered: make(chan struct{})}
	oldWaiterDone := make(chan error, 1)
	go func() { _, err := tools.Get(c, oldWaitCtx, key); oldWaiterDone <- err }()
	select {
	case <-oldWaitCtx.entered:
	case <-ctx.Done():
		t.Fatal("old waiter never joined the pending dependency")
	}

	newStarted, releaseNew := make(chan struct{}, 2), make(chan struct{})
	var releaseNewOnce sync.Once
	defer releaseNewOnce.Do(func() { close(releaseNew) })
	var newCalls atomic.Int32
	tools.Override(c, key, func(context.Context) (string, error) {
		newCalls.Add(1)
		newStarted <- struct{}{}
		<-releaseNew
		return "override", nil
	})
	type result struct {
		value string
		err   error
	}
	newOwnerDone := make(chan result, 1)
	go func() { value, err := tools.Get(c, ctx, key); newOwnerDone <- result{value, err} }()
	select {
	case <-newStarted:
	case <-ctx.Done():
		t.Fatal("override did not start")
	}

	// Unwind the old provider while its replacement is still resolving.
	releaseOldOnce.Do(func() { close(releaseOld) })
	select {
	case <-ownerDone:
	case <-ctx.Done():
		t.Fatal("Goexit provider owner did not unwind")
	}
	if ownerReturned.Load() {
		t.Fatal("dependency resolution swallowed runtime.Goexit")
	}
	select {
	case err := <-oldWaiterDone:
		if err == nil || errors.Is(err, context.DeadlineExceeded) || errors.Is(err, context.Canceled) {
			t.Fatalf("old waiter did not receive the completed provider failure: %v", err)
		}
	case <-ctx.Done():
		t.Fatal("old waiter remained stranded after Goexit")
	}

	newWaitCtx := &goexitWaitContext{Context: ctx, entered: make(chan struct{})}
	newWaiterDone := make(chan result, 1)
	go func() { value, err := tools.Get(c, newWaitCtx, key); newWaiterDone <- result{value, err} }()
	select {
	case <-newWaitCtx.entered:
	case <-newStarted:
		t.Fatal("old Goexit cleanup detached the replacement's inflight resolution")
	case <-ctx.Done():
		t.Fatal("new waiter did not join the override")
	}
	releaseNewOnce.Do(func() { close(releaseNew) })
	for _, done := range []<-chan result{newOwnerDone, newWaiterDone} {
		select {
		case got := <-done:
			if got.err != nil || got.value != "override" {
				t.Fatalf("override result: value=%q err=%v", got.value, got.err)
			}
		case <-ctx.Done():
			t.Fatal("replacement resolution did not finish")
		}
	}
	if got, err := tools.Get(c, ctx, key); err != nil || got != "override" || newCalls.Load() != 1 {
		t.Fatalf("override cache changed: value=%q err=%v provider calls=%d", got, err, newCalls.Load())
	}
}
