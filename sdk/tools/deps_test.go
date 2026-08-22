package tools

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestContainerGetConcurrent(t *testing.T) {
	t.Parallel()

	c := NewContainer()
	key := Dep[int]("answer")

	const goroutines = 10
	gate := make(chan struct{})
	started := make(chan struct{})
	release := make(chan struct{})
	var startedOnce sync.Once
	var calls int32

	Provide(c, key, func(_ context.Context) (int, error) {
		atomic.AddInt32(&calls, 1)
		startedOnce.Do(func() { close(started) })
		<-release
		return 42, nil
	})

	var wg sync.WaitGroup
	results := make([]int, goroutines)
	errs := make([]error, goroutines)
	wg.Add(goroutines)
	for i := 0; i < goroutines; i++ {
		go func(i int) {
			defer wg.Done()
			<-gate
			v, err := Get(c, context.Background(), key)
			results[i] = v
			errs[i] = err
		}(i)
	}

	close(gate)

	select {
	case <-started:
	case <-time.After(2 * time.Second):
		t.Fatal("timeout waiting for provider to start")
	}

	close(release)
	wg.Wait()

	if got := atomic.LoadInt32(&calls); got != 1 {
		t.Fatalf("expected 1 provider call, got %d", got)
	}
	for i, err := range errs {
		if err != nil {
			t.Fatalf("goroutine %d: unexpected error: %v", i, err)
		}
	}
	for i, v := range results {
		if v != 42 {
			t.Fatalf("goroutine %d: expected 42, got %d", i, v)
		}
	}
}

func TestContainerGetConcurrentErrorDoesNotCache(t *testing.T) {
	t.Parallel()

	c := NewContainer()
	key := Dep[string]("value")

	var calls int32
	errBoom := errors.New("boom")

	Provide(c, key, func(_ context.Context) (string, error) {
		if atomic.AddInt32(&calls, 1) == 1 {
			return "", errBoom
		}
		return "ok", nil
	})

	_, err := Get(c, context.Background(), key)
	if err == nil || !errors.Is(err, errBoom) {
		t.Fatalf("expected boom error, got %v", err)
	}
	if got := atomic.LoadInt32(&calls); got != 1 {
		t.Fatalf("expected 1 provider call, got %d", got)
	}

	v, err := Get(c, context.Background(), key)
	if err != nil {
		t.Fatalf("unexpected error on retry: %v", err)
	}
	if v != "ok" {
		t.Fatalf("expected ok, got %q", v)
	}
	if got := atomic.LoadInt32(&calls); got != 2 {
		t.Fatalf("expected 2 provider calls, got %d", got)
	}
}

func TestContainerCloneSnapshotsBindingsAndIsolatesOverrides(t *testing.T) {
	t.Parallel()

	parent := NewContainer()
	resolvedKey := Dep[string]("resolved")
	lazyKey := Dep[string]("lazy")
	lateKey := Dep[string]("late")

	var resolvedCalls atomic.Int32
	Provide(parent, resolvedKey, func(context.Context) (string, error) {
		resolvedCalls.Add(1)
		return "parent-resolved", nil
	})
	Provide(parent, lazyKey, func(context.Context) (string, error) {
		return "parent-lazy", nil
	})
	if got, err := Get(parent, context.Background(), resolvedKey); err != nil || got != "parent-resolved" {
		t.Fatalf("resolve parent: got %q err=%v", got, err)
	}

	child := parent.Clone()
	if child == nil || child == parent {
		t.Fatalf("Clone() = %p, want a distinct non-nil container", child)
	}
	if !Has(parent, resolvedKey) || !Has(child, resolvedKey) || !Has(child, lazyKey) {
		t.Fatal("cloned container did not preserve provider bindings")
	}
	if got, err := Get(child, context.Background(), resolvedKey); err != nil || got != "parent-resolved" {
		t.Fatalf("resolve cloned cached value: got %q err=%v", got, err)
	}
	if got := resolvedCalls.Load(); got != 1 {
		t.Fatalf("resolved provider calls = %d, want cached snapshot call count 1", got)
	}

	Override(child, resolvedKey, func(context.Context) (string, error) { return "child", nil })
	if got, err := Get(child, context.Background(), resolvedKey); err != nil || got != "child" {
		t.Fatalf("resolve child override: got %q err=%v", got, err)
	}
	if got, err := Get(parent, context.Background(), resolvedKey); err != nil || got != "parent-resolved" {
		t.Fatalf("child override changed parent: got %q err=%v", got, err)
	}

	Provide(parent, lateKey, func(context.Context) (string, error) { return "late", nil })
	if Has(child, lateKey) {
		t.Fatal("clone observed a provider added to the parent after snapshot")
	}
	if got, err := Get(child, context.Background(), lazyKey); err != nil || got != "parent-lazy" {
		t.Fatalf("resolve cloned lazy provider: got %q err=%v", got, err)
	}
}

func TestContainerCloneConcurrentOverridesDoNotDriftOwners(t *testing.T) {
	t.Parallel()

	parent := NewContainer()
	ownerKey := Dep[string]("artifact-owner")
	Provide(parent, ownerKey, func(context.Context) (string, error) { return "parent", nil })

	const children = 24
	var wg sync.WaitGroup
	errs := make(chan error, children)
	for i := 0; i < children; i++ {
		i := i
		wg.Add(1)
		go func() {
			defer wg.Done()
			child := parent.Clone()
			want := fmt.Sprintf("child-%d", i)
			Override(child, ownerKey, func(context.Context) (string, error) { return want, nil })
			for attempt := 0; attempt < 20; attempt++ {
				got, err := Get(child, context.Background(), ownerKey)
				if err != nil || got != want {
					errs <- fmt.Errorf("child %d attempt %d: got %q err=%v", i, attempt, got, err)
					return
				}
			}
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Error(err)
	}
	if got, err := Get(parent, context.Background(), ownerKey); err != nil || got != "parent" {
		t.Fatalf("parent owner drifted: got %q err=%v", got, err)
	}
}
