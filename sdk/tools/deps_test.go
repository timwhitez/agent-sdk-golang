package tools

import (
	"context"
	"errors"
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
