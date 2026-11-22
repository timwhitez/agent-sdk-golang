package tools

import (
	"context"
	"sync/atomic"
	"testing"
	"time"
)

func TestContainerOverrideDuringResolution(t *testing.T) {
	c := NewContainer()
	key := Dep[string]("value")
	entered, release, finished := make(chan struct{}), make(chan struct{}), make(chan struct{})
	Provide(c, key, func(context.Context) (string, error) { close(entered); <-release; return "old", nil })
	go func() { defer close(finished); _, _ = Get(c, context.Background(), key) }()
	<-entered
	Override(c, key, func(context.Context) (string, error) { return "new", nil })
	close(release)
	<-finished
	got, err := Get(c, context.Background(), key)
	if err != nil || got != "new" {
		t.Fatalf("Get after Override and old resolution=%q err=%v, want new", got, err)
	}
}

func TestContainerOverrideDetachesOldInflight(t *testing.T) {
	c := NewContainer()
	key := Dep[string]("value")
	oldEntered, oldRelease := make(chan struct{}), make(chan struct{})
	oldDone := make(chan string, 1)
	Provide(c, key, func(context.Context) (string, error) { close(oldEntered); <-oldRelease; return "old", nil })
	go func() { v, _ := Get(c, context.Background(), key); oldDone <- v }()
	<-oldEntered
	newEntered, newRelease := make(chan struct{}), make(chan struct{})
	newDone := make(chan string, 2)
	var calls atomic.Int32
	Override(c, key, func(context.Context) (string, error) {
		if calls.Add(1) == 1 {
			close(newEntered)
		}
		<-newRelease
		return "new", nil
	})
	go func() { v, _ := Get(c, context.Background(), key); newDone <- v }()
	select {
	case <-newEntered:
	case <-time.After(time.Second):
		close(oldRelease)
		close(newRelease)
		t.Fatal("new binding waited for detached provider")
	}
	close(oldRelease)
	if v := <-oldDone; v != "old" {
		t.Fatalf("existing caller = %q", v)
	}
	go func() { v, _ := Get(c, context.Background(), key); newDone <- v }()
	close(newRelease)
	for range 2 {
		if v := <-newDone; v != "new" {
			t.Fatalf("new caller = %q", v)
		}
	}
	if calls.Load() != 1 {
		t.Fatalf("new provider called %d times", calls.Load())
	}
	if v, err := Get(c, context.Background(), key); v != "new" || err != nil {
		t.Fatalf("cached = %q / %v", v, err)
	}
}
