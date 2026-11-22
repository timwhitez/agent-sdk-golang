package tools

import (
	"context"
	"io"
	"testing"
)

func TestContainerMemoizedNilInterface(t *testing.T) {
	c := NewContainer()
	key := Dep[io.Reader]("optional-reader")
	calls := 0
	Provide(c, key, func(context.Context) (io.Reader, error) { calls++; return nil, nil })
	first, e1 := Get(c, context.Background(), key)
	second, e2 := Get(c, context.Background(), key)
	if first != nil || second != nil || e1 != nil || e2 != nil || calls != 1 {
		t.Fatalf("first=%v/%v; cached=%v/%v; provider calls=%d", first, e1, second, e2, calls)
	}
}

func TestContainerClonedNilInterfaceRetainsBindingType(t *testing.T) {
	c := NewContainer()
	key := Dep[io.Reader]("optional-reader")
	Provide(c, key, func(context.Context) (io.Reader, error) { return nil, nil })
	if _, err := Get(c, context.Background(), key); err != nil {
		t.Fatal(err)
	}
	clone := c.Clone()
	if got, err := Get(clone, context.Background(), key); got != nil || err != nil {
		t.Fatalf("clone: %v / %v", got, err)
	}
	if _, err := Get(clone, context.Background(), Dep[io.Writer](key.Name)); err == nil {
		t.Fatal("nil cache accepted incompatible binding type")
	}
}
