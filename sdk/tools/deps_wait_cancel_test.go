package tools

import (
	"context"
	"errors"
	"fmt"
	"testing"
	"time"
)

func TestContainerDependencyWaiterCancellation(t *testing.T) {
	for _, cancelEarly := range []bool{true, false} {
		t.Run(fmt.Sprint(cancelEarly), func(t *testing.T) {
			c := NewContainer()
			key := Dep[string]("value")
			entered, release, finished := make(chan struct{}), make(chan struct{}), make(chan struct{})
			Provide(c, key, func(context.Context) (string, error) { close(entered); <-release; return "old", nil })
			go func() { defer close(finished); _, _ = Get(c, context.Background(), key) }()
			<-entered
			defer func() { close(release); <-finished }()
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			if cancelEarly {
				cancel()
			}
			waiter := make(chan error, 1)
			go func() { _, err := Get(c, ctx, key); waiter <- err }()
			if !cancelEarly {
				cancel()
			}
			select {
			case err := <-waiter:
				if !errors.Is(err, context.Canceled) {
					t.Fatalf("Get error=%v, want canceled", err)
				}
			case <-time.After(150 * time.Millisecond):
				t.Fatal("canceled waiter still blocked behind another caller's provider")
			}
			if ctx.Err() == nil {
				t.Fatal("waiter did not cancel")
			}
		})
	}
}
