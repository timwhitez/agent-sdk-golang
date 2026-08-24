package openai

import (
	"context"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

// forwardOpenAIStreamEvents owns the public stream channel. Before
// cancellation it preserves backpressure. After cancellation it drains and
// drops internal events until the provider producer closes, guaranteeing that
// a producer blocked on its bounded queue can exit and release the HTTP body.
func forwardOpenAIStreamEvents(ctx context.Context, dst chan<- llm.StreamEvent, src <-chan llm.StreamEvent) {
	defer close(dst)
	dropping := false
	for event := range src {
		if dropping {
			continue
		}
		select {
		case dst <- event:
		case <-ctx.Done():
			dropping = true
		}
	}
}
