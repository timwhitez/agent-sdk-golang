package openai

import (
	"context"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

// Private per-invocation barriers exercise the actual parser-to-delivery
// boundary without changing client configuration or global forwarding state.
type streamForwardTestHookKey struct{}
type streamForwardTestHooks struct {
	beforeDelivery func(llm.StreamEvent, chan<- llm.StreamEvent)
	dropped        func(llm.StreamEvent)
}

// forwardOpenAIStreamEvents owns the public stream channel. Before
// cancellation it preserves backpressure. After cancellation it drains and
// drops internal events until the provider producer closes, guaranteeing that
// a producer blocked on its bounded queue can exit and release the HTTP body.
func forwardOpenAIStreamEvents(ctx context.Context, dst chan<- llm.StreamEvent, src <-chan llm.StreamEvent) {
	defer close(dst)
	outcome := llm.BeginStreamOutcome(ctx)
	defer outcome.Finish()
	hooks, _ := ctx.Value(streamForwardTestHookKey{}).(*streamForwardTestHooks)
	dropping := false
	for event := range src {
		// Publication precedes both cancellation selection and drain-and-drop.
		outcome.Observe(event)
		if hooks != nil && hooks.beforeDelivery != nil {
			hooks.beforeDelivery(event, dst)
		}
		if dropping {
			if hooks != nil && hooks.dropped != nil {
				hooks.dropped(event)
			}
			continue
		}
		select {
		case dst <- event:
		case <-ctx.Done():
			dropping = true
			if hooks != nil && hooks.dropped != nil {
				hooks.dropped(event)
			}
		}
	}
}
