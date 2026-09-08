package anthropic

import (
	"context"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

// BindFrameModel owns this client's semantic configuration. HTTPClient and
// Warningf remain runtime handles, not cloned transports or closure state.
// Configure the source before use; this is not synchronization for concurrent
// writes to the source's exported fields.
func (c *Client) BindFrameModel(ctx context.Context) (llm.ChatModel, bool, error) {
	if err := ctx.Err(); err != nil {
		return nil, false, err
	}
	if c == nil {
		// A nil embedded client may promote this method on an otherwise valid
		// outer model. There is no client configuration to bind in that case.
		return nil, false, nil
	}
	bound := *c
	bound.Temperature = copyBindingValue(c.Temperature)
	bound.TopP = copyBindingValue(c.TopP)
	bound.Seed = copyBindingValue(c.Seed)
	bound.ThinkingBudgetTokens = copyBindingValue(c.ThinkingBudgetTokens)
	if c.Beta != nil {
		bound.Beta = append([]string{}, c.Beta...)
	}
	if c.RetryableStatusCodes != nil {
		bound.RetryableStatusCodes = make(map[int]struct{}, len(c.RetryableStatusCodes))
		for code := range c.RetryableStatusCodes {
			bound.RetryableStatusCodes[code] = struct{}{}
		}
	}
	return &bound, true, nil
}

func copyBindingValue[T any](value *T) *T {
	if value == nil {
		return nil
	}
	copy := *value
	return &copy
}
