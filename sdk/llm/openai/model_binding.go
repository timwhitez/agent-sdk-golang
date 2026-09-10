package openai

import (
	"context"
	"errors"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

// BindFrameModel captures owned client configuration, not HTTP transport or
// diagnostic closure state. Configure the input before binding, never during it.
func (c *ChatClient) BindFrameModel(ctx context.Context) (llm.ChatModel, bool, error) {
	if err := ctx.Err(); err != nil {
		return nil, false, err
	}
	if c == nil {
		return nil, false, nil
	}
	bound := *c
	var err error
	bound.Extra, bound.ExtraBody, err = bindExtraMaps(c.Extra, c.ExtraBody)
	if err != nil {
		return nil, false, err
	}
	bound.Temperature = bindingValue(c.Temperature)
	bound.TopP = bindingValue(c.TopP)
	bound.Seed = bindingValue(c.Seed)
	bound.MaxCompletionTokens = bindingValue(c.MaxCompletionTokens)
	bound.RetryableStatusCodes = bindingCodes(c.RetryableStatusCodes)
	return &bound, true, nil
}

// BindFrameModel preserves compatibility flags and provider label while owning
// mutable parameters and extras. Custom marshalers cannot provide static intent.
func (c *ResponsesClient) BindFrameModel(ctx context.Context) (llm.ChatModel, bool, error) {
	if err := ctx.Err(); err != nil {
		return nil, false, err
	}
	if c == nil {
		return nil, false, nil
	}
	bound := *c
	var err error
	bound.Extra, bound.ExtraBody, err = bindExtraMaps(c.Extra, c.ExtraBody)
	if err != nil {
		return nil, false, err
	}
	bound.Temperature = bindingValue(c.Temperature)
	bound.TopP = bindingValue(c.TopP)
	bound.Seed = bindingValue(c.Seed)
	bound.MaxOutputTokens = bindingValue(c.MaxOutputTokens)
	bound.RetryableStatusCodes = bindingCodes(c.RetryableStatusCodes)
	return &bound, true, nil
}

func bindExtraMaps(extra, body map[string]any) (map[string]any, map[string]any, error) {
	owned, err := llm.CloneStaticJSONMap(extra)
	if err != nil {
		return nil, nil, errors.New("OpenAI frame configuration extras cannot be bound")
	}
	ownedBody, err := llm.CloneStaticJSONMap(body)
	if err != nil {
		return nil, nil, errors.New("OpenAI frame configuration extras cannot be bound")
	}
	return owned, ownedBody, nil
}

func bindingValue[T any](value *T) *T {
	if value == nil {
		return nil
	}
	copy := *value
	return &copy
}

func bindingCodes(codes map[int]struct{}) map[int]struct{} {
	if codes == nil {
		return nil
	}
	copy := make(map[int]struct{}, len(codes))
	for code := range codes {
		copy[code] = struct{}{}
	}
	return copy
}
