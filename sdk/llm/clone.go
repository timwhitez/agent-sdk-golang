package llm

import (
	"encoding/json"
	"fmt"
)

// CloneInvokeRequest returns a deep copy of a provider request. The clone owns
// all mutable message, tool-schema, option, slice, map, and pointer state.
func CloneInvokeRequest(request InvokeRequest) (InvokeRequest, error) {
	out := request
	out.Messages = CloneMessages(request.Messages)
	if request.Tools != nil {
		out.Tools = make([]ToolDefinition, len(request.Tools))
		for i, definition := range request.Tools {
			parameters, err := cloneJSONMap(definition.Parameters)
			if err != nil {
				return InvokeRequest{}, fmt.Errorf("clone tool %q parameters: %w", definition.Name, err)
			}
			out.Tools[i] = definition
			out.Tools[i].Parameters = parameters
		}
	}
	if request.Temperature != nil {
		value := *request.Temperature
		out.Temperature = &value
	}
	if request.Responses != nil {
		responses, err := cloneResponsesOptions(request.Responses)
		if err != nil {
			return InvokeRequest{}, err
		}
		out.Responses = responses
	}
	return out, nil
}

func cloneResponsesOptions(options *ResponsesOptions) (*ResponsesOptions, error) {
	if options == nil {
		return nil, nil
	}
	out := *options
	out.UseResponseItems = cloneBool(options.UseResponseItems)
	out.UseInstructions = cloneBool(options.UseInstructions)
	out.ParallelToolCalls = cloneBool(options.ParallelToolCalls)
	out.Store = cloneBool(options.Store)
	out.Include = append([]string(nil), options.Include...)
	if options.Text != nil {
		text := *options.Text
		if options.Text.Format != nil {
			format := *options.Text.Format
			schema, err := cloneJSONMap(options.Text.Format.Schema)
			if err != nil {
				return nil, fmt.Errorf("clone responses text schema: %w", err)
			}
			format.Schema = schema
			text.Format = &format
		}
		out.Text = &text
	}
	if options.Reasoning != nil {
		reasoning := *options.Reasoning
		out.Reasoning = &reasoning
	}
	outputSchema, err := cloneJSONMap(options.OutputSchema)
	if err != nil {
		return nil, fmt.Errorf("clone responses output schema: %w", err)
	}
	out.OutputSchema = outputSchema
	return &out, nil
}

func cloneBool(value *bool) *bool {
	if value == nil {
		return nil
	}
	out := *value
	return &out
}

func cloneJSONMap(value map[string]any) (map[string]any, error) {
	if value == nil {
		return nil, nil
	}
	data, err := json.Marshal(value)
	if err != nil {
		return nil, err
	}
	var out map[string]any
	if err := json.Unmarshal(data, &out); err != nil {
		return nil, err
	}
	return out, nil
}

// CloneMessages returns a deep copy of a conversation history. The returned
// graph owns every mutable slice and pointee carried by Message, so callers may
// modify it without changing the source history.
func CloneMessages(messages []Message) []Message {
	if messages == nil {
		return nil
	}
	out := make([]Message, len(messages))
	for i := range messages {
		out[i] = CloneMessage(messages[i])
	}
	return out
}

// CloneMessage returns a deep copy of one provider-neutral message.
func CloneMessage(message Message) Message {
	out := message
	out.Content = CloneContent(message.Content)
	if message.ToolCalls != nil {
		out.ToolCalls = make([]ToolCall, len(message.ToolCalls))
		for i := range message.ToolCalls {
			out.ToolCalls[i] = CloneToolCall(message.ToolCalls[i])
		}
	}
	return out
}

// CloneProviderState returns a deep copy of opaque provider history.
func CloneProviderState(state []ProviderState) []ProviderState {
	if state == nil {
		return nil
	}
	out := make([]ProviderState, len(state))
	for i := range state {
		out[i] = state[i]
		out[i].Data = append([]byte(nil), state[i].Data...)
	}
	return out
}

// CloneContent returns a deep copy of message content and its pointer-bearing
// blocks.
func CloneContent(content Content) Content {
	out := content
	if content.Blocks == nil {
		return out
	}
	out.Blocks = make([]ContentBlock, len(content.Blocks))
	for i, block := range content.Blocks {
		out.Blocks[i] = block
		if block.ImageURL != nil {
			image := *block.ImageURL
			out.Blocks[i].ImageURL = &image
		}
		if block.Source != nil {
			source := *block.Source
			out.Blocks[i].Source = &source
		}
	}
	return out
}

// CloneToolCall returns a deep copy of one tool call, including provider-owned
// opaque byte signatures.
func CloneToolCall(call ToolCall) ToolCall {
	out := call
	if call.ThoughtSig != nil {
		out.ThoughtSig = append([]byte(nil), call.ThoughtSig...)
	}
	return out
}
