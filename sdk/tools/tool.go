package tools

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

// Tool is an executable capability exposed to the model.
type Tool struct {
	Name        string
	Description string

	// EphemeralKeep controls how many recent outputs are kept in context.
	// 0 means keep all; 1 means keep last 1, etc.
	EphemeralKeep int

	Schema map[string]any

	Handler func(ctx context.Context, args json.RawMessage, deps *Container) (llm.Content, error)
}

func (t Tool) Definition() llm.ToolDefinition {
	return llm.ToolDefinition{
		Name:        t.Name,
		Description: t.Description,
		Parameters:  t.Schema,
		Strict:      true,
	}
}

func (t Tool) Execute(ctx context.Context, argsJSON string, deps *Container) (llm.Content, error) {
	if t.Handler == nil {
		return llm.Content{}, fmt.Errorf("tool %q missing handler", t.Name)
	}
	dec := json.NewDecoder(bytes.NewReader([]byte(argsJSON)))
	dec.DisallowUnknownFields()
	var raw json.RawMessage
	if err := dec.Decode(&raw); err != nil {
		return llm.TextContent(fmt.Sprintf("Error parsing arguments: %v", err)), err
	}
	return t.Handler(ctx, raw, deps)
}

// Func creates a tool from an Args struct and a handler.
// Args should be a struct type with json tags.
func Func[Args any](name, description string, fn func(ctx context.Context, args Args, deps *Container) (any, error)) Tool {
	schema := SchemaFor[Args]()
	return Tool{
		Name:        name,
		Description: description,
		EphemeralKeep: 0,
		Schema:      schema,
		Handler: func(ctx context.Context, raw json.RawMessage, deps *Container) (llm.Content, error) {
			var a Args
			dec := json.NewDecoder(bytes.NewReader(raw))
			dec.DisallowUnknownFields()
			if err := dec.Decode(&a); err != nil {
				return llm.TextContent(fmt.Sprintf("Error parsing arguments: %v", err)), err
			}
			res, err := fn(ctx, a, deps)
			if err != nil {
				return llm.TextContent(fmt.Sprintf("Error: %v", err)), err
			}
			return SerializeResult(res)
		},
	}
}

func (t Tool) WithEphemeralKeep(n int) Tool {
	t.EphemeralKeep = n
	return t
}
