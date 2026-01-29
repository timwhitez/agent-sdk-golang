package tools

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"strings"

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
	s := strings.TrimSpace(argsJSON)
	if s == "" {
		return t.Handler(ctx, json.RawMessage([]byte(`{}`)), deps)
	}
	dec := json.NewDecoder(bytes.NewReader([]byte(s)))
	dec.DisallowUnknownFields()
	var raw json.RawMessage
	if err := dec.Decode(&raw); err == nil {
		return t.Handler(ctx, raw, deps)
	}

	// Fallback: some providers/gateways occasionally produce non-JSON tool arguments
	// (e.g. a bare string like "/path"). Best-effort repair for common tools.
	// If repair fails, surface the original parse error.
	repaired, ok := repairToolArgs(t.Name, s)
	if ok {
		return t.Handler(ctx, json.RawMessage(repaired), deps)
	}

	// Preserve the original parsing error message for debuggability.
	dec2 := json.NewDecoder(bytes.NewReader([]byte(s)))
	dec2.DisallowUnknownFields()
	var raw2 json.RawMessage
	if err2 := dec2.Decode(&raw2); err2 != nil {
		return llm.TextContent(fmt.Sprintf("Error parsing arguments: %v", err2)), err2
	}
	return t.Handler(ctx, raw2, deps)
}

func repairToolArgs(toolName string, raw string) ([]byte, bool) {
	toolName = strings.TrimSpace(toolName)
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return []byte(`{}`), true
	}
	// If it's already JSON (object/array/string/number/bool/null), don't attempt.
	if strings.HasPrefix(raw, "{") || strings.HasPrefix(raw, "[") || strings.HasPrefix(raw, "\"") || raw == "null" || raw == "true" || raw == "false" {
		return nil, false
	}

	// Treat as a plain string argument.
	// Map to the most likely single-field schema.
	switch toolName {
	case "ls":
		b, _ := json.Marshal(map[string]any{"path": raw})
		return b, true
	case "read":
		b, _ := json.Marshal(map[string]any{"file_path": raw})
		return b, true
	case "bash":
		b, _ := json.Marshal(map[string]any{"command": raw})
		return b, true
	case "glob":
		b, _ := json.Marshal(map[string]any{"pattern": raw})
		return b, true
	case "grep":
		b, _ := json.Marshal(map[string]any{"pattern": raw})
		return b, true
	case "webfetch":
		b, _ := json.Marshal(map[string]any{"url": raw})
		return b, true
	case "apply_patch":
		b, _ := json.Marshal(map[string]any{"patch": raw})
		return b, true
	default:
		return nil, false
	}
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
