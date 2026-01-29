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
	call := func(raw []byte) (llm.Content, error) {
		content, err := t.Handler(ctx, json.RawMessage(raw), deps)
		if err == nil {
			return content, nil
		}
		// Second-chance: some models/proxies emit slightly-wrong keys (e.g. "content content")
		// that fail strict decoding. Try to normalize keys to the schema and retry.
		if looksLikeUnknownFieldErr(err) {
			if repaired, ok := repairJSONKeysBySchema(t.Schema, raw); ok {
				if content2, err2 := t.Handler(ctx, json.RawMessage(repaired), deps); err2 == nil {
					return content2, nil
				}
			}
		}
		return content, err
	}

	if s == "" {
		return call([]byte(`{}`))
	}
	dec := json.NewDecoder(bytes.NewReader([]byte(s)))
	dec.DisallowUnknownFields()
	var raw json.RawMessage
	if err := dec.Decode(&raw); err == nil {
		return call(raw)
	}

	// Fallback: some providers/gateways occasionally produce non-JSON (or malformed-JSON)
	// tool arguments (e.g. a bare string like "/path" or {"path":/tmp}).
	// Best-effort repair for common tools.
	if repaired, ok := repairToolArgs(t.Name, s); ok {
		return call(repaired)
	}

	// Preserve the original parsing error message for debuggability.
	dec2 := json.NewDecoder(bytes.NewReader([]byte(s)))
	dec2.DisallowUnknownFields()
	var raw2 json.RawMessage
	if err2 := dec2.Decode(&raw2); err2 != nil {
		return llm.TextContent(fmt.Sprintf("Error parsing arguments: %v", err2)), err2
	}
	return call(raw2)
}

func repairToolArgs(toolName string, raw string) ([]byte, bool) {
	toolName = strings.TrimSpace(toolName)
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return []byte(`{}`), true
	}

	// If it looks like a JSON object but is malformed (common: unquoted string values),
	// attempt a minimal repair.
	if strings.HasPrefix(raw, "{") {
		if repaired, ok := repairLooseJSONObject(raw); ok {
			return repaired, true
		}
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

func looksLikeUnknownFieldErr(err error) bool {
	if err == nil {
		return false
	}
	// json.Decoder with DisallowUnknownFields() returns errors like:
	//   json: unknown field "foo"
	return strings.Contains(err.Error(), "unknown field")
}

func repairJSONKeysBySchema(schema map[string]any, raw []byte) ([]byte, bool) {
	if len(raw) == 0 || schema == nil {
		return nil, false
	}
	propsAny, ok := schema["properties"]
	if !ok {
		return nil, false
	}
	props, ok := propsAny.(map[string]any)
	if !ok || len(props) == 0 {
		return nil, false
	}

	var m map[string]any
	if err := json.Unmarshal(raw, &m); err != nil {
		return nil, false
	}

	expected := map[string]struct{}{}
	expectedNoDelims := map[string]string{} // normalized -> canonical key
	for k := range props {
		kk := strings.TrimSpace(k)
		if kk == "" {
			continue
		}
		expected[kk] = struct{}{}
		expectedNoDelims[normalizeKeyNoDelims(kk)] = kk
	}

	changed := false
	for k, v := range m {
		if _, ok := expected[k]; ok {
			continue
		}
		cand := normalizeCandidateKey(k)
		if cand != "" {
			if canon, ok := expectedNoDelims[normalizeKeyNoDelims(cand)]; ok {
				if _, exists := m[canon]; !exists {
					m[canon] = v
					changed = true
				}
				delete(m, k)
				continue
			}
		}
		if canon, ok := expectedNoDelims[normalizeKeyNoDelims(k)]; ok {
			if canon != k {
				if _, exists := m[canon]; !exists {
					m[canon] = v
					changed = true
				}
				delete(m, k)
			}
		}
	}
	if !changed {
		return nil, false
	}
	b, err := json.Marshal(m)
	if err != nil {
		return nil, false
	}
	return b, true
}

func normalizeCandidateKey(k string) string {
	k = strings.TrimSpace(k)
	if k == "" {
		return ""
	}
	low := strings.ToLower(k)
	// Collapse duplicated whitespace-separated tokens: "content content" -> "content".
	parts := strings.Fields(low)
	if len(parts) > 1 {
		same := true
		for i := 1; i < len(parts); i++ {
			if parts[i] != parts[0] {
				same = false
				break
			}
		}
		if same {
			return parts[0]
		}
		return strings.Join(parts, "_")
	}
	// Replace common separators.
	low = strings.ReplaceAll(low, "-", "_")
	low = strings.ReplaceAll(low, " ", "_")
	return low
}

func normalizeKeyNoDelims(k string) string {
	k = strings.ToLower(strings.TrimSpace(k))
	if k == "" {
		return ""
	}
	// Keep only [a-z0-9].
	return strings.Map(func(r rune) rune {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') {
			return r
		}
		return -1
	}, k)
}

// repairLooseJSONObject tries to repair a JSON-object-like string where some string
// values are unquoted (common in tool args): {"path":/tmp} -> {"path":"/tmp"}.
// It is intentionally conservative; returns ok=false if it cannot safely repair.
func repairLooseJSONObject(raw string) ([]byte, bool) {
	raw = strings.TrimSpace(raw)
	if !strings.HasPrefix(raw, "{") {
		return nil, false
	}
	// Fast path: if it becomes valid after trimming, no repair.
	{
		dec := json.NewDecoder(bytes.NewReader([]byte(raw)))
		dec.DisallowUnknownFields()
		var tmp json.RawMessage
		if err := dec.Decode(&tmp); err == nil {
			return tmp, true
		}
	}

	// Minimal state machine: whenever we see a ':' and the next non-space byte starts an
	// unquoted token, wrap it in JSON string quotes until ',' or '}'.
	out := make([]byte, 0, len(raw)+16)
	inStr := false
	esc := false
	for i := 0; i < len(raw); {
		c := raw[i]
		if inStr {
			out = append(out, c)
			if esc {
				esc = false
				i++
				continue
			}
			if c == '\\' {
				esc = true
				i++
				continue
			}
			if c == '"' {
				inStr = false
			}
			i++
			continue
		}
		if c == '"' {
			inStr = true
			out = append(out, c)
			i++
			continue
		}
		if c != ':' {
			out = append(out, c)
			i++
			continue
		}

		// ':' encountered
		out = append(out, c)
		i++
		// Copy whitespace
		for i < len(raw) {
			s := raw[i]
			if s == ' ' || s == '\n' || s == '\r' || s == '\t' {
				out = append(out, s)
				i++
				continue
			}
			break
		}
		if i >= len(raw) {
			break
		}
		n := raw[i]
		// Already quoted or starts a structured / literal value.
		if n == '"' || n == '{' || n == '[' || n == '-' || (n >= '0' && n <= '9') {
			continue
		}
		// true/false/null
		if strings.HasPrefix(raw[i:], "true") || strings.HasPrefix(raw[i:], "false") || strings.HasPrefix(raw[i:], "null") {
			continue
		}
		// Wrap token as string.
		out = append(out, '"')
		start := len(out)
		for i < len(raw) {
			cc := raw[i]
			if cc == ',' || cc == '}' {
				break
			}
			out = append(out, cc)
			i++
		}
		// Trim trailing whitespace inside the quotes.
		for len(out) > start {
			last := out[len(out)-1]
			if last == ' ' || last == '\n' || last == '\r' || last == '\t' {
				out = out[:len(out)-1]
				continue
			}
			break
		}
		out = append(out, '"')
		// Do not consume ',' / '}' here; outer loop will handle it.
	}

	// Validate repaired JSON.
	dec := json.NewDecoder(bytes.NewReader(out))
	dec.DisallowUnknownFields()
	var fixed json.RawMessage
	if err := dec.Decode(&fixed); err != nil {
		return nil, false
	}
	return fixed, true
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
