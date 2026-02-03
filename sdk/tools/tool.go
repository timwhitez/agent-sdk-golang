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
	norm := NormalizeToolArgs(t.Name, argsJSON, t.Schema)
	if norm.Err != nil {
		if argsRepaired(norm.Meta) {
			UpsertToolResultMetadata(ctx, norm.Meta)
		}
		return llm.TextContent(fmt.Sprintf("Error parsing arguments: %v", norm.Err)), norm.Err
	}
	if norm.Normalized == nil {
		if norm.Err != nil {
			return llm.TextContent(fmt.Sprintf("Error parsing arguments: %v", norm.Err)), norm.Err
		}
		return llm.TextContent("Error parsing arguments: invalid tool args"), fmt.Errorf("invalid tool args")
	}

	call := func(raw json.RawMessage, meta map[string]any) (llm.Content, error, map[string]any) {
		content, err := t.Handler(ctx, raw, deps)
		if err == nil {
			return content, nil, meta
		}
		// Second-chance: some models/proxies emit slightly-wrong keys (e.g. "content content")
		// that fail strict decoding. Try to normalize keys to the schema and retry.
		if looksLikeUnknownFieldErr(err) {
			meta = ensureArgsRaw(meta, argsJSON)
			if repaired, meta2, ok := repairToolArgsBySchema(t.Schema, raw, meta); ok {
				if content2, err2 := t.Handler(ctx, repaired, deps); err2 == nil {
					return content2, nil, meta2
				}
			}
		}
		return content, err, meta
	}

	content, err, meta := call(norm.Normalized, norm.Meta)
	if argsRepaired(meta) {
		UpsertToolResultMetadata(ctx, meta)
	}
	return content, err
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
	aliasMap := map[string]string{} // normalized alias -> canonical key
	for k := range props {
		for _, alias := range aliasKeysForExpected(k) {
			if alias == "" {
				continue
			}
			aliasMap[normalizeKeyNoDelims(alias)] = k
		}
	}
	if len(props) == 1 {
		for k := range props {
			for _, alias := range singleFieldAliases() {
				if alias == "" {
					continue
				}
				aliasMap[normalizeKeyNoDelims(alias)] = k
			}
			break
		}
	}

	changed := false
	for k, v := range m {
		if _, ok := expected[k]; ok {
			continue
		}
		cand := normalizeCandidateKey(k)
		if cand != "" {
			norm := normalizeKeyNoDelims(cand)
			if canon, ok := expectedNoDelims[norm]; ok {
				if _, exists := m[canon]; !exists {
					m[canon] = v
					changed = true
				}
				delete(m, k)
				continue
			}
			if canon, ok := aliasMap[norm]; ok {
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
				continue
			}
		}
		if canon, ok := aliasMap[normalizeKeyNoDelims(k)]; ok {
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
		// Drop tokens that are substrings of longer tokens (e.g. "file filepath" -> "filepath").
		filtered := make([]string, 0, len(parts))
		for i, p := range parts {
			keep := true
			for j, q := range parts {
				if i == j {
					continue
				}
				if strings.Contains(q, p) && len(q) >= len(p) {
					keep = false
					break
				}
			}
			if keep {
				filtered = append(filtered, p)
			}
		}
		if len(filtered) == 1 {
			return filtered[0]
		}
		if len(filtered) > 1 {
			return strings.Join(filtered, "_")
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

func aliasKeysForExpected(key string) []string {
	switch normalizeKeyNoDelims(key) {
	case "filepath":
		return []string{"path", "file", "filename", "file_path"}
	case "path":
		return []string{"filepath", "file_path", "dir", "directory", "folder"}
	case "command":
		return []string{"cmd", "shell", "bash", "sh"}
	case "content":
		return []string{"contents", "data", "text", "body"}
	case "pattern":
		return []string{"query", "regex", "search", "match"}
	case "url":
		return []string{"uri", "link"}
	case "oldstring":
		return []string{"old", "from", "before"}
	case "newstring":
		return []string{"new", "to", "after", "replacement"}
	case "patch":
		return []string{"diff"}
	case "offset":
		return []string{"start", "line", "start_line"}
	case "limit":
		return []string{"lines", "max_lines", "count"}
	default:
		return nil
	}
}

func singleFieldAliases() []string {
	return []string{"input", "args", "argument", "value", "text", "data"}
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
		Name:          name,
		Description:   description,
		EphemeralKeep: 0,
		Schema:        schema,
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
