package tools

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"math"
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

	// Hidden excludes the tool from model-visible tool definitions.
	Hidden bool

	Handler func(ctx context.Context, args json.RawMessage, deps *Container) (llm.Content, error)
}

const (
	toolDiagnosticInvalidArgsAction = "Provide valid JSON arguments that match the tool schema and retry."
	toolDiagnosticDefaultAction     = "Review the diagnostic details and retry."
)

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
		return llm.TextContent(formatToolErrorDiagnostic("Invalid tool arguments", norm.Err, toolDiagnosticInvalidArgsAction)), norm.Err
	}
	if norm.Normalized == nil {
		parseErr := norm.Err
		if parseErr == nil {
			parseErr = fmt.Errorf("invalid tool args")
		}
		return llm.TextContent(formatToolErrorDiagnostic("Invalid tool arguments", parseErr, toolDiagnosticInvalidArgsAction)), parseErr
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
			if repaired, meta2, ok := repairToolArgsBySchema(t.Name, t.Schema, raw, meta); ok {
				if content2, err2 := t.Handler(ctx, repaired, deps); err2 == nil {
					return content2, nil, meta2
				}
			}
		}
		return content, err, meta
	}

	content, err, meta := call(norm.Normalized, norm.Meta)
	if err != nil && content.IsEmpty() {
		content = llm.TextContent(formatToolErrorDiagnostic("Tool execution failed", err, toolDiagnosticDefaultAction))
	}
	if err == nil && content.IsEmpty() {
		content = llm.TextContent("Warning: tool returned no output.")
		UpsertToolResultMetadata(ctx, map[string]any{"tool_warning": "handler returned empty content"})
	}
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

func formatToolErrorDiagnostic(summary string, err error, action string) string {
	if err != nil {
		if detail := strings.TrimSpace(err.Error()); isSeverityActionDiagnostic(detail) {
			return detail
		} else if detail != "" {
			detail = strings.Join(strings.Fields(detail), " ")
			summary = strings.TrimSpace(summary)
			if summary == "" {
				summary = detail
			} else {
				summary = fmt.Sprintf("%s (%s)", summary, detail)
			}
		}
	}
	summary = strings.TrimSpace(summary)
	if summary == "" {
		summary = "Tool execution failed"
	}
	action = strings.TrimSpace(action)
	if action == "" {
		action = toolDiagnosticDefaultAction
	}
	return fmt.Sprintf("[ERROR] %s - %s", summary, action)
}

func isSeverityActionDiagnostic(text string) bool {
	text = strings.TrimSpace(text)
	if !strings.HasPrefix(text, "[") {
		return false
	}
	end := strings.Index(text, "]")
	if end <= 1 {
		return false
	}
	severity := strings.ToUpper(strings.TrimSpace(text[1:end]))
	switch severity {
	case "INFO", "WARN", "ERROR":
	default:
		return false
	}
	body := strings.TrimSpace(text[end+1:])
	if strings.Contains(body, " - ") {
		return true
	}
	return strings.Contains(body, "stage=") && strings.Contains(body, "action=")
}

type schemaRepairOptions struct {
	StripUnknown bool
	ToolName     string
}

func repairJSONKeysBySchema(schema map[string]any, raw []byte) ([]byte, bool) {
	return repairJSONKeysBySchemaWithOptions(schema, raw, schemaRepairOptions{})
}

func repairJSONKeysBySchemaWithOptions(schema map[string]any, raw []byte, opts schemaRepairOptions) ([]byte, bool) {
	if len(raw) == 0 || schema == nil {
		return nil, false
	}
	var v any
	if err := json.Unmarshal(raw, &v); err != nil {
		return nil, false
	}
	m, ok := v.(map[string]any)
	if !ok {
		return nil, false
	}

	repaired, changed := repairObjectBySchema(m, schema, opts)
	if !changed {
		return nil, false
	}
	b, err := json.Marshal(repaired)
	if err != nil {
		return nil, false
	}
	return b, true
}

type objectKeyMatcher struct {
	expected        map[string]struct{}
	expectedNoDelim map[string]string
	aliasByNoDelim  map[string]string
}

func newObjectKeyMatcher(props map[string]any, toolName string) objectKeyMatcher {
	matcher := objectKeyMatcher{
		expected:        map[string]struct{}{},
		expectedNoDelim: map[string]string{},
		aliasByNoDelim:  map[string]string{},
	}
	for k := range props {
		kk := strings.TrimSpace(k)
		if kk == "" {
			continue
		}
		matcher.expected[kk] = struct{}{}
		matcher.expectedNoDelim[normalizeKeyNoDelims(kk)] = kk
	}
	for k := range props {
		for _, alias := range aliasKeysForExpected(toolName, k) {
			if alias == "" {
				continue
			}
			matcher.aliasByNoDelim[normalizeKeyNoDelims(alias)] = k
		}
	}
	if len(props) == 1 {
		for k := range props {
			for _, alias := range singleFieldAliases() {
				if alias == "" {
					continue
				}
				matcher.aliasByNoDelim[normalizeKeyNoDelims(alias)] = k
			}
			break
		}
	}
	return matcher
}

func (m objectKeyMatcher) canonicalKey(k string) (string, bool) {
	if _, ok := m.expected[k]; ok {
		return k, true
	}
	norm := normalizeKeyNoDelims(k)
	if canon, ok := m.expectedNoDelim[norm]; ok {
		return canon, true
	}
	if canon, ok := m.aliasByNoDelim[norm]; ok {
		return canon, true
	}

	cand := normalizeCandidateKey(k)
	if cand == "" {
		return "", false
	}
	candNorm := normalizeKeyNoDelims(cand)
	if canon, ok := m.expectedNoDelim[candNorm]; ok {
		return canon, true
	}
	if canon, ok := m.aliasByNoDelim[candNorm]; ok {
		return canon, true
	}
	return "", false
}

func repairBySchemaValue(v any, schema map[string]any, opts schemaRepairOptions) (any, bool) {
	if schema == nil {
		return v, false
	}
	switch vv := v.(type) {
	case map[string]any:
		return repairObjectBySchema(vv, schema, opts)
	case []any:
		return repairArrayBySchema(vv, schema, opts)
	default:
		return v, false
	}
}

func repairArrayBySchema(in []any, schema map[string]any, opts schemaRepairOptions) ([]any, bool) {
	if len(in) == 0 {
		return in, false
	}
	itemSchema, ok := schema["items"].(map[string]any)
	if !ok || itemSchema == nil {
		return in, false
	}
	changed := false
	for i, item := range in {
		repaired, itemChanged := repairBySchemaValue(item, itemSchema, opts)
		if !itemChanged {
			continue
		}
		in[i] = repaired
		changed = true
	}
	return in, changed
}

func repairObjectBySchema(in map[string]any, schema map[string]any, opts schemaRepairOptions) (map[string]any, bool) {
	props, _ := schema["properties"].(map[string]any)
	matcher := newObjectKeyMatcher(props, opts.ToolName)
	_, additionalSchema := schemaAllowsAdditionalProperties(schema)
	stripUnknown := opts.StripUnknown
	// Keep map-like payloads (no fixed properties + typed additionalProperties).
	if len(props) == 0 && additionalSchema != nil {
		stripUnknown = false
	}

	out := make(map[string]any, len(in))
	for k, v := range in {
		out[k] = v
	}

	changed := false
	for k, v := range in {
		canon, ok := matcher.canonicalKey(k)
		if ok {
			if canon != k {
				if _, exists := out[canon]; !exists {
					out[canon] = v
				}
				delete(out, k)
				changed = true
			}
			continue
		}
		if stripUnknown {
			delete(out, k)
			changed = true
		}
	}

	for k, v := range out {
		childSchema := map[string]any(nil)
		if propSchema, ok := props[k].(map[string]any); ok {
			childSchema = propSchema
		} else if additionalSchema != nil {
			childSchema = additionalSchema
		}
		if childSchema == nil {
			continue
		}
		repaired, childChanged := repairBySchemaValue(v, childSchema, opts)
		if !childChanged {
			continue
		}
		out[k] = repaired
		changed = true
	}
	if !changed {
		return in, false
	}
	return out, true
}

func schemaAllowsAdditionalProperties(schema map[string]any) (bool, map[string]any) {
	apAny, ok := schema["additionalProperties"]
	if !ok {
		return true, nil
	}
	switch ap := apAny.(type) {
	case bool:
		return ap, nil
	case map[string]any:
		return true, ap
	default:
		return true, nil
	}
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

func aliasKeysForExpected(toolName, key string) []string {
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
		aliases := []string{"start"}
		if supportsLineOffsetAliases(toolName) {
			aliases = append(aliases, "line", "start_line")
		}
		return aliases
	case "limit":
		return []string{"lines", "max_lines", "count"}
	default:
		return nil
	}
}

func supportsLineOffsetAliases(toolName string) bool {
	switch NormalizeToolName(toolName) {
	case "read":
		return true
	default:
		return false
	}
}

func singleFieldAliases() []string {
	return []string{"input", "args", "argument", "value", "text", "data"}
}

// repairLooseJSONObject tries to repair a JSON-object-like string where some scalar
// values are unquoted (for example {"path":/tmp}). It applies conservative heuristics
// and validates the repaired payload shape against the tool schema when available.
func repairLooseJSONObject(raw string, schema map[string]any) ([]byte, bool) {
	raw = strings.TrimSpace(raw)
	if !strings.HasPrefix(raw, "{") || !strings.HasSuffix(raw, "}") {
		return nil, false
	}

	if v, err := decodeJSONValueStrict(raw); err == nil {
		obj, ok := v.(map[string]any)
		if !ok || !looseRepairSchemaCompatible(obj, schema) {
			return nil, false
		}
		return []byte(raw), true
	}

	repaired, changed, ok := quoteLooseObjectScalars(raw)
	if !ok || !changed {
		return nil, false
	}

	var parsed any
	if err := json.Unmarshal(repaired, &parsed); err != nil {
		return nil, false
	}
	obj, ok := parsed.(map[string]any)
	if !ok || !looseRepairSchemaCompatible(obj, schema) {
		return nil, false
	}
	return repaired, true
}

func quoteLooseObjectScalars(raw string) ([]byte, bool, bool) {
	out := make([]byte, 0, len(raw)+16)
	inStr := false
	esc := false
	changed := false

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

		out = append(out, c)
		i++
		if c != ':' {
			continue
		}

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
			return nil, false, false
		}
		if !shouldRepairLooseScalar(raw[i:]) {
			continue
		}

		end, token, ok := readLooseScalarToken(raw, i)
		if !ok {
			return nil, false, false
		}
		if !isSafeLooseScalarToken(token) {
			continue
		}
		quoted, err := json.Marshal(token)
		if err != nil {
			return nil, false, false
		}
		out = append(out, quoted...)
		i = end
		changed = true
	}

	if inStr {
		return nil, false, false
	}
	return out, changed, true
}

func shouldRepairLooseScalar(raw string) bool {
	if raw == "" {
		return false
	}
	n := raw[0]
	if n == '"' || n == '{' || n == '[' || n == '-' || (n >= '0' && n <= '9') {
		return false
	}
	if n == ',' || n == '}' || n == ']' {
		return false
	}
	if hasJSONLiteralPrefix(raw) {
		return false
	}
	return true
}

func hasJSONLiteralPrefix(raw string) bool {
	for _, lit := range []string{"true", "false", "null"} {
		if !strings.HasPrefix(raw, lit) {
			continue
		}
		if len(raw) == len(lit) {
			return true
		}
		if isJSONValueDelimiter(raw[len(lit)]) {
			return true
		}
	}
	return false
}

func isJSONValueDelimiter(b byte) bool {
	switch b {
	case ' ', '\n', '\r', '\t', ',', '}', ']':
		return true
	default:
		return false
	}
}

func readLooseScalarToken(raw string, start int) (int, string, bool) {
	i := start
	for i < len(raw) {
		c := raw[i]
		if c == ',' || c == '}' || c == ']' {
			break
		}
		if c == '"' || c == '{' || c == '[' {
			return 0, "", false
		}
		i++
	}
	token := strings.TrimSpace(raw[start:i])
	if token == "" {
		return 0, "", false
	}
	return i, token, true
}

func isSafeLooseScalarToken(token string) bool {
	token = strings.TrimSpace(token)
	if token == "" {
		return false
	}
	if strings.ContainsAny(token, "\"'{}[]") {
		return false
	}
	if token == "-" {
		return false
	}
	return true
}

func looseRepairSchemaCompatible(value any, schema map[string]any) bool {
	if schema == nil || len(schema) == 0 {
		return true
	}
	return schemaAllowsValue(value, schema)
}

func schemaAllowsValue(value any, schema map[string]any) bool {
	if schema == nil || len(schema) == 0 {
		return true
	}
	if !schemaAllowsType(value, schema) {
		return false
	}

	switch v := value.(type) {
	case map[string]any:
		props, _ := schema["properties"].(map[string]any)
		allowUnknown, additionalSchema := schemaAllowsAdditionalProperties(schema)
		for k, child := range v {
			if propSchema, ok := props[k].(map[string]any); ok {
				if !schemaAllowsValue(child, propSchema) {
					return false
				}
				continue
			}
			if additionalSchema != nil {
				if !schemaAllowsValue(child, additionalSchema) {
					return false
				}
				continue
			}
			if !allowUnknown {
				return false
			}
		}
	case []any:
		itemSchema, _ := schema["items"].(map[string]any)
		if itemSchema == nil {
			return true
		}
		for _, child := range v {
			if !schemaAllowsValue(child, itemSchema) {
				return false
			}
		}
	}

	return true
}

func schemaAllowsType(value any, schema map[string]any) bool {
	types := schemaTypeSet(schema)
	if len(types) == 0 {
		return true
	}
	vType := jsonValueType(value)
	if vType == "" {
		return false
	}
	if _, ok := types[vType]; ok {
		return true
	}
	if vType == "integer" {
		_, ok := types["number"]
		return ok
	}
	return false
}

func schemaTypeSet(schema map[string]any) map[string]struct{} {
	out := map[string]struct{}{}
	t, ok := schema["type"]
	if !ok {
		return out
	}
	switch vv := t.(type) {
	case string:
		v := strings.ToLower(strings.TrimSpace(vv))
		if v != "" {
			out[v] = struct{}{}
		}
	case []any:
		for _, item := range vv {
			s, ok := item.(string)
			if !ok {
				continue
			}
			v := strings.ToLower(strings.TrimSpace(s))
			if v != "" {
				out[v] = struct{}{}
			}
		}
	}
	return out
}

func jsonValueType(value any) string {
	switch v := value.(type) {
	case nil:
		return "null"
	case map[string]any:
		return "object"
	case []any:
		return "array"
	case string:
		return "string"
	case bool:
		return "boolean"
	case float64:
		if v == math.Trunc(v) {
			return "integer"
		}
		return "number"
	default:
		return ""
	}
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
				return llm.TextContent(formatToolErrorDiagnostic("Invalid tool arguments", err, toolDiagnosticInvalidArgsAction)), err
			}
			res, err := fn(ctx, a, deps)
			if err != nil {
				return llm.TextContent(formatToolErrorDiagnostic("Tool execution failed", err, toolDiagnosticDefaultAction)), err
			}
			return SerializeResult(res)
		},
	}
}

func (t Tool) WithEphemeralKeep(n int) Tool {
	t.EphemeralKeep = n
	return t
}
