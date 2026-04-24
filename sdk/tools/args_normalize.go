package tools

import (
	"encoding/json"
	"errors"
	"io"
	"strings"
)

// ToolArgsNormalization captures normalized tool arguments and metadata.
type ToolArgsNormalization struct {
	Normalized json.RawMessage
	Display    map[string]any
	Meta       map[string]any
	Err        error
}

// NormalizeToolArgs normalizes tool arguments for execution and UI display.
// It performs best-effort repairs for common malformed inputs and returns
// metadata describing any repair applied.
func NormalizeToolArgs(toolName, raw string, schema map[string]any) ToolArgsNormalization {
	rawOriginal := raw
	raw = strings.TrimSpace(raw)
	meta := baseArgsMeta()
	raw, meta = normalizeToolArgsRaw(raw, rawOriginal, meta)
	out := ToolArgsNormalization{Display: map[string]any{}, Meta: meta}
	if raw == "" {
		out.Normalized = json.RawMessage(`{}`)
		out.Display = map[string]any{}
		return out
	}

	if v, err := decodeJSONValueStrict(raw); err == nil {
		switch vv := v.(type) {
		case map[string]any:
			if vv == nil {
				vv = map[string]any{}
			}
			out.Normalized = json.RawMessage([]byte(raw))
			out.Display = vv
			return out
		case string:
			s := strings.TrimSpace(vv)
			if normalized, display, ok := wrapStringToolArg(toolName, s, schema); ok {
				out.Normalized = normalized
				out.Display = display
				out.Meta = markArgsRepair(meta, "string_wrapped", rawOriginal)
				return out
			}
			out.Normalized = json.RawMessage(`{}`)
			out.Display = map[string]any{"__raw": rawOriginal}
			out.Meta = markArgsRepair(meta, "non_object", rawOriginal)
			out.Err = errExpectedJSONObject
			return out
		default:
			out.Normalized = json.RawMessage(`{}`)
			out.Display = map[string]any{"__raw": rawOriginal}
			out.Meta = markArgsRepair(meta, "non_object", rawOriginal)
			out.Err = errExpectedJSONObject
			return out
		}
	} else {
		if strings.HasPrefix(raw, "{") {
			if repaired, ok := repairLooseJSONObject(raw, schema); ok {
				out.Normalized = repaired
				out.Display = decodeArgsObject(repaired, rawOriginal)
				out.Meta = markArgsRepair(meta, "loose_object", rawOriginal)
				return out
			}
		}
		if canWrapInvalidRawAsString(raw) {
			if normalized, display, ok := wrapStringToolArg(toolName, raw, schema); ok {
				out.Normalized = normalized
				out.Display = display
				out.Meta = markArgsRepair(meta, "string_wrapped", rawOriginal)
				return out
			}
		}
		if normalized, display, ok := wrapWriteToolArgs(toolName, raw); ok {
			out.Normalized = normalized
			out.Display = display
			out.Meta = markArgsRepair(meta, "write_text", rawOriginal)
			return out
		}
		out.Normalized = json.RawMessage(`{}`)
		out.Display = map[string]any{"__raw": rawOriginal}
		out.Meta = markArgsDecodeError(meta, err, rawOriginal)
		out.Err = err
		return out
	}
}

func canWrapInvalidRawAsString(raw string) bool {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return false
	}
	if strings.HasPrefix(raw, "{") || strings.HasPrefix(raw, "[") {
		return false
	}
	return true
}

var errExpectedJSONObject = errors.New("expected JSON object")

func decodeJSONValueStrict(raw string) (any, error) {
	dec := json.NewDecoder(strings.NewReader(raw))
	dec.DisallowUnknownFields()
	var v any
	if err := dec.Decode(&v); err != nil {
		return nil, err
	}
	if err := ensureDecoderEOF(dec); err != nil {
		return nil, err
	}
	return v, nil
}

func ensureDecoderEOF(dec *json.Decoder) error {
	var extra any
	if err := dec.Decode(&extra); err != io.EOF {
		if err == nil {
			return errors.New("extra JSON values")
		}
		return err
	}
	return nil
}

func normalizeToolArgsRaw(raw, rawOriginal string, meta map[string]any) (string, map[string]any) {
	if raw == "" {
		return raw, meta
	}
	if stripped, ok, trailing := stripJSONCodeFence(raw); ok {
		raw = strings.TrimSpace(stripped)
		meta = appendArgsRepairKind(meta, "json_fence")
		meta = ensureArgsRaw(meta, rawOriginal)
		if trailing {
			meta = appendArgsRepairKind(meta, "trailing_text")
		}
	}
	if obj, leading, trailing, ok := extractFirstJSONObject(raw); ok && (leading || trailing) {
		raw = strings.TrimSpace(obj)
		meta = appendArgsRepairKind(meta, "first_object")
		meta = ensureArgsRaw(meta, rawOriginal)
		if trailing {
			meta = appendArgsRepairKind(meta, "trailing_text")
		}
	}
	return raw, meta
}

func stripJSONCodeFence(raw string) (string, bool, bool) {
	trimmed := strings.TrimSpace(raw)
	if !strings.HasPrefix(trimmed, "```") {
		return raw, false, false
	}
	lineEnd := strings.IndexByte(trimmed, '\n')
	if lineEnd == -1 {
		return raw, false, false
	}
	lang := strings.TrimSpace(trimmed[3:lineEnd])
	if lang != "" && !strings.HasPrefix(strings.ToLower(lang), "json") {
		return raw, false, false
	}
	rest := trimmed[lineEnd+1:]
	closeIdx := strings.LastIndex(rest, "```")
	if closeIdx == -1 {
		return raw, false, false
	}
	inner := rest[:closeIdx]
	trailing := strings.TrimSpace(rest[closeIdx+3:]) != ""
	return inner, true, trailing
}

func extractFirstJSONObject(raw string) (string, bool, bool, bool) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return "", false, false, false
	}
	inStr := false
	esc := false
	start := -1
	depth := 0
	for i := 0; i < len(raw); i++ {
		c := raw[i]
		if inStr {
			if esc {
				esc = false
				continue
			}
			if c == '\\' {
				esc = true
				continue
			}
			if c == '"' {
				inStr = false
			}
			continue
		}
		switch c {
		case '"':
			inStr = true
		case '{':
			if start == -1 {
				start = i
				depth = 1
			} else {
				depth++
			}
		case '}':
			if start == -1 {
				continue
			}
			depth--
			if depth == 0 {
				obj := raw[start : i+1]
				if !json.Valid([]byte(obj)) {
					start = -1
					continue
				}
				leading := strings.TrimSpace(raw[:start]) != ""
				trailing := strings.TrimSpace(raw[i+1:]) != ""
				return obj, leading, trailing, true
			}
		}
	}
	return "", false, false, false
}

func decodeArgsObject(raw json.RawMessage, rawFallback string) map[string]any {
	if len(raw) == 0 {
		return map[string]any{}
	}
	var m map[string]any
	if err := json.Unmarshal(raw, &m); err == nil {
		if m == nil {
			return map[string]any{}
		}
		return m
	}
	if strings.TrimSpace(rawFallback) == "" {
		return map[string]any{}
	}
	return map[string]any{"__raw": rawFallback}
}

func wrapStringToolArg(toolName, value string, schema map[string]any) (json.RawMessage, map[string]any, bool) {
	value = strings.TrimSpace(value)
	if value == "" {
		return nil, nil, false
	}
	key := singleStringArgKeyFromSchema(schema)
	if key == "" {
		key = stringArgKeyForTool(toolName)
	}
	if key == "" {
		return nil, nil, false
	}
	m := map[string]any{key: value}
	b, _ := json.Marshal(m)
	return b, m, true
}

func singleStringArgKeyFromSchema(schema map[string]any) string {
	if len(schema) == 0 {
		return ""
	}
	if t, ok := schema["type"].(string); ok && strings.TrimSpace(t) != "" && t != "object" {
		return ""
	}
	props, ok := schema["properties"].(map[string]any)
	if !ok || len(props) != 1 {
		return ""
	}
	for key, prop := range props {
		propSchema, ok := prop.(map[string]any)
		if !ok || !schemaTypeIncludesString(propSchema) {
			return ""
		}
		return key
	}
	return ""
}

func schemaTypeIncludesString(schema map[string]any) bool {
	if schema == nil {
		return false
	}
	typeVal, ok := schema["type"]
	if !ok {
		return false
	}
	switch v := typeVal.(type) {
	case string:
		return v == "string"
	case []any:
		for _, item := range v {
			ts, ok := item.(string)
			if ok && ts == "string" {
				return true
			}
		}
	}
	return false
}

func stringArgKeyForTool(toolName string) string {
	switch NormalizeToolName(toolName) {
	case "ls", "list":
		return "path"
	case "read":
		return "file_path"
	case "bash":
		return "command"
	case "glob":
		return "pattern"
	case "grep":
		return "pattern"
	case "webfetch", "webrequest":
		return "url"
	case "applypatch", "patch":
		return "patch"
	default:
		return ""
	}
}

func wrapWriteToolArgs(toolName, raw string) (json.RawMessage, map[string]any, bool) {
	if NormalizeToolName(toolName) != "write" {
		return nil, nil, false
	}
	raw = strings.ReplaceAll(raw, "\r\n", "\n")
	raw = strings.ReplaceAll(raw, "\r", "\n")
	lines := strings.Split(raw, "\n")
	pathIdx := -1
	path := ""
	for i, line := range lines {
		if p, ok := parseWritePathLine(line); ok {
			pathIdx = i
			path = p
			break
		}
	}
	if pathIdx < 0 || strings.TrimSpace(path) == "" {
		return nil, nil, false
	}
	if len(lines) <= pathIdx+1 {
		return nil, nil, false
	}
	contentLines := lines[pathIdx+1:]
	if len(contentLines) > 0 {
		first := strings.TrimSpace(contentLines[0])
		lower := strings.ToLower(first)
		for _, prefix := range []string{"content:", "content="} {
			if strings.HasPrefix(lower, prefix) {
				head := strings.TrimSpace(first[len(prefix):])
				contentLines = contentLines[1:]
				if head != "" {
					contentLines = append([]string{head}, contentLines...)
				}
				break
			}
		}
	}
	content := strings.Join(contentLines, "\n")
	if strings.TrimSpace(content) == "" {
		return nil, nil, false
	}
	m := map[string]any{
		"file_path": path,
		"content":   content,
	}
	b, _ := json.Marshal(m)
	return b, m, true
}

func parseWritePathLine(line string) (string, bool) {
	trimmed := strings.TrimSpace(line)
	if trimmed == "" {
		return "", false
	}
	lower := strings.ToLower(trimmed)
	for _, prefix := range []string{"file_path:", "filepath:", "path:", "file_path=", "filepath=", "path="} {
		if strings.HasPrefix(lower, prefix) {
			val := strings.TrimSpace(trimmed[len(prefix):])
			val = stripWrappedQuotes(val)
			if looksLikeFilePath(val) {
				return val, true
			}
			return "", false
		}
	}
	val := stripWrappedQuotes(trimmed)
	if looksLikeFilePath(val) {
		return val, true
	}
	return "", false
}

func stripWrappedQuotes(s string) string {
	s = strings.TrimSpace(s)
	for {
		if len(s) < 2 {
			return s
		}
		if (s[0] == 34 && s[len(s)-1] == 34) || (s[0] == 39 && s[len(s)-1] == 39) || (s[0] == 96 && s[len(s)-1] == 96) {
			s = strings.TrimSpace(s[1 : len(s)-1])
			continue
		}
		return s
	}
}

func looksLikeFilePath(s string) bool {
	s = strings.TrimSpace(s)
	if s == "" {
		return false
	}
	if strings.Contains(s, "\n") || strings.Contains(s, "\r") {
		return false
	}
	if strings.Contains(s, "://") {
		return false
	}
	if strings.HasPrefix(s, "/") || strings.HasPrefix(s, "./") || strings.HasPrefix(s, "../") || strings.HasPrefix(s, "~/") {
		return true
	}
	if len(s) >= 3 {
		c := s[0]
		if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) && s[1] == ':' && (s[2] == 92 || s[2] == '/') {
			return true
		}
	}
	if strings.ContainsAny(s, `/\\`) {
		return true
	}
	if strings.Contains(s, ".") && !strings.Contains(s, " ") {
		return true
	}
	return false
}

func baseArgsMeta() map[string]any {
	return map[string]any{
		"args_repaired":    false,
		"args_repair_kind": "",
	}
}

func markArgsRepair(meta map[string]any, kind, raw string) map[string]any {
	meta = appendArgsRepairKind(meta, kind)
	meta = ensureArgsRaw(meta, raw)
	return meta
}

func markArgsDecodeError(meta map[string]any, err error, raw string) map[string]any {
	meta = markArgsRepair(meta, "decode_error", raw)
	if err != nil {
		meta["args_decode_error"] = err.Error()
	}
	return meta
}

func appendArgsRepairKind(meta map[string]any, kind string) map[string]any {
	if meta == nil {
		meta = map[string]any{}
	}
	meta["args_repaired"] = true
	if existing, ok := meta["args_repair_kind"].(string); ok && strings.TrimSpace(existing) != "" {
		if !strings.Contains(existing, kind) {
			meta["args_repair_kind"] = existing + "," + kind
		}
	} else {
		meta["args_repair_kind"] = kind
	}
	return meta
}

func ensureArgsRaw(meta map[string]any, raw string) map[string]any {
	if meta == nil {
		meta = map[string]any{}
	}
	if _, ok := meta["args_raw"]; !ok && strings.TrimSpace(raw) != "" {
		meta["args_raw"] = raw
	}
	return meta
}

func repairToolArgsBySchema(toolName string, schema map[string]any, raw json.RawMessage, meta map[string]any) (json.RawMessage, map[string]any, bool) {
	repaired, ok := repairJSONKeysBySchemaWithOptions(schema, raw, schemaRepairOptions{StripUnknown: true, ToolName: toolName})
	if !ok {
		return nil, meta, false
	}
	meta = appendArgsRepairKind(meta, "schema_key")
	meta = ensureArgsRaw(meta, string(raw))
	return repaired, meta, true
}

func argsRepaired(meta map[string]any) bool {
	if meta == nil {
		return false
	}
	if v, ok := meta["args_repaired"].(bool); ok {
		return v
	}
	return false
}
