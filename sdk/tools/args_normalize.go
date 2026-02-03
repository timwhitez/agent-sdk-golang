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
			if normalized, display, ok := wrapStringToolArg(toolName, s); ok {
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
			if repaired, ok := repairLooseJSONObject(raw); ok {
				out.Normalized = repaired
				out.Display = decodeArgsObject(repaired, rawOriginal)
				out.Meta = markArgsRepair(meta, "loose_object", rawOriginal)
				return out
			}
		}
		if normalized, display, ok := wrapStringToolArg(toolName, raw); ok {
			out.Normalized = normalized
			out.Display = display
			out.Meta = markArgsRepair(meta, "string_wrapped", rawOriginal)
			return out
		}
		out.Normalized = json.RawMessage(`{}`)
		out.Display = map[string]any{"__raw": rawOriginal}
		out.Meta = markArgsDecodeError(meta, err, rawOriginal)
		out.Err = err
		return out
	}
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

func wrapStringToolArg(toolName, value string) (json.RawMessage, map[string]any, bool) {
	value = strings.TrimSpace(value)
	if value == "" {
		return nil, nil, false
	}
	key := stringArgKeyForTool(toolName)
	if key == "" {
		return nil, nil, false
	}
	m := map[string]any{key: value}
	b, _ := json.Marshal(m)
	return b, m, true
}

func stringArgKeyForTool(toolName string) string {
	switch strings.ToLower(strings.TrimSpace(toolName)) {
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
	case "webfetch", "web_request":
		return "url"
	case "apply_patch", "patch":
		return "patch"
	default:
		return ""
	}
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

func repairToolArgsBySchema(schema map[string]any, raw json.RawMessage, meta map[string]any) (json.RawMessage, map[string]any, bool) {
	repaired, ok := repairJSONKeysBySchema(schema, raw)
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
