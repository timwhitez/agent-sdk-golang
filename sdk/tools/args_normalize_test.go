package tools

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

func TestNormalizeToolArgsLooseObject(t *testing.T) {
	out := NormalizeToolArgs("read", `{"file_path": /tmp}`, nil)
	if out.Err != nil {
		t.Fatalf("unexpected error: %v", out.Err)
	}
	if out.Normalized == nil {
		t.Fatalf("expected normalized args")
	}
	if v, ok := out.Display["file_path"].(string); !ok || v != "/tmp" {
		t.Fatalf("expected file_path=/tmp, got %v", out.Display["file_path"])
	}
	if out.Meta == nil {
		t.Fatalf("expected meta for repaired args")
	}
	if repaired, ok := out.Meta["args_repaired"].(bool); !ok || !repaired {
		t.Fatalf("expected args_repaired=true, got %v", out.Meta["args_repaired"])
	}
	if kind, ok := out.Meta["args_repair_kind"].(string); !ok || kind != "loose_object" {
		t.Fatalf("expected args_repair_kind=loose_object, got %v", out.Meta["args_repair_kind"])
	}
}

func TestNormalizeToolArgsStringWrap(t *testing.T) {
	out := NormalizeToolArgs("read", "/tmp", nil)
	if out.Err != nil {
		t.Fatalf("unexpected error: %v", out.Err)
	}
	if v, ok := out.Display["file_path"].(string); !ok || v != "/tmp" {
		t.Fatalf("expected file_path=/tmp, got %v", out.Display["file_path"])
	}
	if out.Meta == nil {
		t.Fatalf("expected meta for repaired args")
	}
	if kind, ok := out.Meta["args_repair_kind"].(string); !ok || kind != "string_wrapped" {
		t.Fatalf("expected args_repair_kind=string_wrapped, got %v", out.Meta["args_repair_kind"])
	}

	out = NormalizeToolArgs("read", `" /tmp "`, nil)
	if out.Err != nil {
		t.Fatalf("unexpected error: %v", out.Err)
	}
	if v, ok := out.Display["file_path"].(string); !ok || v != "/tmp" {
		t.Fatalf("expected file_path=/tmp, got %v", out.Display["file_path"])
	}

	out = NormalizeToolArgs("tools.function.read", "/tmp", nil)
	if out.Err != nil {
		t.Fatalf("unexpected error: %v", out.Err)
	}
	if v, ok := out.Display["file_path"].(string); !ok || v != "/tmp" {
		t.Fatalf("expected prefixed tool name to map to file_path=/tmp, got %v", out.Display["file_path"])
	}

	out = NormalizeToolArgs("bash", "{", nil)
	if out.Err == nil {
		t.Fatalf("expected decode error for malformed object fragment")
	}
	if _, ok := out.Display["command"]; ok {
		t.Fatalf("did not expect malformed fragment to be wrapped as command")
	}
	if _, ok := out.Display["__raw"]; !ok {
		t.Fatalf("expected __raw display fallback for malformed fragment")
	}

	out = NormalizeToolArgs("bash", `"{"`, nil)
	if out.Err != nil {
		t.Fatalf("expected JSON string to still wrap successfully, got %v", out.Err)
	}
	if v, ok := out.Display["command"].(string); !ok || v != "{" {
		t.Fatalf("expected explicit JSON string to map to command {, got %#v", out.Display["command"])
	}
}

func TestNormalizeToolArgsStringWrapUsesSingleStringSchema(t *testing.T) {
	type args struct {
		Query string `json:"query"`
	}

	out := NormalizeToolArgs("custom_search", "golang", SchemaFor[args]())
	if out.Err != nil {
		t.Fatalf("unexpected error: %v", out.Err)
	}
	if v, ok := out.Display["query"].(string); !ok || v != "golang" {
		t.Fatalf("expected query=golang, got %#v", out.Display["query"])
	}
	if kind, ok := out.Meta["args_repair_kind"].(string); !ok || kind != "string_wrapped" {
		t.Fatalf("expected args_repair_kind=string_wrapped, got %v", out.Meta["args_repair_kind"])
	}
}

func TestNormalizeToolArgsStringWrapSkipsMultiFieldSchema(t *testing.T) {
	type args struct {
		Query string `json:"query"`
		Limit int    `json:"limit"`
	}

	out := NormalizeToolArgs("custom_search", "golang", SchemaFor[args]())
	if out.Err == nil {
		t.Fatalf("expected error for plain string with multi-field schema")
	}
	if _, ok := out.Display["__raw"]; !ok {
		t.Fatalf("expected __raw display fallback")
	}
}

func TestNormalizeToolArgsWriteTextFallbackPathAndContent(t *testing.T) {
	raw := "/tmp/issues.md\n# Goode 项目代码检查报告\n- item"
	out := NormalizeToolArgs("write", raw, nil)
	if out.Err != nil {
		t.Fatalf("unexpected error: %v", out.Err)
	}
	if got, _ := out.Display["file_path"].(string); got != "/tmp/issues.md" {
		t.Fatalf("expected file_path=/tmp/issues.md, got %#v", out.Display["file_path"])
	}
	if got, _ := out.Display["content"].(string); !strings.Contains(got, "Goode 项目代码检查报告") {
		t.Fatalf("expected content preserved, got %#v", out.Display["content"])
	}
	if kind, _ := out.Meta["args_repair_kind"].(string); !strings.Contains(kind, "write_text") {
		t.Fatalf("expected args_repair_kind to include write_text, got %q", kind)
	}
	if rawMeta, _ := out.Meta["args_raw"].(string); rawMeta != raw {
		t.Fatalf("expected args_raw to preserve original raw text")
	}
}

func TestNormalizeToolArgsWriteTextFallbackSupportsPathLabel(t *testing.T) {
	raw := "file_path: /tmp/issues.md\ncontent:\nhello"
	out := NormalizeToolArgs("tools.function.write", raw, nil)
	if out.Err != nil {
		t.Fatalf("unexpected error: %v", out.Err)
	}
	if got, _ := out.Display["file_path"].(string); got != "/tmp/issues.md" {
		t.Fatalf("expected file_path=/tmp/issues.md, got %#v", out.Display["file_path"])
	}
	if got, _ := out.Display["content"].(string); got != "hello" {
		t.Fatalf("expected content=hello, got %#v", out.Display["content"])
	}
}

func TestNormalizeToolArgsWriteTextFallbackRejectsPathOnlyInput(t *testing.T) {
	out := NormalizeToolArgs("write", "/tmp/issues.md", nil)
	if out.Err == nil {
		t.Fatalf("expected error for path-only write input")
	}
	if _, ok := out.Display["__raw"]; !ok {
		t.Fatalf("expected __raw display fallback")
	}
}

func TestNormalizeToolArgsNonObject(t *testing.T) {
	out := NormalizeToolArgs("read", `[1, 2]`, nil)
	if out.Err == nil {
		t.Fatalf("expected error for non-object JSON")
	}
	if out.Normalized == nil || string(out.Normalized) != "{}" {
		t.Fatalf("expected normalized {} for non-object, got %s", string(out.Normalized))
	}
	if _, ok := out.Display["__raw"]; !ok {
		t.Fatalf("expected __raw display fallback")
	}
	if repaired, ok := out.Meta["args_repaired"].(bool); !ok || !repaired {
		t.Fatalf("expected args_repaired=true, got %v", out.Meta["args_repaired"])
	}
	if kind, ok := out.Meta["args_repair_kind"].(string); !ok || kind != "non_object" {
		t.Fatalf("expected args_repair_kind=non_object, got %v", out.Meta["args_repair_kind"])
	}
}

func TestNormalizeToolArgsDecodeError(t *testing.T) {
	out := NormalizeToolArgs("unknown", "{oops", nil)
	if out.Err == nil {
		t.Fatalf("expected error for invalid JSON")
	}
	if out.Normalized == nil || string(out.Normalized) != "{}" {
		t.Fatalf("expected normalized {} for decode error, got %s", string(out.Normalized))
	}
	if _, ok := out.Display["__raw"]; !ok {
		t.Fatalf("expected __raw display fallback")
	}
	if repaired, ok := out.Meta["args_repaired"].(bool); !ok || !repaired {
		t.Fatalf("expected args_repaired=true, got %v", out.Meta["args_repaired"])
	}
	if kind, ok := out.Meta["args_repair_kind"].(string); !ok || kind != "decode_error" {
		t.Fatalf("expected args_repair_kind=decode_error, got %v", out.Meta["args_repair_kind"])
	}
	if _, ok := out.Meta["args_decode_error"]; !ok {
		t.Fatalf("expected args_decode_error")
	}
}

func TestNormalizeToolArgsJSONFence(t *testing.T) {
	out := NormalizeToolArgs("read", "```json\n{\"file_path\":\"/tmp\"}\n```", nil)
	if out.Err != nil {
		t.Fatalf("unexpected error: %v", out.Err)
	}
	if v, ok := out.Display["file_path"].(string); !ok || v != "/tmp" {
		t.Fatalf("expected file_path=/tmp, got %v", out.Display["file_path"])
	}
	if repaired, ok := out.Meta["args_repaired"].(bool); !ok || !repaired {
		t.Fatalf("expected args_repaired=true, got %v", out.Meta["args_repaired"])
	}
	if kind, ok := out.Meta["args_repair_kind"].(string); !ok || !strings.Contains(kind, "json_fence") {
		t.Fatalf("expected args_repair_kind to include json_fence, got %v", out.Meta["args_repair_kind"])
	}
}

func TestNormalizeToolArgsJSONFenceTrailingText(t *testing.T) {
	raw := "```json\n{\"file_path\":\"/tmp\"}\n```\nextra"
	out := NormalizeToolArgs("read", raw, nil)
	if out.Err != nil {
		t.Fatalf("unexpected error: %v", out.Err)
	}
	if v, ok := out.Display["file_path"].(string); !ok || v != "/tmp" {
		t.Fatalf("expected file_path=/tmp, got %v", out.Display["file_path"])
	}
	if repaired, ok := out.Meta["args_repaired"].(bool); !ok || !repaired {
		t.Fatalf("expected args_repaired=true, got %v", out.Meta["args_repaired"])
	}
	if kind, ok := out.Meta["args_repair_kind"].(string); !ok || !strings.Contains(kind, "json_fence") || !strings.Contains(kind, "trailing_text") {
		t.Fatalf("expected args_repair_kind to include json_fence,trailing_text, got %v", out.Meta["args_repair_kind"])
	}
	if rawMeta, ok := out.Meta["args_raw"].(string); !ok || rawMeta != raw {
		t.Fatalf("expected args_raw=%q, got %v", raw, out.Meta["args_raw"])
	}
}

func TestNormalizeToolArgsFirstObjectExtraction(t *testing.T) {
	out := NormalizeToolArgs("read", "args: {\"file_path\":\"/tmp\"} trailing", nil)
	if out.Err != nil {
		t.Fatalf("unexpected error: %v", out.Err)
	}
	if v, ok := out.Display["file_path"].(string); !ok || v != "/tmp" {
		t.Fatalf("expected file_path=/tmp, got %v", out.Display["file_path"])
	}
	if repaired, ok := out.Meta["args_repaired"].(bool); !ok || !repaired {
		t.Fatalf("expected args_repaired=true, got %v", out.Meta["args_repaired"])
	}
	if kind, ok := out.Meta["args_repair_kind"].(string); !ok || !strings.Contains(kind, "first_object") {
		t.Fatalf("expected args_repair_kind to include first_object, got %v", out.Meta["args_repair_kind"])
	}
	if kind, ok := out.Meta["args_repair_kind"].(string); !ok || !strings.Contains(kind, "trailing_text") {
		t.Fatalf("expected args_repair_kind to include trailing_text, got %v", out.Meta["args_repair_kind"])
	}
}

func TestExtractFirstJSONObjectRejectsInvalidBalancedJSON(t *testing.T) {
	if _, _, _, ok := extractFirstJSONObject(`prefix {"file_path": /tmp} suffix`); ok {
		t.Fatalf("expected invalid JSON object to be ignored")
	}
}

func TestExtractFirstJSONObjectFindsLaterValidObject(t *testing.T) {
	obj, leading, trailing, ok := extractFirstJSONObject(`noise {"file_path": /tmp} and {"file_path":"/tmp"} done`)
	if !ok {
		t.Fatalf("expected valid JSON object extraction")
	}
	if obj != `{"file_path":"/tmp"}` {
		t.Fatalf("unexpected object %q", obj)
	}
	if !leading {
		t.Fatalf("expected leading text marker")
	}
	if !trailing {
		t.Fatalf("expected trailing text marker")
	}
}

func TestToolExecuteArgsRepairMetadata(t *testing.T) {
	t.Run("loose_object", func(t *testing.T) {
		tool := Tool{
			Name: "read",
			Handler: func(_ context.Context, _ json.RawMessage, _ *Container) (llm.Content, error) {
				return llm.TextContent("ok"), nil
			},
		}
		ctx := WithToolResultMetadata(context.Background())
		if _, err := tool.Execute(ctx, `{"file_path": /tmp}`, NewContainer()); err != nil {
			t.Fatalf("execute: %v", err)
		}
		meta := ToolResultMetadataSnapshot(ctx)
		if meta == nil {
			t.Fatalf("expected metadata")
		}
		if repaired, ok := meta["args_repaired"].(bool); !ok || !repaired {
			t.Fatalf("expected args_repaired=true, got %v", meta["args_repaired"])
		}
		if kind, ok := meta["args_repair_kind"].(string); !ok || !strings.Contains(kind, "loose_object") {
			t.Fatalf("expected args_repair_kind to include loose_object, got %v", meta["args_repair_kind"])
		}
	})

	t.Run("schema_key", func(t *testing.T) {
		type writeArgs struct {
			FilePath string `json:"file_path"`
			Content  string `json:"content"`
		}
		writeTool := Func[writeArgs]("write", "test", func(_ context.Context, a writeArgs, _ *Container) (any, error) {
			return a.FilePath + "|" + a.Content, nil
		})
		ctx := WithToolResultMetadata(context.Background())
		if _, err := writeTool.Execute(ctx, `{"path":"notes.txt","contents":"hello","extra":"drop-me"}`, NewContainer()); err != nil {
			t.Fatalf("execute: %v", err)
		}
		meta := ToolResultMetadataSnapshot(ctx)
		if meta == nil {
			t.Fatalf("expected metadata")
		}
		if repaired, ok := meta["args_repaired"].(bool); !ok || !repaired {
			t.Fatalf("expected args_repaired=true, got %v", meta["args_repaired"])
		}
		if kind, ok := meta["args_repair_kind"].(string); !ok || !strings.Contains(kind, "schema_key") {
			t.Fatalf("expected args_repair_kind to include schema_key, got %v", meta["args_repair_kind"])
		}
	})
}

func TestRepairJSONKeysBySchemaPrefersConservativeMatch(t *testing.T) {
	type args struct {
		UserID     string `json:"user_id"`
		UserUserID string `json:"user_user_id"`
	}
	schema := SchemaFor[args]()
	raw := json.RawMessage(`{"user user_id":"abc"}`)
	repaired, ok := repairJSONKeysBySchema(schema, raw)
	if !ok {
		t.Fatalf("expected repair to succeed")
	}
	var m map[string]any
	if err := json.Unmarshal(repaired, &m); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if v, ok := m["user_user_id"].(string); !ok || v != "abc" {
		t.Fatalf("expected user_user_id=abc, got %v", m["user_user_id"])
	}
	if _, ok := m["user_id"]; ok {
		t.Fatalf("expected user_id to remain unset")
	}
}

func TestRepairJSONKeysBySchemaRepairsNestedObjects(t *testing.T) {
	type nested struct {
		FilePath string `json:"file_path"`
	}
	type args struct {
		Payload nested `json:"payload"`
	}
	schema := SchemaFor[args]()
	raw := json.RawMessage(`{"payload":{"file_Path":"notes.txt"}}`)
	repaired, ok := repairJSONKeysBySchema(schema, raw)
	if !ok {
		t.Fatalf("expected nested repair to succeed")
	}
	var m map[string]any
	if err := json.Unmarshal(repaired, &m); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	payload, ok := m["payload"].(map[string]any)
	if !ok {
		t.Fatalf("expected payload object, got %#v", m["payload"])
	}
	if v, ok := payload["file_path"].(string); !ok || v != "notes.txt" {
		t.Fatalf("expected payload.file_path=notes.txt, got %#v", payload["file_path"])
	}
	if _, ok := payload["file_Path"]; ok {
		t.Fatalf("expected malformed nested key to be repaired")
	}
}

func TestRepairJSONKeysBySchemaStripsUnknownKeysRecursivelyWhenEnabled(t *testing.T) {
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"payload": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"file_path": map[string]any{"type": "string"},
				},
			},
		},
	}
	raw := json.RawMessage(`{"payload":{"filePath":"notes.txt","extra":"drop"},"extra_top":"drop"}`)

	repaired, ok := repairJSONKeysBySchemaWithOptions(schema, raw, schemaRepairOptions{StripUnknown: true, ToolName: "read"})
	if !ok {
		t.Fatalf("expected recursive schema repair to run")
	}

	var m map[string]any
	if err := json.Unmarshal(repaired, &m); err != nil {
		t.Fatalf("unmarshal repaired: %v", err)
	}
	if _, exists := m["extra_top"]; exists {
		t.Fatalf("expected top-level unknown key to be dropped, got %#v", m)
	}
	payload, ok := m["payload"].(map[string]any)
	if !ok {
		t.Fatalf("expected payload object, got %#v", m["payload"])
	}
	if got, _ := payload["file_path"].(string); got != "notes.txt" {
		t.Fatalf("expected payload.file_path=notes.txt, got %#v", payload["file_path"])
	}
	if _, exists := payload["extra"]; exists {
		t.Fatalf("expected nested unknown key to be dropped, got %#v", payload)
	}
}

func TestRepairJSONKeysBySchemaKeepsUnknownKeysWhenStripUnknownDisabled(t *testing.T) {
	type args struct {
		FilePath string `json:"file_path"`
	}

	schema := SchemaFor[args]()
	raw := json.RawMessage(`{"filePath":"notes.txt","extra":"keep"}`)

	repaired, ok := repairJSONKeysBySchemaWithOptions(schema, raw, schemaRepairOptions{StripUnknown: false, ToolName: "read"})
	if !ok {
		t.Fatalf("expected key repair to run")
	}

	var m map[string]any
	if err := json.Unmarshal(repaired, &m); err != nil {
		t.Fatalf("unmarshal repaired: %v", err)
	}
	if got, _ := m["file_path"].(string); got != "notes.txt" {
		t.Fatalf("expected repaired file_path, got %#v", m["file_path"])
	}
	if got, _ := m["extra"].(string); got != "keep" {
		t.Fatalf("expected unknown key to be preserved when StripUnknown=false, got %#v", m)
	}
}

func TestRepairJSONKeysBySchemaOffsetLineAliasIsToolSpecific(t *testing.T) {
	type args struct {
		Offset int `json:"offset"`
	}

	schema := SchemaFor[args]()
	raw := json.RawMessage(`{"line":12}`)

	readRepaired, ok := repairJSONKeysBySchemaWithOptions(schema, raw, schemaRepairOptions{StripUnknown: true, ToolName: "read"})
	if !ok {
		t.Fatalf("expected read tool alias repair")
	}
	var readMap map[string]any
	if err := json.Unmarshal(readRepaired, &readMap); err != nil {
		t.Fatalf("unmarshal read repaired: %v", err)
	}
	if got, ok := readMap["offset"].(float64); !ok || got != 12 {
		t.Fatalf("expected offset=12 for read alias repair, got %#v", readMap["offset"])
	}

	readPrefixedRepaired, ok := repairJSONKeysBySchemaWithOptions(schema, raw, schemaRepairOptions{StripUnknown: true, ToolName: "tools.function.read"})
	if !ok {
		t.Fatalf("expected prefixed read tool alias repair")
	}
	var readPrefixedMap map[string]any
	if err := json.Unmarshal(readPrefixedRepaired, &readPrefixedMap); err != nil {
		t.Fatalf("unmarshal prefixed read repaired: %v", err)
	}
	if got, ok := readPrefixedMap["offset"].(float64); !ok || got != 12 {
		t.Fatalf("expected offset=12 for prefixed read alias repair, got %#v", readPrefixedMap["offset"])
	}

	otherRepaired, ok := repairJSONKeysBySchemaWithOptions(schema, raw, schemaRepairOptions{StripUnknown: true, ToolName: "custom_tool"})
	if !ok {
		t.Fatalf("expected custom tool payload to be normalized")
	}
	var otherMap map[string]any
	if err := json.Unmarshal(otherRepaired, &otherMap); err != nil {
		t.Fatalf("unmarshal custom repaired: %v", err)
	}
	if len(otherMap) != 0 {
		t.Fatalf("expected ambiguous line alias dropped for custom tool, got %#v", otherMap)
	}
}

func TestNormalizeToolArgsLooseObjectRejectsSchemaTypeMismatch(t *testing.T) {
	type args struct {
		Count int `json:"count"`
	}
	schema := SchemaFor[args]()
	out := NormalizeToolArgs("counter", `{"count": abc}`, schema)
	if out.Err == nil {
		t.Fatalf("expected error when loose-object repair violates schema")
	}
	if kind, _ := out.Meta["args_repair_kind"].(string); strings.Contains(kind, "loose_object") {
		t.Fatalf("expected loose_object repair to be rejected, got kind=%q", kind)
	}
}

func TestNormalizeToolArgsLooseObjectPreservesColonStringValue(t *testing.T) {
	type args struct {
		Path string `json:"path"`
		Note string `json:"note"`
	}
	schema := SchemaFor[args]()
	raw := "{\"path\": /tmp, \"note\":\"Time: 3pm {}\"}"
	out := NormalizeToolArgs("read", raw, schema)
	if out.Err != nil {
		t.Fatalf("unexpected error: %v", out.Err)
	}
	if got, _ := out.Display["path"].(string); got != "/tmp" {
		t.Fatalf("expected path=/tmp, got %q", got)
	}
	if got, _ := out.Display["note"].(string); got != "Time: 3pm {}" {
		t.Fatalf("expected note to stay intact, got %q", got)
	}
	if kind, _ := out.Meta["args_repair_kind"].(string); !strings.Contains(kind, "loose_object") {
		t.Fatalf("expected loose_object repair kind, got %q", kind)
	}
}

func TestNormalizeToolArgsLooseObjectRepairsUnquotedColonValue(t *testing.T) {
	type args struct {
		Path string `json:"path"`
		Note string `json:"note"`
	}
	schema := SchemaFor[args]()
	raw := "{\"path\": /tmp, \"note\": Time: 3pm}"
	out := NormalizeToolArgs("read", raw, schema)
	if out.Err != nil {
		t.Fatalf("unexpected error: %v", out.Err)
	}
	if got, _ := out.Display["note"].(string); got != "Time: 3pm" {
		t.Fatalf("expected repaired note value, got %q", got)
	}
}

func TestToolExecuteEmptyContentAddsWarning(t *testing.T) {
	tool := Tool{
		Name: "noop",
		Handler: func(_ context.Context, _ json.RawMessage, _ *Container) (llm.Content, error) {
			return llm.Content{}, nil
		},
	}
	ctx := WithToolResultMetadata(context.Background())
	out, err := tool.Execute(ctx, `{}`, NewContainer())
	if err != nil {
		t.Fatalf("execute: %v", err)
	}
	if got := out.PlainText(); got != "Warning: tool returned no output." {
		t.Fatalf("expected warning fallback content, got %q", got)
	}
	meta := ToolResultMetadataSnapshot(ctx)
	if meta == nil {
		t.Fatalf("expected metadata snapshot")
	}
	if got, _ := meta["tool_warning"].(string); got != "handler returned empty content" {
		t.Fatalf("expected tool_warning metadata, got %q", got)
	}
}
