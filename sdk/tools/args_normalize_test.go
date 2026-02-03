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
		if _, err := writeTool.Execute(ctx, `{"path":"notes.txt","contents":"hello"}`, NewContainer()); err != nil {
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
