package tools

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

func TestToolExecute_ErrorDiagnosticSeverityActionFormat(t *testing.T) {
	tool := Tool{
		Name: "parse_test",
		Handler: func(_ context.Context, _ json.RawMessage, _ *Container) (llm.Content, error) {
			return llm.TextContent("ok"), nil
		},
	}

	out, err := tool.Execute(context.Background(), `[]`, NewContainer())
	if err == nil {
		t.Fatal("expected argument parsing error")
	}
	assertSeverityActionDiagnostic(t, out.PlainText())
	if !strings.Contains(out.PlainText(), "Invalid tool arguments") {
		t.Fatalf("expected invalid-arguments summary, got %q", out.PlainText())
	}
	if !strings.Contains(out.PlainText(), "Provide valid JSON arguments") {
		t.Fatalf("expected actionable guidance, got %q", out.PlainText())
	}
	if strings.HasPrefix(strings.TrimSpace(out.PlainText()), "Error parsing arguments:") {
		t.Fatalf("expected structured diagnostic, got %q", out.PlainText())
	}
}

func TestFunc_ErrorDiagnosticSeverityActionFormat(t *testing.T) {
	type args struct {
		Count int `json:"count"`
	}

	t.Run("decode error", func(t *testing.T) {
		tool := Func[args]("decode_fail", "decode fail test", func(_ context.Context, _ args, _ *Container) (any, error) {
			return "ok", nil
		})
		out, err := tool.Execute(context.Background(), `{"count":"oops"}`, NewContainer())
		if err == nil {
			t.Fatal("expected decode error")
		}
		assertSeverityActionDiagnostic(t, out.PlainText())
		if strings.HasPrefix(strings.TrimSpace(out.PlainText()), "Error parsing arguments:") {
			t.Fatalf("expected structured diagnostic, got %q", out.PlainText())
		}
	})

	t.Run("handler error", func(t *testing.T) {
		tool := Func[args]("handler_fail", "handler fail test", func(_ context.Context, _ args, _ *Container) (any, error) {
			return nil, errors.New("upstream unavailable")
		})
		out, err := tool.Execute(context.Background(), `{"count":1}`, NewContainer())
		if err == nil {
			t.Fatal("expected handler error")
		}
		assertSeverityActionDiagnostic(t, out.PlainText())
		if strings.HasPrefix(strings.TrimSpace(out.PlainText()), "Error:") {
			t.Fatalf("expected structured diagnostic, got %q", out.PlainText())
		}
	})

	t.Run("preserves structured diagnostic", func(t *testing.T) {
		const diag = "[ERROR] Upstream request failed - Check network connectivity and retry."
		tool := Func[args]("handler_diag", "handler diag test", func(_ context.Context, _ args, _ *Container) (any, error) {
			return nil, errors.New(diag)
		})
		out, err := tool.Execute(context.Background(), `{"count":1}`, NewContainer())
		if err == nil {
			t.Fatal("expected handler error")
		}
		if got := out.PlainText(); got != diag {
			t.Fatalf("expected downstream diagnostic passthrough, got %q", got)
		}
	})
}

func assertSeverityActionDiagnostic(t *testing.T, text string) {
	t.Helper()
	trimmed := strings.TrimSpace(text)
	if !strings.HasPrefix(trimmed, "[ERROR] ") {
		t.Fatalf("expected [ERROR] prefix, got %q", text)
	}
	if !strings.Contains(trimmed, " - ") {
		t.Fatalf("expected summary-action delimiter, got %q", text)
	}
}

func TestUpsertToolResultMetadata_NilSafe(t *testing.T) {
	// Bug fix #2: Verify that UpsertToolResultMetadata is safe with nil meta
	// and that all meta manipulation helpers handle nil correctly.
	ctx := context.Background()

	// Test with nil context - should not panic
	UpsertToolResultMetadata(nil, nil)
	UpsertToolResultMetadata(nil, map[string]any{"key": "value"})

	// Test with context without metadata - should not panic
	UpsertToolResultMetadata(ctx, nil)
	UpsertToolResultMetadata(ctx, map[string]any{"key": "value"})

	// Test with context with metadata
	ctx = WithToolResultMetadata(ctx)
	UpsertToolResultMetadata(ctx, nil)
	UpsertToolResultMetadata(ctx, map[string]any{"key": "value"})

	snapshot := ToolResultMetadataSnapshot(ctx)
	if snapshot == nil || len(snapshot) == 0 {
		t.Fatal("expected metadata to be captured")
	}
	if snapshot["key"] != "value" {
		t.Fatalf("expected key=value, got %v", snapshot)
	}
}

func TestTakeToolResultMetadataSnapshot_NilSafe(t *testing.T) {
	// Bug fix #2: Verify that TakeToolResultMetadataSnapshot is safe with nil context
	// and contexts without metadata.

	// Test with nil context - should not panic
	meta := TakeToolResultMetadataSnapshot(nil)
	if meta != nil {
		t.Fatalf("expected nil metadata for nil context, got %v", meta)
	}

	// Test with context without metadata - should not panic
	ctx := context.Background()
	meta = TakeToolResultMetadataSnapshot(ctx)
	if meta != nil {
		t.Fatalf("expected nil metadata for context without metadata, got %v", meta)
	}

	// Test with context with metadata
	ctx = WithToolResultMetadata(ctx)
	UpsertToolResultMetadata(ctx, map[string]any{"key": "value"})
	meta = TakeToolResultMetadataSnapshot(ctx)
	if meta == nil || len(meta) == 0 {
		t.Fatal("expected metadata to be captured and cleared")
	}
	if meta["key"] != "value" {
		t.Fatalf("expected key=value, got %v", meta)
	}

	// After take, metadata should be cleared
	meta2 := ToolResultMetadataSnapshot(ctx)
	if meta2 != nil {
		t.Fatalf("expected nil metadata after take, got %v", meta2)
	}
}

func TestMetaHelpers_NilSafe(t *testing.T) {
	// Bug fix #2: Verify that all meta helpers handle nil safely
	tests := []struct {
		name string
		fn   func(map[string]any) map[string]any
	}{
		{"appendArgsRepairKind", func(m map[string]any) map[string]any {
			return appendArgsRepairKind(m, "test_kind")
		}},
		{"ensureArgsRaw", func(m map[string]any) map[string]any {
			return ensureArgsRaw(m, "test_raw")
		}},
		{"markArgsRepair", func(m map[string]any) map[string]any {
			return markArgsRepair(m, "test_kind", "test_raw")
		}},
		{"markArgsDecodeError", func(m map[string]any) map[string]any {
			return markArgsDecodeError(m, errors.New("test error"), "test_raw")
		}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Test with nil - should not panic and should return non-nil
			result := tt.fn(nil)
			if result == nil {
				t.Fatalf("%s: expected non-nil result when input is nil", tt.name)
			}
		})
	}
}
