package llm

import (
	"encoding/json"
	"strings"
	"testing"
)

type marshalOnlySchemaValue struct {
	state map[string]any
}

func (value marshalOnlySchemaValue) MarshalJSON() ([]byte, error) {
	return json.Marshal(value.state)
}

func TestCloneInvokeRequestOwnsNestedMutableState(t *testing.T) {
	enabled := true
	temperature := 0.25
	original := InvokeRequest{
		Messages: []Message{{
			Role:    RoleUser,
			Content: Content{Blocks: []ContentBlock{{Type: "image_url", ImageURL: &ImageURL{URL: "before"}}}},
		}},
		Tools: []ToolDefinition{{Name: "read", Parameters: map[string]any{
			"properties": map[string]any{"path": map[string]any{"type": "string", "maximum": uint64(9007199254740993), "multipleOf": json.Number("0.1")}},
			"required":   []string{"path"},
			"opaque":     json.RawMessage(`{"type":"string"}`),
		}}},
		Temperature: &temperature,
		Responses: &ResponsesOptions{
			UseResponseItems:  &enabled,
			Include:           []string{"usage"},
			Text:              &ResponsesTextControls{Format: &ResponsesTextFormat{Schema: map[string]any{"type": "object"}}},
			ParallelToolCalls: &enabled,
			OutputSchema:      map[string]any{"type": "object"},
		},
	}

	cloned, err := CloneInvokeRequest(original)
	if err != nil {
		t.Fatal(err)
	}
	original.Messages[0].Content.Blocks[0].ImageURL.URL = "after"
	original.Tools[0].Parameters["properties"].(map[string]any)["path"].(map[string]any)["type"] = "number"
	*original.Temperature = 1
	*original.Responses.UseResponseItems = false
	original.Responses.Include[0] = "mutated"
	original.Responses.Text.Format.Schema["type"] = "array"
	*original.Responses.ParallelToolCalls = false
	original.Responses.OutputSchema["type"] = "array"

	if got := cloned.Messages[0].Content.Blocks[0].ImageURL.URL; got != "before" {
		t.Fatalf("message URL = %q", got)
	}
	if got := cloned.Tools[0].Parameters["properties"].(map[string]any)["path"].(map[string]any)["type"]; got != "string" {
		t.Fatalf("tool schema type = %#v", got)
	}
	pathSchema := cloned.Tools[0].Parameters["properties"].(map[string]any)["path"].(map[string]any)
	if got, ok := pathSchema["maximum"].(uint64); !ok || got != 9007199254740993 {
		t.Fatalf("tool schema maximum = %#v (%T)", pathSchema["maximum"], pathSchema["maximum"])
	}
	if got, ok := pathSchema["multipleOf"].(json.Number); !ok || got != json.Number("0.1") {
		t.Fatalf("tool schema multipleOf = %#v (%T)", pathSchema["multipleOf"], pathSchema["multipleOf"])
	}
	if _, ok := cloned.Tools[0].Parameters["required"].([]string); !ok {
		t.Fatalf("required type = %T, want []string", cloned.Tools[0].Parameters["required"])
	}
	if _, ok := cloned.Tools[0].Parameters["opaque"].(json.RawMessage); !ok {
		t.Fatalf("opaque type = %T, want json.RawMessage", cloned.Tools[0].Parameters["opaque"])
	}
	if *cloned.Temperature != 0.25 || !*cloned.Responses.UseResponseItems || cloned.Responses.Include[0] != "usage" {
		t.Fatalf("request pointers or slices were shared: %#v", cloned)
	}
	if got := cloned.Responses.Text.Format.Schema["type"]; got != "object" {
		t.Fatalf("text schema type = %#v", got)
	}
	if !*cloned.Responses.ParallelToolCalls || cloned.Responses.OutputSchema["type"] != "object" {
		t.Fatalf("responses state was shared: %#v", cloned.Responses)
	}
}

func TestCloneInvokeRequestPreservesNonNilEmptyInclude(t *testing.T) {
	cloned, err := CloneInvokeRequest(InvokeRequest{Responses: &ResponsesOptions{Include: []string{}}})
	if err != nil {
		t.Fatal(err)
	}
	if cloned.Responses.Include == nil {
		t.Fatal("non-nil empty Include became nil")
	}
}

func TestCloneInvokeRequestRejectsUnexportedMutableState(t *testing.T) {
	_, err := CloneInvokeRequest(InvokeRequest{Tools: []ToolDefinition{{
		Name:       "custom",
		Parameters: map[string]any{"custom": marshalOnlySchemaValue{state: map[string]any{"type": "string"}}},
	}}})
	if err == nil || !strings.Contains(err.Error(), "unexported mutable field") {
		t.Fatalf("error = %v, want unexported mutable field rejection", err)
	}
}
