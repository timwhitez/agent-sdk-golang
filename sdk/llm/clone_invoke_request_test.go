package llm

import "testing"

func TestCloneInvokeRequestOwnsNestedMutableState(t *testing.T) {
	enabled := true
	temperature := 0.25
	original := InvokeRequest{
		Messages: []Message{{
			Role:    RoleUser,
			Content: Content{Blocks: []ContentBlock{{Type: "image_url", ImageURL: &ImageURL{URL: "before"}}}},
		}},
		Tools:       []ToolDefinition{{Name: "read", Parameters: map[string]any{"properties": map[string]any{"path": map[string]any{"type": "string"}}}}},
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
