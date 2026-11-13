package openai_test

import (
	"context"
	"encoding/json"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm/openai"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
	"io"
	"net/http"
	"reflect"
	"strings"
	"testing"
)

type schemaTransport func(*http.Request) (*http.Response, error)

func (f schemaTransport) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }
func TestStrictSchemaNormalizesNestedOptionalObjects(t *testing.T) {
	for name, raw := range map[string]string{
		"nullable_object": `{"type":"object","additionalProperties":false,"required":["value"],"properties":{"value":{"type":["object","null"],"additionalProperties":false,"properties":{"x":{"type":"string"}}}}}`,
		"anyOf":           `{"type":"object","additionalProperties":false,"required":["value"],"properties":{"value":{"anyOf":[{"type":"string"},{"type":"object","additionalProperties":false,"properties":{"x":{"type":"string"}}}]}}}`,
		"defs":            `{"type":"object","additionalProperties":false,"required":["value"],"properties":{"value":{"$ref":"#/$defs/child"}},"$defs":{"child":{"type":"object","additionalProperties":false,"properties":{"x":{"type":"string"}}}}}`,
	} {
		for _, provider := range []string{"chat", "responses"} {
			t.Run(name+"/"+provider, func(t *testing.T) {
				var schema map[string]any
				if err := json.Unmarshal([]byte(raw), &schema); err != nil {
					t.Fatal(err)
				}
				before, _ := json.Marshal(schema)
				def := (tools.Tool{Name: "fixture", Schema: schema}).Definition()
				hc := &http.Client{Transport: schemaTransport(func(r *http.Request) (*http.Response, error) {
					var body map[string]any
					if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
						t.Fatal(err)
					}
					wire := body["tools"].([]any)[0].(map[string]any)
					if provider == "chat" {
						wire = wire["function"].(map[string]any)
					}
					params := wire["parameters"].(map[string]any)
					var child map[string]any
					if name == "defs" {
						child = params["$defs"].(map[string]any)["child"].(map[string]any)
					} else {
						child = params["properties"].(map[string]any)["value"].(map[string]any)
						if name == "anyOf" {
							child = child["anyOf"].([]any)[1].(map[string]any)
						}
					}
					if wire["strict"] != true {
						t.Error("expected strict schema")
					}
					if !reflect.DeepEqual(child["required"], []any{"x"}) {
						t.Errorf("schema accepted as strict but nested x remains optional on wire: child=%v", child)
					}
					if !reflect.DeepEqual(child["properties"].(map[string]any)["x"].(map[string]any)["type"], []any{"string", "null"}) {
						t.Error("optional x must allow null")
					}
					result := `{"id":"ok","choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`
					if provider == "responses" {
						result = `{"id":"ok","status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"ok"}]}]}`
					}
					return &http.Response{StatusCode: 200, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(result))}, nil
				})}
				var model llm.ChatModel
				if provider == "chat" {
					model = &openai.ChatClient{BaseURL: "https://fixture.test/v1", ModelName: "fixture", HTTPClient: hc, MaxRetries: 1}
				} else {
					model = &openai.ResponsesClient{BaseURL: "https://fixture.test/v1", ModelName: "fixture", HTTPClient: hc, MaxRetries: 1}
				}
				if _, err := model.Invoke(context.Background(), llm.InvokeRequest{Messages: []llm.Message{llm.NewUserMessage("hi")}, Tools: []llm.ToolDefinition{def}}); err != nil {
					t.Fatal(err)
				}
				after, _ := json.Marshal(schema)
				if string(before) != string(after) {
					t.Fatal("caller schema mutated")
				}
			})
		}
	}
}
