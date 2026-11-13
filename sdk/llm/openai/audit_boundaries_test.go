package openai

import (
	"context"
	"encoding/json"
	"errors"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"io"
	"net/http"
	"reflect"
	"strings"
	"testing"
)

type boundaryTransport func(*http.Request) (*http.Response, error)

func (f boundaryTransport) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }
func boundaryClient(provider string, hc *http.Client) llm.StreamingChatModel {
	if provider == "chat" {
		return &ChatClient{ModelName: "fixture", BaseURL: "https://user:SYNTHETIC_PASSWORD@fixture.test/private-SYNTHETIC_PATH?key=SYNTHETIC_QUERY", HTTPClient: hc, MaxRetries: 1}
	}
	return &ResponsesClient{ModelName: "fixture", BaseURL: "https://user:SYNTHETIC_PASSWORD@fixture.test/private-SYNTHETIC_PATH?key=SYNTHETIC_QUERY", HTTPClient: hc, MaxRetries: 1}
}
func boundaryResponse(body string, status int) *http.Response {
	return &http.Response{StatusCode: status, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(body))}
}
func TestStrictOptionalConstraintsAllowNull(t *testing.T) {
	for _, provider := range []string{"chat", "responses"} {
		for name, raw := range map[string]string{
			"enum": `{"type":"string","enum":["fast","safe"]}`, "const": `{"type":"string","const":"fast"}`, "composition": `{"type":"string","anyOf":[{"type":"string","enum":["fast"]}]}`,
		} {
			t.Run(provider+"/"+name, func(t *testing.T) {
				var prop map[string]any
				if err := json.Unmarshal([]byte(raw), &prop); err != nil {
					t.Fatal(err)
				}
				schema := map[string]any{"type": "object", "additionalProperties": false, "properties": map[string]any{"optional": prop, "required": prop}, "required": []string{"required"}}
				before, _ := json.Marshal(schema)
				hc := &http.Client{Transport: boundaryTransport(func(r *http.Request) (*http.Response, error) {
					var body map[string]any
					if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
						t.Fatal(err)
					}
					def := body["tools"].([]any)[0].(map[string]any)
					if provider == "chat" {
						def = def["function"].(map[string]any)
					}
					params := def["parameters"].(map[string]any)
					props := params["properties"].(map[string]any)
					optional := props["optional"].(map[string]any)
					branches, ok := optional["anyOf"].([]any)
					if !ok || len(branches) != 2 || !reflect.DeepEqual(branches[1], map[string]any{"type": "null"}) || !reflect.DeepEqual(branches[0], prop) {
						t.Errorf("optional constraint must preserve original branch and admit null: %#v", optional)
					}
					if !reflect.DeepEqual(props["required"], prop) {
						t.Error("required constraint changed")
					}
					if def["strict"] != true {
						t.Error("strict missing")
					}
					payload := `{"id":"r","status":"completed","output":[]}`
					if provider == "chat" {
						payload = `{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}]}`
					}
					return boundaryResponse(payload, 200), nil
				})}
				if _, err := boundaryClient(provider, hc).Invoke(context.Background(), llm.InvokeRequest{Tools: []llm.ToolDefinition{{Name: "fixture", Parameters: schema, Strict: true}}}); err != nil {
					t.Fatal(err)
				}
				after, _ := json.Marshal(schema)
				if string(before) != string(after) {
					t.Fatal("caller schema changed")
				}
			})
		}
	}
}
func TestChatNumericSchemaAndExtraMerge(t *testing.T) {
	schema := map[string]any{"type": "object", "properties": map[string]any{"id": map[string]any{"type": "integer", "enum": []any{json.Number("9007199254740993")}}}, "required": []string{"id"}, "additionalProperties": false}
	hc := &http.Client{Transport: boundaryTransport(func(r *http.Request) (*http.Response, error) {
		b, _ := io.ReadAll(r.Body)
		if !strings.Contains(string(b), "9007199254740993") {
			t.Errorf("numeric schema changed: %s", b)
		}
		return boundaryResponse(`{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}]}`, 200), nil
	})}
	if _, err := boundaryClient("chat", hc).Invoke(context.Background(), llm.InvokeRequest{Tools: []llm.ToolDefinition{{Name: "fixture", Parameters: schema, Strict: true}}}); err != nil {
		t.Fatal(err)
	}
	req := chatRequest{Model: "original", Extra: map[string]any{"model": "override", "exact": json.Number("9007199254740993"), "omitted": nil}}
	b, err := json.Marshal(req)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(b), `"model":"override"`) || !strings.Contains(string(b), "9007199254740993") || strings.Contains(string(b), "omitted") {
		t.Fatalf("extra merge changed: %s", b)
	}
}
func TestResponsesNumericObjectArgumentsBufferedAndStream(t *testing.T) {
	for _, stream := range []bool{false, true} {
		for _, object := range []bool{false, true} {
			t.Run(strings.Join([]string{map[bool]string{true: "stream", false: "buffered"}[stream], map[bool]string{true: "object", false: "string"}[object]}, "/"), func(t *testing.T) {
				args := `{"id":9007199254740993,"decimal":1.234567890123456789}`
				wireArgs := args
				if !object {
					b, _ := json.Marshal(args)
					wireArgs = string(b)
				}
				response := `{"id":"r","status":"completed","output":[{"type":"function_call","id":"item","call_id":"call","name":"fixture","arguments":` + wireArgs + `}],"usage":{"input_tokens":11,"output_tokens":7,"total_tokens":18}}`
				payload := response
				if stream {
					payload = "data: " + `{"type":"response.output_item.done","output_index":0,"item":{"type":"function_call","id":"item","call_id":"call","name":"fixture","arguments":` + wireArgs + `}}` + "\n\ndata: " + `{"type":"response.completed","response":` + response + `}` + "\n\n"
				}
				hc := &http.Client{Transport: boundaryTransport(func(*http.Request) (*http.Response, error) { return boundaryResponse(payload, 200), nil })}
				client := boundaryClient("responses", hc)
				got := ""
				usage := false
				if stream {
					events, err := client.InvokeStream(context.Background(), llm.InvokeRequest{})
					if err != nil {
						t.Fatal(err)
					}
					for ev := range events {
						switch e := ev.(type) {
						case llm.StreamToolCallDeltaEvent:
							got += e.ArgumentsDelta
						case llm.StreamErrorEvent:
							t.Fatal(e.AsError())
						case llm.StreamUsageEvent:
							usage = e.Usage.TotalTokens == 18
						}
					}
				} else {
					res, err := client.Invoke(context.Background(), llm.InvokeRequest{})
					if err != nil {
						t.Fatal(err)
					}
					if len(res.ToolCalls) != 1 {
						t.Fatal(res.ToolCalls)
					}
					got = res.ToolCalls[0].Function.Arguments
					usage = res.Usage != nil && res.Usage.TotalTokens == 18
				}
				if !strings.Contains(got, "9007199254740993") || !strings.Contains(got, "1.234567890123456789") {
					t.Errorf("numeric arguments changed: %s", got)
				}
				if !usage {
					t.Error("usage changed")
				}
			})
		}
	}
}
func TestEndpointDiagnosticsExcludeSecrets(t *testing.T) {
	for _, provider := range []string{"chat", "responses"} {
		for _, mode := range []string{"status", "oversize", "stream_status", "truncated", "malformed", "invalid_url"} {
			t.Run(provider+"/"+mode, func(t *testing.T) {
				status := 401
				payload := `{"error":{"message":"invalid credentials"}}`
				if mode == "oversize" {
					payload = strings.Repeat("x", int(maxProviderResponseBytes)+1)
				}
				if mode == "truncated" {
					status = 200
					payload = "data: {}\n\n"
				}
				if mode == "malformed" {
					status = 200
					payload = "data: {\n\n"
				}
				hc := &http.Client{Transport: boundaryTransport(func(r *http.Request) (*http.Response, error) {
					if !strings.Contains(r.URL.String(), "SYNTHETIC_PASSWORD") {
						t.Error("actual endpoint mutated")
					}
					return boundaryResponse(payload, status), nil
				})}
				client := boundaryClient(provider, hc)
				if mode == "invalid_url" {
					switch c := client.(type) {
					case *ChatClient:
						c.BaseURL += "%zz"
					case *ResponsesClient:
						c.BaseURL += "%zz"
					}
				}
				var got error
				if strings.HasPrefix(mode, "stream_") || mode == "truncated" || mode == "malformed" {
					events, err := client.InvokeStream(context.Background(), llm.InvokeRequest{})
					got = err
					if err == nil {
						for ev := range events {
							if e, ok := ev.(llm.StreamErrorEvent); ok {
								got = e.AsError()
							}
						}
					}
				} else {
					_, got = client.Invoke(context.Background(), llm.InvokeRequest{})
				}
				if got == nil {
					t.Fatal("expected diagnostic")
				}
				if mode == "status" || mode == "stream_status" || mode == "oversize" {
					var providerErr *llm.ProviderError
					if !errors.As(got, &providerErr) || providerErr.StatusCode != 401 {
						t.Errorf("status lost: %v", got)
					}
				}
				for _, secret := range []string{"SYNTHETIC_PASSWORD", "SYNTHETIC_PATH", "SYNTHETIC_QUERY"} {
					if strings.Contains(got.Error(), secret) {
						t.Errorf("endpoint diagnostic leaked %s: %v", secret, got)
					}
				}
			})
		}
	}
}

func TestResponsesNumericDecoderRejectsTrailingDocuments(t *testing.T) {
	for _, raw := range []string{`{"output":[]} {"output":[]}`, `{"output":[]} garbage`} {
		if _, err := parseResponses([]byte(raw)); err == nil {
			t.Fatalf("accepted trailing document: %s", raw)
		}
	}
}
