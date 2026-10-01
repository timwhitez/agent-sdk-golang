package openai_test

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/agent"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm/openai"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools/sandbox"
)

type strictRoundTrip func(*http.Request) (*http.Response, error)

func (f strictRoundTrip) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

// Embedding the non-streaming interface exercises Agent's Invoke path.
type strictInvokeOnly struct{ llm.ChatModel }

func strictFixtureTool[T any](name string) tools.Tool {
	return tools.Func[T](name, "strict fixture", func(context.Context, T, *tools.Container) (any, error) { return "ok", nil })
}

func strictFixtureClient(provider string, httpClient *http.Client) llm.StreamingChatModel {
	if provider == "chat" {
		return &openai.ChatClient{ModelName: "fixture", BaseURL: "https://fixture.invalid", HTTPClient: httpClient}
	}
	return &openai.ResponsesClient{ModelName: "fixture", BaseURL: "https://fixture.invalid", HTTPClient: httpClient}
}

func strictFixtureResponse(provider string, r *http.Request) *http.Response {
	body := `{"choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`
	if provider == "responses" {
		body = `{"id":"resp_fixture","status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"ok"}]}]}`
	}
	return &http.Response{StatusCode: 200, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(body)), Request: r}
}

func strictWireDefinitions(t *testing.T, r *http.Request, provider string) map[string]map[string]any {
	t.Helper()
	var body struct {
		Tools []map[string]any `json:"tools"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		t.Fatal(err)
	}
	defs := map[string]map[string]any{}
	for _, def := range body.Tools {
		if provider == "chat" {
			def = def["function"].(map[string]any)
		}
		defs[def["name"].(string)] = def
	}
	return defs
}

func TestStrictPolicyWireMatrix(t *testing.T) {
	type scalar struct {
		Value    string `json:"value"`
		Optional *int   `json:"optional,omitempty"`
	}
	type stringMap struct {
		Headers map[string]string `json:"headers,omitempty"`
	}
	type anyMap struct {
		Values map[string]any `json:"values"`
	}
	type nested struct {
		Child stringMap `json:"child"`
	}
	type arrayMap struct {
		Rows []map[string]string `json:"rows"`
	}
	type anyValue struct {
		Value any `json:"value"`
	}
	type openPointer struct {
		Headers *map[string]string `json:"headers,omitempty"`
	}
	fixtures := []tools.Tool{
		strictFixtureTool[scalar]("scalar"), strictFixtureTool[stringMap]("string_map"),
		strictFixtureTool[anyMap]("any_map"), strictFixtureTool[nested]("nested"),
		strictFixtureTool[arrayMap]("array_map"), strictFixtureTool[anyValue]("any"),
		strictFixtureTool[openPointer]("open_pointer"), strictFixtureTool[scalar]("explicit_false").WithStrict(false),
		strictFixtureTool[stringMap]("explicit_false_open").WithStrict(false),
	}
	for _, provider := range []string{"chat", "responses"} {
		t.Run(provider, func(t *testing.T) {
			req := llm.InvokeRequest{Messages: []llm.Message{{Role: llm.RoleUser, Content: llm.TextContent("hi")}}}
			for _, tool := range fixtures {
				req.Tools = append(req.Tools, tool.Definition())
			}
			before, err := json.Marshal(req)
			if err != nil {
				t.Fatal(err)
			}
			calls := 0
			hc := &http.Client{Transport: strictRoundTrip(func(r *http.Request) (*http.Response, error) {
				calls++
				defs := strictWireDefinitions(t, r, provider)
				for _, tool := range fixtures {
					def := defs[tool.Name]
					wantStrict := tool.Name == "scalar"
					if strict, present := def["strict"]; !present || strict != wantStrict {
						t.Fatalf("%s strict = %#v, present=%v; want %v", tool.Name, strict, present, wantStrict)
					}
					if !wantStrict {
						want, _ := json.Marshal(tool.Schema)
						got, _ := json.Marshal(def["parameters"])
						if string(want) != string(got) {
							t.Fatalf("%s schema changed: %s != %s", tool.Name, got, want)
						}
					}
				}
				params := defs["scalar"]["parameters"].(map[string]any)
				if !reflect.DeepEqual(params["required"], []any{"optional", "value"}) {
					t.Fatalf("required = %#v", params["required"])
				}
				optional := params["properties"].(map[string]any)["optional"].(map[string]any)
				if !reflect.DeepEqual(optional["type"], []any{"integer", "null"}) {
					t.Fatalf("optional type = %#v", optional["type"])
				}
				return strictFixtureResponse(provider, r), nil
			})}
			_, err = strictFixtureClient(provider, hc).Invoke(context.Background(), req)
			if err != nil {
				t.Fatal(err)
			}
			after, _ := json.Marshal(req)
			if calls != 1 || string(before) != string(after) {
				t.Fatalf("calls=%d, original request mutated=%v", calls, string(before) != string(after))
			}
		})
	}
}

func TestExplicitIncompatibleStrictStopsBeforeHTTP(t *testing.T) {
	type args struct {
		Headers map[string]string `json:"headers"`
	}
	tool := strictFixtureTool[args]("headers").WithStrict(true)
	definitions := []llm.ToolDefinition{tool.Definition(), {Name: "manual_headers", Parameters: tool.Schema, Strict: true}}
	for _, provider := range []string{"chat", "responses"} {
		t.Run(provider, func(t *testing.T) {
			var calls atomic.Int32
			hc := &http.Client{Transport: strictRoundTrip(func(*http.Request) (*http.Response, error) {
				calls.Add(1)
				return nil, fmt.Errorf("network must not run")
			})}
			client := strictFixtureClient(provider, hc)
			for _, definition := range definitions {
				req := llm.InvokeRequest{Tools: []llm.ToolDefinition{definition}}
				check := func(err error) {
					t.Helper()
					if err == nil || !strings.Contains(err.Error(), definition.Name) || !strings.Contains(err.Error(), "$.properties.headers.additionalProperties") {
						t.Fatalf("strict error = %v", err)
					}
				}
				_, err := client.Invoke(context.Background(), req)
				check(err)
				events, err := client.InvokeStream(context.Background(), req)
				if err == nil {
					for event := range events {
						if failure, ok := event.(llm.StreamErrorEvent); ok {
							err = failure.Err
						}
					}
				}
				check(err)
			}
			if calls.Load() != 0 {
				t.Fatalf("HTTP calls = %d", calls.Load())
			}
		})
	}
}

func TestSandboxToolsAgentWireStrictPolicy(t *testing.T) {
	for _, provider := range []string{"chat", "responses"} {
		t.Run(provider, func(t *testing.T) {
			calls := 0
			hc := &http.Client{Transport: strictRoundTrip(func(r *http.Request) (*http.Response, error) {
				calls++
				defs := strictWireDefinitions(t, r, provider)
				if defs["webfetch"]["strict"] != false || defs["read"]["strict"] != true || defs["bash"]["strict"] != true {
					t.Fatalf("sandbox policies: %#v", defs)
				}
				params := defs["webfetch"]["parameters"].(map[string]any)
				headers := params["properties"].(map[string]any)["headers"].(map[string]any)
				if !reflect.DeepEqual(headers["additionalProperties"], map[string]any{"type": "string"}) {
					t.Fatalf("headers schema = %#v", headers)
				}
				return strictFixtureResponse(provider, r), nil
			})}
			var warnings []string
			ag, err := agent.New(agent.Config{LLM: strictInvokeOnly{strictFixtureClient(provider, hc)}, Tools: sandbox.Tools(), Warningf: func(format string, args ...any) { warnings = append(warnings, fmt.Sprintf(format, args...)) }})
			if err != nil {
				t.Fatal(err)
			}
			result, err := ag.Query(context.Background(), "hello")
			if err != nil || result != "ok" || calls != 1 {
				t.Fatalf("query: %q, %v; calls=%d", result, err, calls)
			}
			if len(warnings) != 1 || !strings.Contains(warnings[0], "webfetch") || !strings.Contains(warnings[0], "$.properties.headers.additionalProperties") {
				t.Fatalf("compatibility warning = %#v", warnings)
			}
		})
	}
}

func TestCyclicToolSchemaPreservesAgentCloneError(t *testing.T) {
	schema := map[string]any{"type": "object", "additionalProperties": false}
	schema["properties"] = map[string]any{"cycle": schema}
	client := strictFixtureClient("chat", &http.Client{})
	_, err := agent.New(agent.Config{LLM: client, Tools: []tools.Tool{{Name: "cyclic", Schema: schema}}})
	if err == nil || !strings.Contains(err.Error(), "clone tool") {
		t.Fatalf("cyclic schema registration = %v", err)
	}
}

func TestAlreadyStrictSchemasRetainWireCompatibility(t *testing.T) {
	schemas := map[string]string{
		"anyOf":           `{"type":"object","properties":{"value":{"anyOf":[{"type":"string"},{"type":"number"}]}},"required":["value"],"additionalProperties":false}`,
		"definitions":     `{"type":"object","properties":{"value":{"$ref":"#/$defs/child"}},"required":["value"],"additionalProperties":false,"$defs":{"child":{"type":"object","properties":{"x":{"type":"string"}},"required":["x"],"additionalProperties":false}}}`,
		"nullable_object": `{"type":"object","properties":{"value":{"type":["object","null"],"properties":{"x":{"type":"string"}},"required":["x"],"additionalProperties":false}},"required":["value"],"additionalProperties":false}`,
	}
	for _, provider := range []string{"chat", "responses"} {
		for name, raw := range schemas {
			t.Run(provider+"/"+name, func(t *testing.T) {
				var schema map[string]any
				if err := json.Unmarshal([]byte(raw), &schema); err != nil {
					t.Fatal(err)
				}
				tool := tools.Tool{Name: name, Schema: schema}
				if !tool.Definition().Strict || tool.Definition().StrictWarning != "" {
					t.Fatalf("already strict schema auto policy = %#v", tool.Definition())
				}
				calls := 0
				hc := &http.Client{Transport: strictRoundTrip(func(r *http.Request) (*http.Response, error) {
					calls++
					def := strictWireDefinitions(t, r, provider)[name]
					if def["strict"] != true || !reflect.DeepEqual(schema, def["parameters"]) {
						t.Fatalf("already strict schema changed: %#v", def)
					}
					return strictFixtureResponse(provider, r), nil
				})}
				_, err := strictFixtureClient(provider, hc).Invoke(context.Background(), llm.InvokeRequest{Tools: []llm.ToolDefinition{{Name: name, Parameters: schema, Strict: true}}})
				if err != nil || calls != 1 {
					t.Fatalf("already strict request: %v; calls=%d", err, calls)
				}
			})
		}
	}
}

func TestOpenSchemasInsideReferencesAndCompositionsStopBeforeHTTP(t *testing.T) {
	schemas := map[string]string{
		"anyOf":           `{"type":"object","properties":{"value":{"anyOf":[{"type":"string"},{"type":"object","additionalProperties":{"type":"string"}}]}},"required":["value"],"additionalProperties":false}`,
		"definitions":     `{"type":"object","properties":{"value":{"$ref":"#/$defs/child"}},"required":["value"],"additionalProperties":false,"$defs":{"child":{"type":"object","additionalProperties":{"type":"string"}}}}`,
		"nullable_object": `{"type":"object","properties":{"value":{"type":["object","null"],"additionalProperties":{"type":"string"}}},"required":["value"],"additionalProperties":false}`,
	}
	for _, provider := range []string{"chat", "responses"} {
		for name, raw := range schemas {
			t.Run(provider+"/"+name, func(t *testing.T) {
				var schema map[string]any
				if err := json.Unmarshal([]byte(raw), &schema); err != nil {
					t.Fatal(err)
				}
				def := (tools.Tool{Name: name, Schema: schema}).Definition()
				if def.Strict || !strings.Contains(def.StrictWarning, "additionalProperties") {
					t.Fatalf("open schema policy = %#v", def)
				}
				calls := 0
				hc := &http.Client{Transport: strictRoundTrip(func(*http.Request) (*http.Response, error) { calls++; return nil, fmt.Errorf("network must not run") })}
				def.Strict = true
				_, err := strictFixtureClient(provider, hc).Invoke(context.Background(), llm.InvokeRequest{Tools: []llm.ToolDefinition{def}})
				if err == nil || !strings.Contains(err.Error(), name) || !strings.Contains(err.Error(), "additionalProperties") || calls != 0 {
					t.Fatalf("open composed schema: %v; HTTP calls=%d", err, calls)
				}
			})
		}
	}
}
