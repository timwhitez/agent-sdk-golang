package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm/openai"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

type schemaWarningFixture struct{}

func (schemaWarningFixture) Provider() string { return "fixture" }
func (schemaWarningFixture) Model() string    { return "fixture" }
func (schemaWarningFixture) Invoke(ctx context.Context, req llm.InvokeRequest) (*llm.Completion, error) {
	sink := llm.WarningSink(ctx, nil)
	for i := 0; i < 2; i++ {
		for _, def := range req.Tools {
			if def.StrictWarning != "" {
				sink("OpenAI tool %q uses non-strict parameters to preserve its schema: %s", def.Name, def.StrictWarning)
			}
		}
		sink("unrelated warning")
	}
	return &llm.Completion{Content: llm.TextContent("pong"), StopReason: "stop"}, nil
}

type schemaWarningTransport func(*http.Request) (*http.Response, error)

func (f schemaWarningTransport) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

type schemaWarningBufferedModel struct{ llm.ChatModel }

func TestAgentSchemaWarningsThroughOpenAIClients(t *testing.T) {
	for _, responses := range []bool{false, true} {
		for _, streaming := range []bool{false, true} {
			t.Run(testBoolName(responses, "responses")+testBoolName(streaming, "stream"), func(t *testing.T) {
				requests := 0
				httpClient := &http.Client{Transport: schemaWarningTransport(func(r *http.Request) (*http.Response, error) {
					requests++
					var body map[string]any
					if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
						t.Fatal(err)
					}
					defer r.Body.Close()
					definition := body["tools"].([]any)[0].(map[string]any)
					if !responses {
						definition = definition["function"].(map[string]any)
					}
					if definition["strict"] != false || definition["parameters"].(map[string]any)["additionalProperties"] != true {
						t.Fatalf("schema changed: %v", definition)
					}
					status, content := 200, `{"choices":[{"message":{"role":"assistant","content":"pong"},"finish_reason":"stop"}]}`
					if responses {
						content = `{"id":"resp-fixture","status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"pong"}]}]}`
					}
					if body["stream"] == true {
						content = "data: {\"choices\":[{\"delta\":{\"content\":\"pong\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n"
						if responses {
							content = "data: {\"type\":\"response.output_text.delta\",\"response_id\":\"resp-fixture\",\"delta\":\"pong\"}\n\ndata: [DONE]\n\n"
						}
					}
					// The first request retries through the actual client builder.
					if requests == 1 {
						status, content = 500, `{"error":{"message":"fixture retry"}}`
					}
					return &http.Response{StatusCode: status, Header: http.Header{}, Body: io.NopCloser(strings.NewReader(content)), Request: r}, nil
				})}
				fallback := func(string, ...any) { t.Error("shared provider fallback used instead of Agent sink") }
				var model llm.ChatModel = &openai.ChatClient{HTTPClient: httpClient, BaseURL: "https://offline.invalid", ModelName: "fixture", Warningf: fallback, MaxRetries: 2, RetryBaseDelay: time.Millisecond, RetryMaxDelay: time.Millisecond}
				if responses {
					model = &openai.ResponsesClient{HTTPClient: httpClient, BaseURL: "https://offline.invalid", ModelName: "fixture", Warningf: fallback, MaxRetries: 2, RetryBaseDelay: time.Millisecond, RetryMaxDelay: time.Millisecond}
				}
				if !streaming {
					model = schemaWarningBufferedModel{model}
				}
				for owner := 0; owner < 2; owner++ {
					warnings := 0
					tool := tools.Tool{Name: "webfetch", Schema: map[string]any{"type": "object", "additionalProperties": true}}
					ag, err := New(Config{LLM: model, Tools: []tools.Tool{tool}, Warningf: func(f string, args ...any) {
						if f == schemaCompatibilityWarningFormat {
							warnings++
						}
					}})
					if err != nil {
						t.Fatal(err)
					}
					for i := 0; i < 2; i++ {
						if got, err := ag.Query(context.Background(), "ping"); err != nil || got != "pong" {
							t.Fatalf("query=%q err=%v", got, err)
						}
					}
					if warnings != 1 {
						t.Fatalf("owner %d schema warnings=%d", owner, warnings)
					}
				}
				if requests != 5 {
					t.Fatalf("requests=%d; want 4 queries and one retry", requests)
				}
			})
		}
	}
}

func TestSchemaWarningGateRetentionBounded(t *testing.T) {
	var gate schemaWarningGate
	key := schemaWarningFingerprint{}
	if gate.suppress(key) || !gate.suppress(key) {
		t.Fatal("first warning/duplicate admission")
	}
	for i := 1; i <= schemaWarningCapacity; i++ {
		entry := key
		entry.tool[0], entry.tool[1] = byte(i), byte(i>>8)
		if gate.suppress(entry) {
			t.Fatalf("fresh key %d hidden", i)
		}
	}
	if len(gate.seen) != schemaWarningCapacity || gate.recent.Len() != schemaWarningCapacity || gate.suppress(key) {
		t.Fatal("history not bounded or evicted warning hidden")
	}
}

func TestAgentSchemaWarningsRetainedPerAgent(t *testing.T) {
	model := schemaWarningFixture{}
	tool := tools.Tool{Name: "webfetch", Schema: map[string]any{"type": "object", "additionalProperties": true}, Handler: func(context.Context, json.RawMessage, *tools.Container) (llm.Content, error) {
		return llm.TextContent("unused"), nil
	}}
	for owner := 0; owner < 2; owner++ {
		var warnings []string
		ag, err := New(Config{LLM: model, Tools: []tools.Tool{tool}, Warningf: func(f string, args ...any) { warnings = append(warnings, fmt.Sprintf(f, args...)) }})
		if err != nil {
			t.Fatal(err)
		}
		for i := 0; i < 2; i++ {
			if got, err := ag.Query(context.Background(), "ping"); err != nil || got != "pong" {
				t.Fatalf("query = %q, %v", got, err)
			}
		}
		if len(warnings) != 5 {
			t.Fatalf("owner %d got %d warnings: %v; want one schema and four unrelated", owner, len(warnings), warnings)
		}
	}
}

func TestSchemaWarningGateChangesAndConcurrency(t *testing.T) {
	var gate schemaWarningGate
	var mu sync.Mutex
	var warnings []string
	sink := func(f string, args ...any) {
		mu.Lock()
		defer mu.Unlock()
		warnings = append(warnings, fmt.Sprintf(f, args...))
	}
	request := func(name, cause string, schema map[string]any) llm.InvokeRequest {
		return llm.InvokeRequest{Tools: []llm.ToolDefinition{{Name: name, StrictWarning: cause, Parameters: schema}}}
	}
	req := request("webfetch", "open object", map[string]any{"type": "object", "additionalProperties": true})
	var wg sync.WaitGroup
	for i := 0; i < 20; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			llm.WarningSink(gate.bind(context.Background(), req, sink), nil)(schemaCompatibilityWarningFormat, "webfetch", "open object")
		}()
	}
	wg.Wait()
	for _, changed := range []llm.InvokeRequest{
		request("webfetch", "open object", map[string]any{"type": "object", "additionalProperties": true, "description": "changed"}),
		request("webfetch", "changed reason", req.Tools[0].Parameters),
		request("other", "open object", req.Tools[0].Parameters),
	} {
		llm.WarningSink(gate.bind(context.Background(), changed, sink), nil)(schemaCompatibilityWarningFormat, changed.Tools[0].Name, changed.Tools[0].StrictWarning)
	}
	ctx := gate.bind(context.Background(), req, sink)
	llm.WarningSink(ctx, nil)(schemaCompatibilityWarningFormat, "unknown", "open object")
	llm.WarningSink(ctx, nil)(schemaCompatibilityWarningFormat, "webfetch", "unexpected reason")
	llm.WarningSink(ctx, nil)("provider fallback")
	if len(warnings) != 7 {
		t.Fatalf("warnings=%v", warnings)
	}
}
