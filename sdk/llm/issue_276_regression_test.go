package llm_test

import (
	"context"
	"encoding/json"
	"fmt"
	"github.com/timwhitez/agent-sdk-golang/sdk/agent"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm/anthropic"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
	"net/http"
	"net/http/httptest"
	"slices"
	"sync/atomic"
	"testing"
)

type d5RegionKey struct{}
type d5FrameRouter struct {
	defaultModel, regionalModel, bound llm.ChatModel
}

func (m *d5FrameRouter) Provider() string { return "anthropic" }
func (m *d5FrameRouter) Model() string    { return "regional-router" }
func (m *d5FrameRouter) selectModel(ctx context.Context) llm.ChatModel {
	if m.bound != nil {
		return m.bound
	}
	if regional, _ := ctx.Value(d5RegionKey{}).(bool); regional && m.regionalModel != nil {
		return m.regionalModel
	}
	return m.defaultModel
}
func (m *d5FrameRouter) Invoke(ctx context.Context, req llm.InvokeRequest) (*llm.Completion, error) {
	return m.selectModel(ctx).Invoke(ctx, req)
}
func (m *d5FrameRouter) BindFrameModel(ctx context.Context) (llm.ChatModel, bool, error) {
	leaf, known, err := llm.BindFrameModel(ctx, m.selectModel(ctx))
	if err != nil || !known {
		return nil, false, err
	}
	return &d5FrameRouter{bound: leaf}, true, nil
}
func (m *d5FrameRouter) ThinkingCapabilities() (llm.ThinkingCapabilities, bool) {
	if m.bound != nil {
		return llm.ModelThinkingCapabilities(m.bound)
	}
	if m.regionalModel == nil {
		return llm.ModelThinkingCapabilities(m.defaultModel)
	}
	return llm.ThinkingCapabilities{}, false
}
func TestD5RequireDoneConsultsBoundModel(t *testing.T) {
	for _, scope := range []string{"always_known", "known_when_bound"} {
		t.Run(scope, func(t *testing.T) {
			var requests atomic.Int32
			modelServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var body struct {
					ToolChoice struct{ Type string } `json:"tool_choice"`
					Thinking   struct{ Type string } `json:"thinking"`
				}
				if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
					http.Error(w, err.Error(), 400)
					return
				}
				if body.ToolChoice.Type != "auto" || body.Thinking.Type == "disabled" {
					t.Errorf("wire controls unsupported on Sonnet 5.5: %+v", body)
				}
				switch requests.Add(1) {
				case 1:
					fmt.Fprint(w, `{"id":"m1","content":[{"type":"tool_use","id":"w1","name":"work","input":{}}],"stop_reason":"tool_use","usage":{"input_tokens":10,"output_tokens":2}}`)
				case 2:
					fmt.Fprint(w, `{"id":"m2","content":[{"type":"text","text":"answer"}],"stop_reason":"end_turn","usage":{"input_tokens":12,"output_tokens":2}}`)
				case 3:
					fmt.Fprint(w, `{"id":"m3","content":[{"type":"tool_use","id":"d1","name":"done","input":{"message":"finished"}}],"stop_reason":"tool_use","usage":{"input_tokens":14,"output_tokens":2}}`)
				default:
					http.Error(w, "unexpected request", 400)
				}
			}))
			defer modelServer.Close()
			leaf := &anthropic.Client{BaseURL: modelServer.URL, HTTPClient: modelServer.Client(), ModelName: "claude-sonnet-5-5", MaxRetries: 1}
			router := &d5FrameRouter{defaultModel: leaf}
			if scope == "known_when_bound" {
				fallback := *leaf
				fallback.ModelName = "claude-sonnet-5"
				router = &d5FrameRouter{defaultModel: &fallback, regionalModel: leaf}
			}
			ctx := context.WithValue(context.Background(), d5RegionKey{}, true)
			_, knownBefore := llm.ModelThinkingCapabilities(router)
			bound, known, bindErr := llm.BindFrameModel(ctx, router)
			caps, knownAfter := llm.ModelThinkingCapabilities(bound)
			if bindErr != nil || !known || !knownAfter || knownBefore != (scope == "always_known") || caps.SupportsDisableThinking || caps.SupportsForcedToolChoice {
				t.Fatalf("fixture binding failed: knownBefore=%v knownAfter=%v caps=%+v err=%v", knownBefore, knownAfter, caps, bindErr)
			}
			work := tools.Func[struct{}]("work", "work", func(context.Context, struct{}, *tools.Container) (any, error) { return "ok", nil })
			type Done struct {
				Message string `json:"message"`
			}
			done := tools.Func[Done]("done", "finish", func(_ context.Context, arg Done, _ *tools.Container) (any, error) {
				return nil, tools.TaskComplete(arg.Message)
			})
			a, err := agent.New(agent.Config{LLM: router, Tools: []tools.Tool{work, done}, ToolChoice: "auto", RequireDoneTool: true, MaxIterations: 6, InvokeRetryMaxAttempts: 1, Warningf: func(string, ...any) {}})
			if err != nil {
				t.Fatal(err)
			}
			var final, terminalError string
			var provenance []string
			warnings := 0
			for env := range a.QueryStreamEnveloped(ctx, llm.TextContent("run")) {
				if env.RequestControlRelation != "" && !slices.Contains(provenance, env.RequestControlRelation) {
					provenance = append(provenance, env.RequestControlRelation)
				}
				switch event := env.Event.(type) {
				case agent.WarnEvent:
					if event.Kind == "require_done_capability_degraded" {
						warnings++
					}
				case agent.FinalResponseEvent:
					final = event.Content
				case agent.ErrorEvent:
					terminalError = event.Message
				}
			}
			if terminalError != "" || final != "answer\n\nfinished" || warnings != 1 || len(provenance) != 0 || requests.Load() != 3 {
				t.Fatalf("HTTP requests=%d want 3, final=%q want %q, degradation warnings=%d want 1, control provenance=%v want [], terminal error=%q", requests.Load(), final, "answer\n\nfinished", warnings, provenance, terminalError)
			}
		})
	}
}
