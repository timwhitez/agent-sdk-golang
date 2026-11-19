package anthropic_test

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/agent"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm/anthropic"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

func TestAgentRequireDoneAnthropicCapabilities(t *testing.T) {
	t.Parallel()

	sseToolCall := func(id, msgID, name, inputJSON string) string {
		return strings.Join([]string{
			fmt.Sprintf(`data: {"type":"message_start","message":{"id":%q,"usage":{"input_tokens":10}}}`, msgID),
			"",
			fmt.Sprintf(`data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":%q,"name":%q,"input":{}}}`, id, name),
			"",
			fmt.Sprintf(`data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":%q}}`, inputJSON),
			"",
			`data: {"type":"content_block_stop","index":0}`,
			"",
			`data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":5}}`,
			"",
			`data: {"type":"message_stop"}`,
			"",
		}, "\n")
	}
	sseText := func(msgID, text string) string {
		return strings.Join([]string{
			fmt.Sprintf(`data: {"type":"message_start","message":{"id":%q,"usage":{"input_tokens":12}}}`, msgID),
			"",
			`data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
			"",
			fmt.Sprintf(`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":%q}}`, text),
			"",
			`data: {"type":"content_block_stop","index":0}`,
			"",
			`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":6}}`,
			"",
			`data: {"type":"message_stop"}`,
			"",
		}, "\n")
	}

	t.Run("claude-sonnet-5_sends_explicit_disabled_and_records_control_provenance", func(t *testing.T) {
		var (
			mu       sync.Mutex
			requests []map[string]any
		)
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			body, _ := io.ReadAll(r.Body)
			_ = r.Body.Close()
			var payload map[string]any
			if err := json.Unmarshal(body, &payload); err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			mu.Lock()
			requests = append(requests, payload)
			call := len(requests)
			mu.Unlock()

			w.Header().Set("Content-Type", "text/event-stream")
			switch call {
			case 1:
				_, _ = io.WriteString(w, sseToolCall("work_1", "msg_1", "work", `{}`))
			case 2:
				_, _ = io.WriteString(w, sseText("msg_2", "done in text"))
			case 3:
				_, _ = io.WriteString(w, sseToolCall("done_1", "msg_3", "done", `{"message":"finished"}`))
			default:
				http.Error(w, "unexpected call", http.StatusBadRequest)
			}
		}))
		defer server.Close()

		client := &anthropic.Client{
			HTTPClient:     server.Client(),
			BaseURL:        server.URL,
			ModelName:      "claude-sonnet-5",
			MaxTokens:      1024,
			ThinkingMode:   "adaptive",
			ThinkingEffort: "xhigh",
			MaxRetries:     1,
		}
		workTool := tools.Func[struct{}]("work", "work", func(context.Context, struct{}, *tools.Container) (any, error) {
			return "ok", nil
		})
		doneTool := tools.Func[struct {
			Message string `json:"message"`
		}]("done", "finish", func(_ context.Context, args struct {
			Message string `json:"message"`
		}, _ *tools.Container) (any, error) {
			return nil, tools.TaskComplete(args.Message)
		})
		ag, err := agent.New(agent.Config{
			LLM:              client,
			Tools:            []tools.Tool{workTool, doneTool},
			ToolChoice:       llm.ToolChoice("auto"),
			RequireDoneTool:  true,
			MaxIterations:    6,
			Warningf:         func(string, ...any) {},
			QueryIDGenerator: func() string { return "q-s5" },
		})
		if err != nil {
			t.Fatalf("agent.New: %v", err)
		}

		var final string
		sawControlProvenance := false
		for env := range ag.QueryStreamEnveloped(context.Background(), llm.TextContent("run")) {
			if env.FrameID == "q-s5/frame/3" && env.RequestControlRelation == agent.RequestControlRequireDoneDisableThinking && env.RequestControlSourceFrameID == "q-s5/frame/2" {
				sawControlProvenance = true
			}
			switch ev := env.Event.(type) {
			case agent.FinalResponseEvent:
				final = ev.Content
			case agent.ErrorEvent:
				t.Fatalf("unexpected error event: %+v", ev)
			}
		}
		if final != "done in text\n\nfinished" {
			t.Fatalf("final = %q, want %q", final, "done in text\n\nfinished")
		}
		if !sawControlProvenance {
			t.Fatalf("expected truthful RequireDone disable-thinking provenance on frame 3")
		}

		mu.Lock()
		defer mu.Unlock()
		if len(requests) != 3 {
			t.Fatalf("request count = %d, want 3", len(requests))
		}
		// Frame 3 (RequireDone recovery) must explicitly disable thinking and force tool_choice=any.
		thinking3, _ := requests[2]["thinking"].(map[string]any)
		if thinking3["type"] != "disabled" {
			t.Fatalf("frame 3 thinking = %#v, want disabled", requests[2]["thinking"])
		}
		if _, hasOC := requests[2]["output_config"]; hasOC {
			t.Fatalf("frame 3 output_config = %#v, want omitted", requests[2]["output_config"])
		}
		tc3, _ := requests[2]["tool_choice"].(map[string]any)
		if tc3["type"] != "any" {
			t.Fatalf("frame 3 tool_choice = %#v, want any", requests[2]["tool_choice"])
		}
	})

	// Non-disableable models (claude-opus-5-5 and claude-sonnet-5-5) must degrade
	// gracefully: keep thinking active, keep tool_choice=auto, emit an observable
	// warning, and NEVER stamp bogus RequestControlRequireDoneDisableThinking provenance.
	for _, modelName := range []string{"claude-opus-5-5", "claude-sonnet-5-5"} {
		modelName := modelName
		t.Run(modelName+"_degrades_gracefully_with_truthful_provenance", func(t *testing.T) {
			var (
				mu       sync.Mutex
				requests []map[string]any
			)
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				body, _ := io.ReadAll(r.Body)
				_ = r.Body.Close()
				var payload map[string]any
				if err := json.Unmarshal(body, &payload); err != nil {
					http.Error(w, err.Error(), http.StatusBadRequest)
					return
				}
				mu.Lock()
				requests = append(requests, payload)
				call := len(requests)
				mu.Unlock()

				w.Header().Set("Content-Type", "text/event-stream")
				switch call {
				case 1:
					_, _ = io.WriteString(w, sseToolCall("work_1", "msg_1", "work", `{}`))
				case 2:
					_, _ = io.WriteString(w, sseText("msg_2", "done in text"))
				case 3:
					_, _ = io.WriteString(w, sseToolCall("done_1", "msg_3", "done", `{"message":"finished after reminder"}`))
				default:
					http.Error(w, "unexpected call", http.StatusBadRequest)
				}
			}))
			defer server.Close()

			client := &anthropic.Client{
				HTTPClient:     server.Client(),
				BaseURL:        server.URL,
				ModelName:      modelName,
				MaxTokens:      1024,
				ThinkingMode:   "adaptive",
				ThinkingEffort: "xhigh",
				MaxRetries:     1,
			}
			workTool := tools.Func[struct{}]("work", "work", func(context.Context, struct{}, *tools.Container) (any, error) {
				return "ok", nil
			})
			doneTool := tools.Func[struct {
				Message string `json:"message"`
			}]("done", "finish", func(_ context.Context, args struct {
				Message string `json:"message"`
			}, _ *tools.Container) (any, error) {
				return nil, tools.TaskComplete(args.Message)
			})
			ag, err := agent.New(agent.Config{
				LLM:              client,
				Tools:            []tools.Tool{workTool, doneTool},
				ToolChoice:       llm.ToolChoice("auto"),
				RequireDoneTool:  true,
				MaxIterations:    6,
				Warningf:         func(string, ...any) {},
				QueryIDGenerator: func() string { return "q-nodeg" },
			})
			if err != nil {
				t.Fatalf("agent.New: %v", err)
			}

			var final string
			sawDegradedWarn := false
			for env := range ag.QueryStreamEnveloped(context.Background(), llm.TextContent("run")) {
				if env.RequestControlRelation == agent.RequestControlRequireDoneDisableThinking || env.RequestControlSourceFrameID != "" {
					t.Fatalf("unexpected disable-thinking control provenance on %s: rel=%q source=%q frame=%q",
						modelName, env.RequestControlRelation, env.RequestControlSourceFrameID, env.FrameID)
				}
				switch ev := env.Event.(type) {
				case agent.WarnEvent:
					if ev.Kind == "require_done_capability_degraded" {
						sawDegradedWarn = true
					}
				case agent.FinalResponseEvent:
					final = ev.Content
				case agent.ErrorEvent:
					t.Fatalf("unexpected error event on %s: %+v", modelName, ev)
				}
			}
			if final != "done in text\n\nfinished after reminder" {
				t.Fatalf("final = %q, want %q", final, "done in text\n\nfinished after reminder")
			}
			if !sawDegradedWarn {
				t.Fatalf("expected observable require_done_capability_degraded warning for %s", modelName)
			}

			mu.Lock()
			defer mu.Unlock()
			if len(requests) != 3 {
				t.Fatalf("request count = %d, want 3", len(requests))
			}
			// Frame 3 must keep adaptive+xhigh and tool_choice=auto.
			thinking3, _ := requests[2]["thinking"].(map[string]any)
			if thinking3["type"] != "adaptive" {
				t.Fatalf("frame 3 thinking = %#v, want adaptive", requests[2]["thinking"])
			}
			oc3, _ := requests[2]["output_config"].(map[string]any)
			if oc3["effort"] != "xhigh" {
				t.Fatalf("frame 3 output_config = %#v, want effort=xhigh", requests[2]["output_config"])
			}
			tc3, _ := requests[2]["tool_choice"].(map[string]any)
			if tc3["type"] != "auto" {
				t.Fatalf("frame 3 tool_choice = %#v, want auto", requests[2]["tool_choice"])
			}
		})
	}
}
