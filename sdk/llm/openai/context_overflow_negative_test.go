package openai

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/agent"
	"github.com/timwhitez/agent-sdk-golang/sdk/agent/compaction"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

// negativeStep answers the n-th main (non-summary) request. It returns false
// when the step did not write a response and the default success applies.
type negativeStep func(w http.ResponseWriter, r *http.Request, stream bool, responses bool) bool

type negativeRun struct {
	main, summary                     int
	compactions, recoveries, errors   int
	finals                            int
	lastErrorKind, lastErrorMessage   string
	lastErrorStatus                   int
	historyHasSummary, historyHasUser bool
}

func writeOpenAISuccess(w http.ResponseWriter, stream, responses bool, content string) {
	switch {
	case responses && stream:
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprintf(w, "data: {\"type\":\"response.output_text.delta\",\"delta\":%q}\n\n", content)
		fmt.Fprintf(w, "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_ok\",\"status\":\"completed\",\"output\":[{\"type\":\"message\",\"content\":[{\"type\":\"output_text\",\"text\":%q}]}],\"usage\":{\"input_tokens\":10,\"output_tokens\":1,\"total_tokens\":11}}}\n\n", content)
	case responses:
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintf(w, `{"id":"resp_ok","status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":%q}]}],"usage":{"input_tokens":10,"output_tokens":1,"total_tokens":11}}`, content)
	case stream:
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"content\":%q},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":1,\"total_tokens\":11}}\n\ndata: [DONE]\n\n", content)
	default:
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintf(w, `{"choices":[{"message":{"role":"assistant","content":%q},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":1,"total_tokens":11}}`, content)
	}
}

func negativeHistory() []llm.Message {
	history := []llm.Message{llm.NewSystemMessage("system")}
	for i := 0; i < 8; i++ {
		history = append(history, llm.NewUserMessage(strings.Repeat("earlier request ", 60)), llm.NewAssistantMessage(strings.Repeat("earlier answer ", 60), nil))
	}
	return history
}

// runOverflowNegative drives a real Agent through a real OpenAI Chat or
// Responses client against an httptest server. Summary requests always
// succeed and every main request after the scripted ones succeeds, so any
// wrongly admitted overflow recovery would show up as a summary request, a
// CompactionEvent and an extra main request.
func runOverflowNegative(t *testing.T, responses bool, steps []negativeStep, disableRecovery bool, ctx context.Context) negativeRun {
	t.Helper()
	var mu sync.Mutex
	var run negativeRun
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		data, _ := io.ReadAll(r.Body)
		body := string(data)
		stream := strings.Contains(body, `"stream":true`)
		mu.Lock()
		if strings.Contains(body, "operational checkpoint") {
			run.summary++
			mu.Unlock()
			writeOpenAISuccess(w, stream, responses, overflowTestSummary())
			return
		}
		run.main++
		n := run.main
		mu.Unlock()
		if n <= len(steps) && steps[n-1] != nil && steps[n-1](w, r, stream, responses) {
			return
		}
		writeOpenAISuccess(w, stream, responses, "ok")
	}))
	defer server.Close()
	var model llm.ChatModel = &ChatClient{BaseURL: server.URL, ModelName: "m", MaxRetries: 1}
	if responses {
		model = &ResponsesClient{BaseURL: server.URL, ModelName: "m", MaxRetries: 1}
	}
	ag, err := agent.New(agent.Config{
		LLM:                            model,
		InitialMessages:                negativeHistory(),
		InvokeRetryMaxAttempts:         1,
		Compaction:                     &compaction.Config{Enabled: true, ContextWindow: 100000, ThresholdRatio: 0.85},
		DisableContextOverflowRecovery: disableRecovery,
		Warningf:                       func(string, ...any) {},
	})
	if err != nil {
		t.Fatal(err)
	}
	for env := range ag.QueryStreamEnveloped(ctx, llm.TextContent("current request")) {
		switch e := env.Event.(type) {
		case agent.ErrorEvent:
			run.errors++
			run.lastErrorKind, run.lastErrorMessage, run.lastErrorStatus = e.Kind, e.Message, e.StatusCode
		case agent.CompactionEvent:
			run.compactions++
		case agent.WarnEvent:
			if e.Kind == "context_overflow_recovery" {
				run.recoveries++
			}
		case agent.FinalResponseEvent:
			run.finals++
		}
	}
	users := 0
	for _, m := range ag.Messages() {
		text := m.Content.PlainText()
		if strings.Contains(text, "prior work summarized") {
			run.historyHasSummary = true
		}
		if m.Role == llm.RoleUser && strings.Contains(text, "current request") {
			users++
		}
	}
	run.historyHasUser = users == 1
	mu.Lock()
	defer mu.Unlock()
	return run
}

func statusStep(status int, contentType, body string) negativeStep {
	return func(w http.ResponseWriter, _ *http.Request, _, _ bool) bool {
		w.Header().Set("Content-Type", contentType)
		w.WriteHeader(status)
		io.WriteString(w, body)
		return true
	}
}

// sseStep writes raw SSE bytes (Chat or Responses variant) and ends the body.
func sseStep(chat, responses string) negativeStep {
	return func(w http.ResponseWriter, _ *http.Request, _ bool, isResponses bool) bool {
		w.Header().Set("Content-Type", "text/event-stream")
		if isResponses {
			io.WriteString(w, responses)
		} else {
			io.WriteString(w, chat)
		}
		return true
	}
}

// G1: provider failures that are not typed context overflow never enter the
// typed overflow recovery. The run with recovery enabled makes exactly the
// requests of the run with recovery disabled, without a summary request or a
// compaction, and the user's input stays in history exactly once. The typed
// control proves the same harness does recover.
func TestOpenAIAgentNonOverflowFailuresNeverEnterOverflowRecovery(t *testing.T) {
	contextMessage := "This model's maximum context length is 8192 tokens. However, your messages resulted in 9000 tokens."
	jsonError := func(typ, code string) string {
		codeJSON := "null"
		if code != "" {
			codeJSON = fmt.Sprintf("%q", code)
		}
		return fmt.Sprintf(`{"error":{"message":%q,"type":%q,"param":"messages","code":%s}}`, contextMessage, typ, codeJSON)
	}
	sseLimitChat := strings.Repeat("data: {\"choices\":\n\n", 40) + "data: [DONE]\n\n"
	sseLimitResponses := strings.Repeat("data: {\"type\":\n\n", 40) + `data: {"type":"response.completed","response":{"status":"completed"}}` + "\n\n"
	partialChat := `data: {"id":"resp_1","choices":[{"delta":{"content":"partial"}}]}` + "\n\n"
	partialResponses := `data: {"type":"response.output_text.delta","delta":"partial"}` + "\n\n"
	lengthChat := `data: {"choices":[{"delta":{"content":"part"},"finish_reason":"length"}],"usage":{"prompt_tokens":10,"completion_tokens":1,"total_tokens":11}}` + "\n\ndata: [DONE]\n\n"
	lengthResponses := `data: {"type":"response.output_text.delta","delta":"part"}` + "\n\n" +
		`data: {"type":"response.incomplete","response":{"id":"resp_len","status":"incomplete","incomplete_details":{"reason":"max_output_tokens"},"output":[{"type":"message","content":[{"type":"output_text","text":"part"}]}],"usage":{"input_tokens":10,"output_tokens":1,"total_tokens":11}}}` + "\n\n"

	cases := []struct {
		name string
		step negativeStep
		// wantMain is the absolute request count, equal in both runs.
		wantMain int
		// wantFinal reports whether the turn completes (max-output continuation).
		wantFinal bool
	}{
		{"http 403 context message", statusStep(http.StatusForbidden, "application/json", jsonError("permission_error", "")), 1, false},
		{"http 413 context message", statusStep(http.StatusRequestEntityTooLarge, "application/json", jsonError("invalid_request_error", "")), 1, false},
		{"http 413 plain text", statusStep(http.StatusRequestEntityTooLarge, "text/plain", "413 Request Entity Too Large: prompt tokens exceed the context window"), 1, false},
		{"http 422 context message", statusStep(http.StatusUnprocessableEntity, "application/json", jsonError("invalid_request_error", "invalid_value")), 1, false},
		{"http 400 context message without code", statusStep(http.StatusBadRequest, "application/json", jsonError("invalid_request_error", "")), 1, false},
		{"sse resource limit before output", sseStep(sseLimitChat, sseLimitResponses), 1, false},
		{"sse resource limit after output", sseStep(partialChat+sseLimitChat, partialResponses+sseLimitResponses), 1, false},
		{"stream EOF before output", sseStep("", ""), 1, false},
		{"stream EOF after partial output", sseStep(partialChat, partialResponses), 1, false},
		{"max output length stop", sseStep(lengthChat, lengthResponses), 2, true},
	}
	for _, client := range []string{"chat", "responses"} {
		responses := client == "responses"
		for _, tc := range cases {
			t.Run(client+"/"+tc.name, func(t *testing.T) {
				steps := []negativeStep{tc.step}
				enabled := runOverflowNegative(t, responses, steps, false, context.Background())
				disabled := runOverflowNegative(t, responses, steps, true, context.Background())
				if enabled.main != disabled.main || enabled.main != tc.wantMain {
					t.Fatalf("main requests: enabled=%d disabled=%d want %d (enabled run %+v)", enabled.main, disabled.main, tc.wantMain, enabled)
				}
				if enabled.summary != 0 || enabled.compactions != 0 || enabled.recoveries != 0 || enabled.historyHasSummary {
					t.Fatalf("entered overflow recovery: %+v", enabled)
				}
				if !enabled.historyHasUser {
					t.Fatalf("user input not in history exactly once: %+v", enabled)
				}
				if tc.wantFinal {
					if enabled.finals != 1 || enabled.errors != 0 {
						t.Fatalf("max-output continuation did not complete: %+v", enabled)
					}
				} else if enabled.errors != 1 || enabled.finals != 0 {
					t.Fatalf("failure did not end the turn with one error: %+v", enabled)
				}
			})
		}
	}

	// Control: the documented structured code over the same harness recovers
	// once, so the negatives above are not vacuous.
	for _, client := range []string{"chat", "responses"} {
		t.Run(client+"/typed control", func(t *testing.T) {
			run := runOverflowNegative(t, client == "responses", []negativeStep{statusStep(http.StatusBadRequest, "application/json", overflowBody)}, false, context.Background())
			if run.main != 2 || run.summary != 1 || run.compactions != 1 || run.recoveries != 1 || run.finals != 1 || !run.historyHasSummary {
				t.Fatalf("typed overflow was not recovered once: %+v", run)
			}
		})
	}
}

// G1: canceling the turn while the provider holds the request is terminal;
// it never becomes an overflow recovery, a summary request or a retry.
func TestOpenAIAgentCanceledRequestNeverEntersOverflowRecovery(t *testing.T) {
	for _, client := range []string{"chat", "responses"} {
		t.Run(client, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			entered := make(chan struct{})
			var once sync.Once
			hold := func(w http.ResponseWriter, r *http.Request, _, _ bool) bool {
				once.Do(func() { close(entered) })
				<-r.Context().Done()
				return true
			}
			go func() {
				<-entered
				cancel()
			}()
			run := runOverflowNegative(t, client == "responses", []negativeStep{hold}, false, ctx)
			if run.main != 1 || run.summary != 0 || run.compactions != 0 || run.recoveries != 0 || run.historyHasSummary {
				t.Fatalf("canceled request entered recovery or was retried: %+v", run)
			}
			if run.errors != 1 || run.finals != 0 {
				t.Fatalf("cancel did not end the turn with one error: %+v", run)
			}
		})
	}
}
