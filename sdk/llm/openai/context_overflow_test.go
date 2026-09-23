package openai

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/agent"
	"github.com/timwhitez/agent-sdk-golang/sdk/agent/compaction"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

const (
	overflowBody = `{"error":{"message":"This model's maximum context length is 8192 tokens.","type":"invalid_request_error","param":"messages","code":"context_length_exceeded"}}`
	// Same status and message text without the structured code: not typed.
	untypedBody = `{"error":{"message":"This model's maximum context length is 8192 tokens.","type":"invalid_request_error","param":"messages","code":null}}`
	otherBody   = `{"error":{"message":"bad param","type":"invalid_request_error","code":"invalid_value"}}`
)

func errorServer(t *testing.T, body string) *httptest.Server {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		io.WriteString(w, body)
	}))
	t.Cleanup(server.Close)
	return server
}

func streamedError(events <-chan llm.StreamEvent) error {
	for event := range events {
		if e, ok := event.(llm.StreamErrorEvent); ok {
			return e.AsError()
		}
	}
	return nil
}

// Only the documented structured code types an overflow, on buffered and
// streamed requests of both OpenAI clients; status and message text do not.
func TestOpenAIContextOverflowIsTypedOnlyByStructuredCode(t *testing.T) {
	req := llm.InvokeRequest{Messages: []llm.Message{llm.NewUserMessage("hi")}}
	for _, tc := range []struct {
		body string
		want bool
	}{{overflowBody, true}, {untypedBody, false}, {otherBody, false}} {
		server := errorServer(t, tc.body)
		chat := &ChatClient{BaseURL: server.URL, ModelName: "m", MaxRetries: 1}
		responses := &ResponsesClient{BaseURL: server.URL, ModelName: "m", MaxRetries: 1}
		for name, invoke := range map[string]func() error{
			"chat/buffered": func() error { _, err := chat.Invoke(context.Background(), req); return err },
			"chat/stream": func() error {
				events, err := chat.InvokeStream(context.Background(), req)
				if err != nil {
					return err
				}
				return streamedError(events)
			},
			"responses/buffered": func() error { _, err := responses.Invoke(context.Background(), req); return err },
			"responses/stream": func() error {
				events, err := responses.InvokeStream(context.Background(), req)
				if err != nil {
					return err
				}
				return streamedError(events)
			},
		} {
			err := invoke()
			if err == nil || llm.IsContextOverflow(err) != tc.want {
				t.Fatalf("%s body=%s: err=%v typed=%v want %v", name, tc.body, err, llm.IsContextOverflow(err), tc.want)
			}
		}
	}
}

// A Responses stream that fails mid-stream carries the same typing.
func TestResponsesStreamErrorEventContextOverflow(t *testing.T) {
	for _, tc := range []struct {
		code string
		want bool
	}{{`"context_length_exceeded"`, true}, {`"server_error"`, false}} {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			fmt.Fprintf(w, "data: {\"type\":\"error\",\"error\":{\"code\":%s,\"message\":\"failed\"}}\n\n", tc.code)
		}))
		client := &ResponsesClient{BaseURL: server.URL, ModelName: "m", MaxRetries: 1}
		events, err := client.InvokeStream(context.Background(), llm.InvokeRequest{Messages: []llm.Message{llm.NewUserMessage("hi")}})
		if err != nil {
			t.Fatal(err)
		}
		got := streamedError(events)
		server.Close()
		if got == nil || llm.IsContextOverflow(got) != tc.want {
			t.Fatalf("code=%s err=%v typed=%v", tc.code, got, llm.IsContextOverflow(got))
		}
	}
}

// End to end through a real ChatClient and Agent: the provider's typed 400 is
// recovered by one compaction (summary request) and one smaller retry.
func TestAgentRecoversOpenAITypedContextOverflowOverHTTP(t *testing.T) {
	var bodies []string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		data, _ := io.ReadAll(r.Body)
		body := string(data)
		bodies = append(bodies, body)
		reply := func(content string) {
			if strings.Contains(body, `"stream":true`) {
				w.Header().Set("Content-Type", "text/event-stream")
				fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"content\":%q},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":1,\"total_tokens\":11}}\n\ndata: [DONE]\n\n", content)
				return
			}
			w.Header().Set("Content-Type", "application/json")
			fmt.Fprintf(w, `{"choices":[{"message":{"role":"assistant","content":%q},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":1,"total_tokens":11}}`, content)
		}
		switch {
		case strings.Contains(body, "operational checkpoint"):
			reply(overflowTestSummary())
		case len(bodies) == 1:
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusBadRequest)
			io.WriteString(w, overflowBody)
		default:
			reply("ok")
		}
	}))
	defer server.Close()
	history := []llm.Message{llm.NewSystemMessage("system")}
	for i := 0; i < 8; i++ {
		history = append(history, llm.NewUserMessage(strings.Repeat("earlier request ", 60)), llm.NewAssistantMessage(strings.Repeat("earlier answer ", 60), nil))
	}
	ag, err := agent.New(agent.Config{
		LLM:                    &ChatClient{BaseURL: server.URL, ModelName: "m", MaxRetries: 1},
		InitialMessages:        history,
		InvokeRetryMaxAttempts: 1,
		Compaction:             &compaction.Config{Enabled: true, ContextWindow: 100000, ThresholdRatio: 0.85},
		Warningf:               func(string, ...any) {},
	})
	if err != nil {
		t.Fatal(err)
	}
	out, err := ag.Query(context.Background(), "current request")
	if err != nil || out != "ok" {
		t.Fatalf("out=%q err=%v", out, err)
	}
	if len(bodies) != 3 || !strings.Contains(bodies[1], "operational checkpoint") || len(bodies[2]) >= len(bodies[0]) {
		t.Fatalf("requests=%d sizes=%v", len(bodies), []int{len(bodies[0]), len(bodies[len(bodies)-1])})
	}
}

func overflowTestSummary() string {
	var b strings.Builder
	b.WriteString("<summary>\n")
	for _, section := range []string{
		"Current Objective and Latest User Request",
		"Authoritative Current State",
		"Completed Work",
		"In-Progress and Remaining Work",
		"Exact External State",
		"Errors, Failed Attempts, and Successful Recovery",
		"Verification Already Run and Still Required",
		"Conflicts, Uncertainty, and Facts That Must Be Re-read",
	} {
		fmt.Fprintf(&b, "## %s\nprior work summarized\n\n", section)
	}
	b.WriteString("</summary>")
	return b.String()
}
