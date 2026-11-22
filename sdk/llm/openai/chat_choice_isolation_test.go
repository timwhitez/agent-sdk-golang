package openai

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

func TestChatChoiceZeroOwnsAllStreamFields(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for _, event := range []string{
			`{"choices":[{"index":1,"delta":{"content":"other","reasoning_content":"other thought","tool_calls":[{"index":0,"id":"other_call","function":{"name":"other","arguments":"{}"}}]},"finish_reason":"length"}]}`,
			`{"choices":[{"index":0,"delta":{"content":"answer","reasoning_content":"thought","tool_calls":[{"index":0,"id":"chosen_call","function":{"name":"chosen","arguments":"{}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":10,"completion_tokens":20,"total_tokens":30}}`,
			`{"choices":[{"index":1,"delta":{"content":"again"},"finish_reason":"length"}]}`,
			`[DONE]`,
		} {
			fmt.Fprint(w, "data: "+event+"\n\n")
		}
	}))
	defer server.Close()
	client := &ChatClient{BaseURL: server.URL, ModelName: "fixture", MaxRetries: 1, Extra: map[string]any{"n": 2}}
	events, err := client.InvokeStream(t.Context(), llm.InvokeRequest{Messages: []llm.Message{llm.NewUserMessage("choose")}})
	if err != nil {
		t.Fatal(err)
	}
	var text, thinking strings.Builder
	var calls []llm.StreamToolCallDeltaEvent
	var stop string
	var total int
	for event := range events {
		switch e := event.(type) {
		case llm.StreamTextDeltaEvent:
			text.WriteString(e.Delta)
		case llm.StreamThinkingDeltaEvent:
			thinking.WriteString(e.Delta)
		case llm.StreamToolCallDeltaEvent:
			calls = append(calls, e)
		case llm.StreamUsageEvent:
			total = e.Usage.TotalTokens
		case llm.StreamDoneEvent:
			stop = e.StopReason
		case llm.StreamErrorEvent:
			t.Fatal(e.AsError())
		}
	}
	if text.String() != "answer" || thinking.String() != "thought" || len(calls) != 1 || calls[0].ID != "chosen_call" || stop != "tool_calls" || total != 30 {
		t.Fatalf("text=%q thinking=%q calls=%+v stop=%q total=%d", text.String(), thinking.String(), calls, stop, total)
	}
}

func TestChatBufferedSelectsChoiceZeroByIndex(t *testing.T) {
	completion, err := parseChatCompletion([]byte(`{"choices":[{"index":1,"message":{"content":"other"},"finish_reason":"length"},{"index":0,"message":{"content":"answer"},"finish_reason":"stop"}]}`))
	if err != nil || completion.PlainText() != "answer" || completion.StopReason != "stop" {
		t.Fatalf("completion=%+v err=%v", completion, err)
	}
}
