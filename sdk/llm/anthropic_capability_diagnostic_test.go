package llm_test

import (
	"errors"
	"fmt"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm/anthropic"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
)

func TestAuditForcedChoiceCapabilityMessage(t *testing.T) {
	var sent atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		sent.Add(1)
		fmt.Fprint(w, `{"content":[{"type":"text","text":"ok"}],"usage":{"input_tokens":6,"output_tokens":1},"stop_reason":"end_turn"}`)
	}))
	defer srv.Close()
	c := &anthropic.Client{ModelName: "gateway", BaseURL: srv.URL, MaxRetries: 1, Capabilities: &anthropic.ModelCapabilities{DefaultThinkingOn: true, DisableMode: anthropic.ThinkingDisableExplicit, ForcedToolChoiceWhenThinkingDisabled: false}}
	if caps, known := c.ThinkingCapabilities(); !known || !caps.SupportsDisableThinking || caps.SupportsForcedToolChoice {
		t.Fatal("invalid capability fixture")
	}
	req := llm.InvokeRequest{Messages: []llm.Message{llm.NewUserMessage("choose")}, Tools: []llm.ToolDefinition{{Name: "work", Parameters: map[string]any{"type": "object"}}}, ToolChoice: "auto", DisableThinking: true}
	if _, err := c.Invoke(t.Context(), req); err != nil || sent.Load() != 1 {
		t.Fatalf("full disable control: sent=%d err=%v", sent.Load(), err)
	}
	req.ToolChoice = "required"
	req.DisableThinking = false
	for _, mode := range []string{"buffered", "streaming"} {
		t.Run(mode, func(t *testing.T) {
			var err error
			if mode == "buffered" {
				_, err = c.Invoke(t.Context(), req)
			} else {
				var events <-chan llm.StreamEvent
				events, err = c.InvokeStream(t.Context(), req)
				if err == nil {
					for e := range events {
						if stop, ok := e.(llm.StreamErrorEvent); ok {
							err = stop.AsError()
						}
					}
				}
			}
			var mismatch *llm.ThinkingCapabilityError
			if !errors.As(err, &mismatch) || mismatch.Control != "tool_choice" || sent.Load() != 1 {
				t.Fatalf("unexpected rejection: sent=%d err=%v", sent.Load(), err)
			}
			if strings.Contains(mismatch.Message, "does not support full DisableThinking") {
				t.Errorf("full disable succeeded and capability is true; emitted diagnostic: %s", mismatch.Message)
			}
		})
	}
}
