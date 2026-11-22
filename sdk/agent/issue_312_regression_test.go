package agent

import (
	"errors"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"testing"
)

func TestD1PartialProviderHistoryHasOwnedContent(t *testing.T) {
	partial := llm.Content{Blocks: []llm.ContentBlock{{Type: "text", Text: "original partial"}}}
	model := &frameScriptModel{invoke: func(llm.InvokeRequest) (*llm.Completion, error) {
		return &llm.Completion{Content: partial}, errors.New("generation interrupted")
	}}
	a, err := New(Config{LLM: model, InvokeRetryMaxAttempts: 1})
	if err != nil {
		t.Fatal(err)
	}
	if _, err = a.Query(t.Context(), "start"); err == nil {
		t.Fatal("expected interrupted response")
	}
	partial.Blocks[0].Text = "next response"
	history := a.Messages()
	if got := history[len(history)-1].PlainText(); got != "original partial" {
		t.Fatalf("history after provider reused its response buffer=%q; want original partial", got)
	}
}
