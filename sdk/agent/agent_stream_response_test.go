package agent

import (
	"context"
	"errors"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

type streamingResponseIDModel struct{}

func (m *streamingResponseIDModel) Provider() string { return "stub" }
func (m *streamingResponseIDModel) Model() string    { return "stub" }

func (m *streamingResponseIDModel) Invoke(_ context.Context, _ llm.InvokeRequest) (*llm.Completion, error) {
	return nil, errors.New("invoke should not be called")
}

func (m *streamingResponseIDModel) InvokeStream(_ context.Context, _ llm.InvokeRequest) (<-chan llm.StreamEvent, error) {
	ch := make(chan llm.StreamEvent, 4)
	go func() {
		defer close(ch)
		ch <- llm.StreamResponseEvent{ResponseID: "msg_stream_123"}
		ch <- llm.StreamTextDeltaEvent{Delta: "hello"}
		ch <- llm.StreamUsageEvent{Usage: llm.Usage{PromptTokens: 7, CompletionTokens: 5, TotalTokens: 12}}
		ch <- llm.StreamDoneEvent{StopReason: "stop"}
	}()
	return ch, nil
}

func TestInvokeCompletionPreservesStreamResponseID(t *testing.T) {
	model := &streamingResponseIDModel{}
	ag, err := New(Config{LLM: model})
	if err != nil {
		t.Fatalf("new agent: %v", err)
	}

	comp, streamedText, err := ag.invokeCompletion(context.Background(), llm.InvokeRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: llm.TextContent("hi")}},
	}, nil)
	if err != nil {
		t.Fatalf("invoke completion: %v", err)
	}
	if !streamedText {
		t.Fatalf("expected streamed text=true")
	}
	if comp.ResponseID != "msg_stream_123" {
		t.Fatalf("expected response id msg_stream_123, got %q", comp.ResponseID)
	}
	if comp.PlainText() != "hello" {
		t.Fatalf("expected streamed text hello, got %q", comp.PlainText())
	}
}

func TestQueryStreamCarriesResponseIDInMetadataEvents(t *testing.T) {
	model := &streamingResponseIDModel{}
	ag, err := New(Config{LLM: model})
	if err != nil {
		t.Fatalf("new agent: %v", err)
	}

	events := collectEvents(ag.QueryStream(context.Background(), llm.TextContent("hi")))
	usageResponseID := ""
	finalResponseID := ""
	finalText := ""
	for _, ev := range events {
		switch e := ev.(type) {
		case UsageEvent:
			usageResponseID = e.ResponseID
		case FinalResponseEvent:
			finalResponseID = e.ResponseID
			finalText = e.Content
		}
	}

	if usageResponseID != "msg_stream_123" {
		t.Fatalf("expected usage response id msg_stream_123, got %q", usageResponseID)
	}
	if finalResponseID != "msg_stream_123" {
		t.Fatalf("expected final response id msg_stream_123, got %q", finalResponseID)
	}
	if finalText != "hello" {
		t.Fatalf("expected final text hello, got %q", finalText)
	}
}
