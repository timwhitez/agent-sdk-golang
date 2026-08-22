package agent

import (
	"context"
	"errors"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

type terminalContractModel struct {
	events []llm.StreamEvent
}

func (m *terminalContractModel) Provider() string { return "terminal-test" }
func (m *terminalContractModel) Model() string    { return "terminal-model" }
func (m *terminalContractModel) Invoke(context.Context, llm.InvokeRequest) (*llm.Completion, error) {
	return nil, errors.New("unexpected non-stream invoke")
}
func (m *terminalContractModel) InvokeStream(context.Context, llm.InvokeRequest) (<-chan llm.StreamEvent, error) {
	out := make(chan llm.StreamEvent, len(m.events))
	for _, event := range m.events {
		out <- event
	}
	close(out)
	return out, nil
}

func invokeTerminalContract(t *testing.T, events ...llm.StreamEvent) (*llm.Completion, error) {
	t.Helper()
	agent, err := New(Config{LLM: &terminalContractModel{events: events}})
	if err != nil {
		t.Fatal(err)
	}
	out := make(chan Event, 32)
	completion, _, err := agent.invokeCompletion(context.Background(), llm.InvokeRequest{}, out)
	return completion, err
}

func TestAgentRejectsTextStreamClosedWithoutDone(t *testing.T) {
	completion, err := invokeTerminalContract(t, llm.StreamTextDeltaEvent{Delta: "partial"})
	var incomplete *llm.IncompleteStreamError
	if !errors.As(err, &incomplete) {
		t.Fatalf("error = %v, want IncompleteStreamError", err)
	}
	if completion == nil || completion.PlainText() != "partial" {
		t.Fatalf("partial completion = %#v", completion)
	}
}

func TestAgentRejectsIncompleteToolCallStreamClosedWithoutDone(t *testing.T) {
	completion, err := invokeTerminalContract(t,
		llm.StreamToolCallDeltaEvent{Index: 0, ID: "call_1", NameDelta: "read"},
		llm.StreamToolCallDeltaEvent{Index: 0, ArgumentsDelta: `{"path":`},
	)
	var incomplete *llm.IncompleteStreamError
	if !errors.As(err, &incomplete) {
		t.Fatalf("error = %v, want IncompleteStreamError", err)
	}
	if completion == nil || len(completion.ToolCalls) != 1 || completion.ToolCalls[0].Function.Arguments != `{"path":` {
		t.Fatalf("partial tool completion = %#v", completion)
	}
}

func TestAgentAcceptsExplicitDoneBeforeClose(t *testing.T) {
	completion, err := invokeTerminalContract(t,
		llm.StreamTextDeltaEvent{Delta: "complete"},
		llm.StreamDoneEvent{StopReason: "stop"},
	)
	if err != nil {
		t.Fatalf("error = %v", err)
	}
	if completion == nil || completion.PlainText() != "complete" || completion.StopReason != "stop" {
		t.Fatalf("completion = %#v", completion)
	}
}

func TestAgentAcceptsMetadataOnlyStreamWithDone(t *testing.T) {
	completion, err := invokeTerminalContract(t,
		llm.StreamResponseEvent{ResponseID: "resp_1"},
		llm.StreamDoneEvent{StopReason: "stop"},
	)
	if err != nil {
		t.Fatalf("error = %v", err)
	}
	if completion == nil || completion.ResponseID != "resp_1" {
		t.Fatalf("completion = %#v", completion)
	}
}

func TestAgentPreservesProviderStreamError(t *testing.T) {
	providerErr := &llm.ProviderError{Provider: "terminal-test", Message: "broken stream"}
	_, err := invokeTerminalContract(t, llm.StreamErrorEvent{Err: providerErr})
	var got *llm.ProviderError
	if !errors.As(err, &got) || got != providerErr {
		t.Fatalf("error = %v, want original provider error", err)
	}
}
