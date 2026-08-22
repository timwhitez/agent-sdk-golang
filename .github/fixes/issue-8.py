from pathlib import Path

errors = Path("sdk/llm/errors.go")
text = errors.read_text()
append = r'''

// IncompleteStreamError reports that a streaming provider ended transport
// delivery without the explicit terminal event required by the SDK contract.
// Partial content may still be returned alongside this error.
type IncompleteStreamError struct {
	Provider string
	Model    string
	Message  string
}

func (e *IncompleteStreamError) Error() string {
	if e == nil {
		return "<nil>"
	}
	provider := e.Provider
	if provider == "" {
		provider = "provider"
	}
	message := e.Message
	if message == "" {
		message = "stream closed before terminal event"
	}
	if e.Model != "" {
		return fmt.Sprintf("%s stream incomplete for model %s: %s", provider, e.Model, message)
	}
	return fmt.Sprintf("%s stream incomplete: %s", provider, message)
}
'''
if "type IncompleteStreamError struct" in text:
    raise SystemExit("IncompleteStreamError already exists")
errors.write_text(text + append)

agent = Path("sdk/agent/agent.go")
text = agent.read_text()
old = '''\t\tresponseID := ""
\t\tstreamedText := false
'''
new = '''\t\tresponseID := ""
\t\tsawDone := false
\t\tstreamedText := false
'''
if text.count(old) != 1:
    raise SystemExit(f"sawDone declaration anchor count={text.count(old)}")
text = text.replace(old, new)
old = '''\t\t\tcase llm.StreamDoneEvent:
\t\t\t\tstopReason = e.StopReason
'''
new = '''\t\t\tcase llm.StreamDoneEvent:
\t\t\t\tstopReason = e.StopReason
\t\t\t\tsawDone = true
'''
if text.count(old) != 1:
    raise SystemExit(f"done event anchor count={text.count(old)}")
text = text.replace(old, new)
old = '''\t\t\t\tif !ok {
\t\t\t\t\tif err := metadata.flush(processStreamEvent); err != nil {
\t\t\t\t\t\treturn finishProviderStage(partialCompletion(), streamedText, err)
\t\t\t\t\t}
\t\t\t\t\treturn finishProviderStage(partialCompletion(), streamedText, nil)
\t\t\t\t}
'''
new = '''\t\t\t\tif !ok {
\t\t\t\t\tif err := metadata.flush(processStreamEvent); err != nil {
\t\t\t\t\t\treturn finishProviderStage(partialCompletion(), streamedText, err)
\t\t\t\t\t}
\t\t\t\t\tif !sawDone {
\t\t\t\t\t\treturn finishProviderStage(partialCompletion(), streamedText, &llm.IncompleteStreamError{
\t\t\t\t\t\t\tProvider: a.llm.Provider(),
\t\t\t\t\t\t\tModel:    a.llm.Model(),
\t\t\t\t\t\t\tMessage:  "provider event channel closed before StreamDoneEvent",
\t\t\t\t\t\t})
\t\t\t\t\t}
\t\t\t\t\treturn finishProviderStage(partialCompletion(), streamedText, nil)
\t\t\t\t}
'''
if text.count(old) != 1:
    raise SystemExit(f"stream close anchor count={text.count(old)}")
agent.write_text(text.replace(old, new))

Path("sdk/agent/agent_stream_terminal_contract_test.go").write_text(r'''package agent

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
''')
