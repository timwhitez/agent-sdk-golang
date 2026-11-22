package agent

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

type auditTerminalStateStream struct {
	events       []llm.StreamEvent
	calls        int
	replayed     int
	replayedData []string
}

func (*auditTerminalStateStream) Provider() string { return "fixture" }
func (*auditTerminalStateStream) Model() string    { return "state-only" }
func (*auditTerminalStateStream) Invoke(context.Context, llm.InvokeRequest) (*llm.Completion, error) {
	return nil, errors.New("unexpected buffered call")
}
func (m *auditTerminalStateStream) InvokeStream(_ context.Context, req llm.InvokeRequest) (<-chan llm.StreamEvent, error) {
	m.calls++
	events := m.events
	if m.calls > 1 {
		for _, message := range req.Messages {
			if message.Role == llm.RoleAssistant && llm.HasProviderState(message.Content) {
				m.replayed++
				state, err := llm.ProviderStateFromContent(message.Content)
				if err != nil {
					return nil, err
				}
				for _, item := range state {
					m.replayedData = append(m.replayedData, string(item.Data))
				}
			}
		}
		events = []llm.StreamEvent{llm.StreamTextDeltaEvent{Delta: "ok"}, llm.StreamDoneEvent{StopReason: "stop"}}
	}
	ch := make(chan llm.StreamEvent, len(events))
	for _, event := range events {
		ch <- event
	}
	close(ch)
	return ch, nil
}

func TestAuditTerminalErrorsPreserveOpaqueOnlyProviderState(t *testing.T) {
	state := []llm.ProviderState{{Provider: "openai-responses", Kind: "response.output_item.v1", Data: json.RawMessage(`{"id":"rs_retained","type":"reasoning","encrypted_content":"opaque-ciphertext"}`)}}
	for _, terminal := range []string{"error", "missing_done", "buffered_error"} {
		t.Run(terminal, func(t *testing.T) {
			m := &auditTerminalStateStream{events: []llm.StreamEvent{
				llm.StreamProviderStateEvent{State: state},
			}}
			if terminal == "error" {
				m.events = append(m.events, llm.StreamErrorEvent{Err: errors.New("connection lost after state")})
			}
			var model llm.ChatModel = m
			if terminal == "buffered_error" {
				content, err := llm.WithProviderState(llm.Content{}, state)
				if err != nil {
					t.Fatal(err)
				}
				model = &frameScriptModel{invoke: func(req llm.InvokeRequest) (*llm.Completion, error) {
					m.calls++
					if m.calls == 1 {
						return &llm.Completion{Content: content}, errors.New("partial provider failure")
					}
					for _, message := range req.Messages {
						if message.Role == llm.RoleAssistant && llm.HasProviderState(message.Content) {
							m.replayed++
							state, err := llm.ProviderStateFromContent(message.Content)
							if err != nil {
								return nil, err
							}
							for _, item := range state {
								m.replayedData = append(m.replayedData, string(item.Data))
							}
						}
					}
					return &llm.Completion{Content: llm.TextContent("ok")}, nil
				}}
			}
			a, err := New(Config{LLM: model, InvokeRetryMaxAttempts: 1})
			if err != nil {
				t.Fatal(err)
			}
			if _, err := a.Query(t.Context(), "first"); err == nil {
				t.Fatal("expected terminal error")
			}
			retained := 0
			for _, message := range a.Messages() {
				if len(message.ToolCalls) != 0 || message.Role == llm.RoleTool {
					t.Fatal("failed partial tool call was published in history")
				}
				if llm.HasProviderState(message.Content) {
					retained++
					if message.PlainText() != "" {
						t.Fatal("opaque state rendered as text")
					}
					// Messages returns an owned snapshot. Mutating its blocks must
					// not alter the state replayed by the following request.
					message.Content.Blocks[0].Data = "caller mutation"
				}
			}
			if _, err := a.Query(t.Context(), "retry"); err != nil {
				t.Fatal(err)
			}
			if retained != 1 || m.replayed != 1 {
				t.Fatalf("terminal=%s retained=%d replayed=%d; want opaque state persisted and replayed exactly once", terminal, retained, m.replayed)
			}
			if len(m.replayedData) != 1 || m.replayedData[0] != string(state[0].Data) {
				t.Fatalf("replayed opaque state changed: %q", m.replayedData)
			}
		})
	}
}
