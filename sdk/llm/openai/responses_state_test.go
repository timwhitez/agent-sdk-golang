package openai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

func TestParseResponsesPreservesOpaqueItemsWithoutRenderingEncryptedContent(t *testing.T) {
	const encrypted = "encrypted-reasoning-must-stay-opaque"
	payload := `{"id":"resp_state","status":"completed","output":[{"id":"rs_1","type":"reasoning","encrypted_content":"` + encrypted + `","phase":"analysis","summary":[{"type":"summary_text","text":"public summary"}]},{"id":"msg_1","type":"message","role":"assistant","phase":"final_answer","content":[{"type":"output_text","text":"visible answer"}]}]}`
	completion, err := parseResponses([]byte(payload))
	if err != nil {
		t.Fatal(err)
	}
	if completion.PlainText() != "visible answer" || strings.Contains(completion.PlainText(), encrypted) || strings.Contains(completion.Thinking, encrypted) {
		t.Fatalf("visible content leaked or lost opaque data: text=%q thinking=%q", completion.PlainText(), completion.Thinking)
	}
	if len(completion.ProviderState) != 2 {
		t.Fatalf("provider state = %#v, want both output items", completion.ProviderState)
	}
	if !strings.Contains(string(completion.ProviderState[0].Data), encrypted) || !strings.Contains(string(completion.ProviderState[1].Data), `"phase":"final_answer"`) {
		t.Fatalf("opaque output fields were lost: %#v", completion.ProviderState)
	}
}

func TestParseResponsesOnlyRetainsOpaqueStateForSuccessfulTerminals(t *testing.T) {
	tests := []struct {
		name      string
		payload   string
		wantState bool
	}{
		{
			name:      "supported incomplete",
			payload:   `{"status":"incomplete","incomplete_details":{"reason":"max_output_tokens"},"output":[{"type":"reasoning","encrypted_content":"partial"}]}`,
			wantState: true,
		},
		{
			name:    "failed",
			payload: `{"status":"failed","error":{"message":"failed"},"output":[{"type":"reasoning","encrypted_content":"failed"}]}`,
		},
		{
			name:    "unsupported incomplete",
			payload: `{"status":"incomplete","incomplete_details":{"reason":"unknown"},"output":[{"type":"reasoning","encrypted_content":"unknown"}]}`,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			completion, _ := parseResponses([]byte(tc.payload))
			if completion == nil {
				t.Fatal("missing partial completion")
			}
			if got := len(completion.ProviderState) > 0; got != tc.wantState {
				t.Fatalf("provider state retained = %t, want %t", got, tc.wantState)
			}
		})
	}
}

func TestResponsesBuildRequestRejectsMalformedOrUnboundedOpaqueState(t *testing.T) {
	useItems := true
	legacyMessages := false
	client := &ResponsesClient{ModelName: "test-model"}
	tests := []struct {
		name  string
		state []llm.ProviderState
	}{
		{
			name:  "null item",
			state: []llm.ProviderState{{Provider: responsesStateProvider, Kind: responsesOutputItemStateKind, Data: json.RawMessage(`null`)}},
		},
		{
			name:  "missing type",
			state: []llm.ProviderState{{Provider: responsesStateProvider, Kind: responsesOutputItemStateKind, Data: json.RawMessage(`{"id":"rs_1"}`)}},
		},
	}
	tooMany := make([]llm.ProviderState, maxResponsesOutputItems+1)
	for index := range tooMany {
		tooMany[index] = llm.ProviderState{Provider: responsesStateProvider, Kind: responsesOutputItemStateKind, Data: json.RawMessage(`{"type":"reasoning"}`)}
	}
	tests = append(tests, struct {
		name  string
		state []llm.ProviderState
	}{name: "too many items", state: tooMany})
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := client.buildRequest(llm.InvokeRequest{
				Messages:  []llm.Message{{Role: llm.RoleAssistant, ProviderState: tc.state}},
				Responses: &llm.ResponsesOptions{UseResponseItems: &useItems},
			})
			if err == nil {
				t.Fatal("invalid opaque provider state was accepted")
			}
		})
	}
	perMessage := make([]llm.ProviderState, 600)
	for index := range perMessage {
		perMessage[index] = llm.ProviderState{Provider: responsesStateProvider, Kind: responsesOutputItemStateKind, Data: json.RawMessage(`{"type":"reasoning"}`)}
	}
	_, err := client.buildRequest(llm.InvokeRequest{
		Messages: []llm.Message{
			{Role: llm.RoleAssistant, ProviderState: perMessage},
			{Role: llm.RoleAssistant, ProviderState: perMessage},
		},
		Responses: &llm.ResponsesOptions{UseResponseItems: &useItems},
	})
	if err == nil || !strings.Contains(err.Error(), "opaque output history exceeds") {
		t.Fatalf("cross-message opaque history limit error = %v", err)
	}
	_, err = client.buildRequest(llm.InvokeRequest{
		Messages: []llm.Message{{
			Role: llm.RoleUser,
			ProviderState: []llm.ProviderState{{
				Provider: responsesStateProvider,
				Kind:     responsesOutputItemStateKind,
				Data:     json.RawMessage(`{"type":"reasoning"}`),
			}},
		}},
		Responses: &llm.ResponsesOptions{UseResponseItems: &useItems},
	})
	if err == nil || !strings.Contains(err.Error(), "only valid on assistant") {
		t.Fatalf("non-assistant opaque state error = %v", err)
	}
	_, err = client.buildRequest(llm.InvokeRequest{
		Messages: []llm.Message{{
			Role: llm.RoleAssistant,
			ProviderState: []llm.ProviderState{{
				Provider: responsesStateProvider,
				Kind:     responsesOutputItemStateKind,
				Data:     json.RawMessage(`{"type":"reasoning"}`),
			}},
		}},
		Responses: &llm.ResponsesOptions{UseResponseItems: &legacyMessages},
	})
	if err == nil || !strings.Contains(err.Error(), "requires response item input mode") {
		t.Fatalf("legacy input opaque state error = %v", err)
	}
}

func TestResponsesPreviousResponseIDAndConversationAreMutuallyExclusive(t *testing.T) {
	client := &ResponsesClient{ModelName: "test-model"}
	built, err := client.buildRequest(llm.InvokeRequest{
		Messages:  []llm.Message{llm.NewUserMessage("continue")},
		Responses: &llm.ResponsesOptions{PreviousResponseID: "resp_previous"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if built.PreviousResponseID != "resp_previous" {
		t.Fatalf("previous_response_id = %q", built.PreviousResponseID)
	}
	_, err = client.buildRequest(llm.InvokeRequest{
		Messages: []llm.Message{llm.NewUserMessage("continue")},
		Responses: &llm.ResponsesOptions{
			PreviousResponseID: "resp_previous",
			ConversationID:     "conv_1",
		},
	})
	if err == nil || !strings.Contains(err.Error(), "cannot both be set") {
		t.Fatalf("conflicting stateful options error = %v", err)
	}
	_, err = client.buildRequest(llm.InvokeRequest{
		Messages: []llm.Message{{
			Role: llm.RoleAssistant,
			ProviderState: []llm.ProviderState{{
				Provider: responsesStateProvider,
				Kind:     responsesOutputItemStateKind,
				Data:     json.RawMessage(`{"type":"reasoning"}`),
			}},
		}},
		Responses: &llm.ResponsesOptions{PreviousResponseID: "resp_previous"},
	})
	if err == nil || !strings.Contains(err.Error(), "manually replayed provider state") {
		t.Fatalf("mixed stateful/manual continuation error = %v", err)
	}
	_, err = client.buildRequest(llm.InvokeRequest{
		Messages: []llm.Message{{
			Role: llm.RoleAssistant,
			ProviderState: []llm.ProviderState{{
				Provider: responsesStateProvider,
				Kind:     responsesOutputItemStateKind,
				Data:     json.RawMessage(`{"type":"reasoning"}`),
			}},
		}},
		Responses: &llm.ResponsesOptions{ConversationID: "conv_1"},
	})
	if err == nil || !strings.Contains(err.Error(), "manually replayed provider state") {
		t.Fatalf("mixed conversation/manual continuation error = %v", err)
	}
}

func TestResponsesStreamStateValidation(t *testing.T) {
	t.Run("semantic terminal match", func(t *testing.T) {
		server := openAIStreamFixture(t,
			`{"type":"response.output_item.done","output_index":0,"item":{"type":"reasoning","id":"rs_same","summary":[],"encrypted_content":"ciphertext"}}`,
			`{"type":"response.completed","response":{"id":"resp_same","status":"completed","output":[{"encrypted_content":"ciphertext","summary":[],"id":"rs_same","type":"reasoning"}]}}`,
		)
		defer server.Close()
		client := &ResponsesClient{BaseURL: server.URL, ModelName: "test-model", MaxRetries: 1}
		stream, err := client.InvokeStream(context.Background(), llm.InvokeRequest{Messages: []llm.Message{llm.NewUserMessage("hello")}})
		if err != nil {
			t.Fatal(err)
		}
		sawState := false
		sawDone := false
		for _, event := range collectOpenAIStream(stream) {
			switch typed := event.(type) {
			case llm.StreamProviderStateEvent:
				sawState = len(typed.State) == 1
			case llm.StreamDoneEvent:
				sawDone = true
			case llm.StreamErrorEvent:
				t.Fatalf("semantically identical state failed: %v", typed.AsError())
			}
		}
		if !sawState || !sawDone {
			t.Fatalf("state/done = %t/%t", sawState, sawDone)
		}
	})

	t.Run("sparse indexes before done", func(t *testing.T) {
		server := openAIStreamFixture(t,
			`{"type":"response.output_item.done","output_index":1,"item":{"id":"rs_sparse","type":"reasoning","summary":[]}}`,
			`[DONE]`,
		)
		defer server.Close()
		client := &ResponsesClient{BaseURL: server.URL, ModelName: "test-model", MaxRetries: 1}
		stream, err := client.InvokeStream(context.Background(), llm.InvokeRequest{Messages: []llm.Message{llm.NewUserMessage("hello")}})
		if err != nil {
			t.Fatal(err)
		}
		var streamErr error
		for _, event := range collectOpenAIStream(stream) {
			switch typed := event.(type) {
			case llm.StreamDoneEvent:
				t.Fatal("sparse opaque indexes emitted success")
			case llm.StreamErrorEvent:
				streamErr = typed.AsError()
			}
		}
		if streamErr == nil || !strings.Contains(streamErr.Error(), "not contiguous") {
			t.Fatalf("sparse index error = %v", streamErr)
		}
	})

	t.Run("invalid output item envelope", func(t *testing.T) {
		tests := []struct {
			name    string
			payload string
			want    string
		}{
			{name: "missing index", payload: `{"item":{"type":"reasoning"}}`, want: "no output_index"},
			{name: "missing item", payload: `{"output_index":0}`, want: "no item"},
			{name: "negative index", payload: `{"output_index":-1,"item":{"type":"reasoning"}}`, want: "non-negative"},
			{name: "oversized index", payload: fmt.Sprintf(`{"output_index":%d,"item":{"type":"reasoning"}}`, maxResponsesOutputItems), want: "exceeds limit"},
		}
		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				_, _, _, err := responsesProviderStateFromRawStreamItem([]byte(tc.payload))
				if err == nil || !strings.Contains(err.Error(), tc.want) {
					t.Fatalf("invalid envelope error = %v, want %q", err, tc.want)
				}
			})
		}
	})

	t.Run("stream item count and index are bounded immediately", func(t *testing.T) {
		events := make([]string, 0, maxResponsesOutputItems+2)
		for index := 0; index <= maxResponsesOutputItems; index++ {
			events = append(events, fmt.Sprintf(
				`{"type":"response.output_item.done","output_index":%d,"item":{"id":"rs_%d","type":"reasoning"}}`,
				index,
				index,
			))
		}
		events = append(events, `[DONE]`)
		server := openAIStreamFixture(t, events...)
		defer server.Close()
		client := &ResponsesClient{BaseURL: server.URL, ModelName: "test-model", MaxRetries: 1}
		stream, err := client.InvokeStream(context.Background(), llm.InvokeRequest{Messages: []llm.Message{llm.NewUserMessage("hello")}})
		if err != nil {
			t.Fatal(err)
		}
		var streamErr error
		for _, event := range collectOpenAIStream(stream) {
			switch typed := event.(type) {
			case llm.StreamDoneEvent:
				t.Fatal("unbounded opaque stream emitted success")
			case llm.StreamErrorEvent:
				streamErr = typed.AsError()
			}
		}
		if streamErr == nil || !strings.Contains(streamErr.Error(), "exceeds limit") {
			t.Fatalf("stream item limit error = %v", streamErr)
		}
	})
}

func TestResponsesStreamFallsBackToCompletedOutputItemsBeforeDone(t *testing.T) {
	server := openAIStreamFixture(t,
		`{"type":"response.output_item.done","output_index":0,"item":{"id":"rs_done","type":"reasoning","encrypted_content":"ciphertext","summary":[]}}`,
		`[DONE]`,
	)
	defer server.Close()
	client := &ResponsesClient{BaseURL: server.URL, ModelName: "test-model", MaxRetries: 1}
	stream, err := client.InvokeStream(context.Background(), llm.InvokeRequest{Messages: []llm.Message{llm.NewUserMessage("hello")}})
	if err != nil {
		t.Fatal(err)
	}
	events := collectOpenAIStream(stream)
	sawDone := false
	var state []llm.ProviderState
	for _, event := range events {
		switch typed := event.(type) {
		case llm.StreamProviderStateEvent:
			state = append(state, typed.State...)
		case llm.StreamDoneEvent:
			sawDone = true
		case llm.StreamErrorEvent:
			t.Fatalf("unexpected stream error: %v", typed.AsError())
		}
	}
	if !sawDone || len(state) != 1 || !strings.Contains(string(state[0].Data), "ciphertext") {
		t.Fatalf("terminal state/done = %#v/%t", state, sawDone)
	}
}

func TestResponsesStreamRejectsTerminalOpaqueItemConflict(t *testing.T) {
	server := openAIStreamFixture(t,
		`{"type":"response.output_item.done","output_index":0,"item":{"id":"rs_conflict","type":"reasoning","encrypted_content":"first","summary":[]}}`,
		`{"type":"response.completed","response":{"id":"resp_conflict","status":"completed","output":[{"id":"rs_conflict","type":"reasoning","encrypted_content":"second","summary":[]}]}}`,
	)
	defer server.Close()
	client := &ResponsesClient{BaseURL: server.URL, ModelName: "test-model", MaxRetries: 1, ProviderLabel: "gateway-responses"}
	stream, err := client.InvokeStream(context.Background(), llm.InvokeRequest{Messages: []llm.Message{llm.NewUserMessage("hello")}})
	if err != nil {
		t.Fatal(err)
	}
	var providerErr *llm.ProviderError
	for _, event := range collectOpenAIStream(stream) {
		switch typed := event.(type) {
		case llm.StreamDoneEvent:
			t.Fatal("conflicting opaque state emitted success")
		case llm.StreamErrorEvent:
			if !errors.As(typed.AsError(), &providerErr) {
				t.Fatalf("error = %v, want ProviderError", typed.AsError())
			}
		}
	}
	if providerErr == nil || providerErr.Provider != "gateway-responses" || !strings.Contains(providerErr.Message, "conflicts") {
		t.Fatalf("provider error = %#v", providerErr)
	}
}
