package openai

import (
	"io"
	"net/http"
	"reflect"
	"strings"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

func TestResponsesReasoningSummaryDeltasAndSnapshots(t *testing.T) {
	const first = `{"id":"rs_first","type":"reasoning","summary":[{"type":"summary_text","text":"hello"},{"type":"summary_text","text":" world"}]}`
	const second = `{"id":"rs_second","type":"reasoning","summary":[{"type":"summary_text","text":"next"}]}`
	body := strings.Join([]string{
		`data: {"type":"response.reasoning_summary_text.delta","item_id":"rs_first","output_index":0,"summary_index":0,"delta":"hel"}`,
		`data: {"type":"response.reasoning_summary_text.done","item_id":"rs_first","output_index":0,"summary_index":0,"text":"hello"}`,
		`data: {"type":"response.reasoning_summary_part.done","item_id":"rs_first","output_index":0,"summary_index":0,"part":{"type":"summary_text","text":"hello"}}`,
		`data: {"type":"response.output_item.done","output_index":0,"item":` + first + `}`,
		`data: {"type":"response.completed","response":{"id":"resp_summary","status":"completed","output":[` + first + `,` + second + `]}}`,
	}, "\n\n") + "\n\n"
	client := &ResponsesClient{ModelName: "fixture", MaxRetries: 1, HTTPClient: &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		return &http.Response{StatusCode: 200, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(body)), Request: r}, nil
	})}}
	events, err := client.InvokeStream(t.Context(), llm.InvokeRequest{Messages: []llm.Message{llm.NewUserMessage("reason")}})
	if err != nil {
		t.Fatal(err)
	}
	var deltas []string
	for event := range events {
		switch e := event.(type) {
		case llm.StreamThinkingDeltaEvent:
			deltas = append(deltas, e.Delta)
		case llm.StreamErrorEvent:
			t.Fatal(e.AsError())
		}
	}
	if want := []string{"hel", "lo", " world", "next"}; !reflect.DeepEqual(deltas, want) {
		t.Fatalf("deltas=%q want %q", deltas, want)
	}
}

func TestResponsesReasoningSummaryAlignsCompatibleItemIdentity(t *testing.T) {
	const item = `{"id":"rs_compat","type":"reasoning","summary":[{"type":"summary_text","text":"hello"}]}`
	for _, tc := range []struct {
		name   string
		events []string
		want   []string
	}{
		{"added_without_index", []string{`{"type":"response.output_item.added","item":` + item + `}`}, []string{"hello"}},
		{"delta_with_id_only", []string{`{"type":"response.reasoning_summary_text.delta","item_id":"rs_compat","summary_index":0,"delta":"hel"}`, `{"type":"response.output_item.done","output_index":0,"item":` + item + `}`}, []string{"hel", "lo"}},
		{"delta_with_index_only", []string{`{"type":"response.reasoning_summary_text.delta","output_index":0,"summary_index":0,"delta":"hel"}`, `{"type":"response.output_item.done","output_index":0,"item":` + item + `}`}, []string{"hel", "lo"}},
		{"done_with_index_only", []string{`{"type":"response.reasoning_summary_text.delta","item_id":"rs_compat","output_index":0,"summary_index":0,"delta":"hel"}`, `{"type":"response.reasoning_summary_text.done","output_index":0,"summary_index":0,"text":"hello"}`}, []string{"hel", "lo"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			events := append(append([]string(nil), tc.events...), `{"type":"response.completed","response":{"status":"completed","output":[`+item+`]}}`)
			server := openAIStreamFixture(t, events...)
			defer server.Close()
			c := &ResponsesClient{BaseURL: server.URL, ModelName: "fixture", MaxRetries: 1}
			stream, err := c.InvokeStream(t.Context(), llm.InvokeRequest{Messages: []llm.Message{llm.NewUserMessage("reason")}})
			if err != nil {
				t.Fatal(err)
			}
			var got []string
			for _, event := range collectOpenAIStream(stream) {
				switch e := event.(type) {
				case llm.StreamThinkingDeltaEvent:
					got = append(got, e.Delta)
				case llm.StreamErrorEvent:
					t.Fatal(e.AsError())
				}
			}
			if !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("deltas=%q want %q", got, tc.want)
			}
		})
	}
}
