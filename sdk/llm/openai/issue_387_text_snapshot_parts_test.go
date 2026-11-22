package openai

import (
	"fmt"
	"strings"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

func TestResponsesTextSnapshotsKeepEveryContentPart(t *testing.T) {
	for _, tc := range []struct {
		name, output string
		events       []string
	}{
		{
			name:   "multiple_parts_one_item",
			output: `[{"id":"msg_0","type":"message","content":[{"type":"output_text","text":"FIRST"},{"type":"output_text","text":"SECOND"}]}]`,
			events: []string{`{"type":"response.output_item.done","output_index":0,"item":{"id":"msg_0","type":"message","content":[{"type":"output_text","text":"FIRST"},{"type":"output_text","text":"SECOND"}]}}`},
		},
		{
			name:   "later_item_without_deltas",
			output: `[{"id":"msg_0","type":"message","content":[{"type":"output_text","text":"FIRST"}]},{"id":"msg_1","type":"message","content":[{"type":"output_text","text":"SECOND"}]}]`,
			events: []string{`{"type":"response.output_text.delta","item_id":"msg_0","output_index":0,"content_index":0,"delta":"FIRST"}`, `{"type":"response.output_item.done","output_index":1,"item":{"id":"msg_1","type":"message","content":[{"type":"output_text","text":"SECOND"}]}}`},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			final := `{"id":"resp_parts","status":"completed","output":` + tc.output + `}`
			buffered, err := parseResponses([]byte(final))
			if err != nil {
				t.Fatal(err)
			}
			if !strings.Contains(buffered.Content.PlainText(), "FIRST") || !strings.Contains(buffered.Content.PlainText(), "SECOND") {
				t.Fatal("invalid buffered control")
			}
			events := append(append([]string(nil), tc.events...), `{"type":"response.completed","response":`+final+`}`)
			server := openAIStreamFixture(t, events...)
			defer server.Close()
			client := &ResponsesClient{BaseURL: server.URL, ModelName: "fixture", MaxRetries: 1}
			stream, err := client.InvokeStream(t.Context(), llm.InvokeRequest{Messages: []llm.Message{llm.NewUserMessage("inspect")}})
			if err != nil {
				t.Fatal(err)
			}
			text, done := "", 0
			for _, event := range collectOpenAIStream(stream) {
				switch event := event.(type) {
				case llm.StreamTextDeltaEvent:
					text += event.Delta
				case llm.StreamErrorEvent:
					t.Fatal(event.AsError())
				case llm.StreamDoneEvent:
					done++
				}
			}
			if strings.Count(text, "FIRST") != 1 || strings.Count(text, "SECOND") != 1 || done != 1 {
				t.Fatalf("stream text=%q done=%d; buffered=%q", text, done, buffered.Content.PlainText())
			}
		})
	}
}

func TestResponsesTextSnapshotsReconcileEveryEventLayer(t *testing.T) {
	const message = `{"id":"msg_0","type":"message","content":[{"type":"output_text","text":"first "},{"type":"output_text","text":"second\n"}]}`
	const completed = `{"type":"response.completed","response":{"status":"completed","output":[` + message + `]}}`
	for _, tc := range []struct {
		name   string
		events []string
		want   string
	}{
		{
			"terminal_only", []string{completed}, "first second\n",
		},
		{
			"added_item_then_repeated_snapshots",
			[]string{
				`{"type":"response.output_item.added","output_index":0,"item":` + message + `}`,
				`{"type":"response.output_item.done","output_index":0,"item":` + message + `}`,
				completed,
			}, "first second\n",
		},
		{
			"part_and_text_done_suffixes",
			[]string{
				`{"type":"response.content_part.added","item_id":"msg_0","output_index":0,"content_index":0,"part":{"type":"output_text","text":""}}`,
				`{"type":"response.output_text.delta","item_id":"msg_0","output_index":0,"content_index":0,"delta":"fir"}`,
				`{"type":"response.output_text.done","item_id":"msg_0","output_index":0,"content_index":0,"text":"first "}`,
				`{"type":"response.content_part.done","item_id":"msg_0","output_index":0,"content_index":0,"part":{"type":"output_text","text":"first "}}`,
				`{"type":"response.content_part.done","item_id":"msg_0","output_index":0,"content_index":1,"part":{"type":"output_text","text":"second\n"}}`,
				completed,
			}, "first second\n",
		},
		{
			"whitespace_only_part",
			[]string{
				`{"type":"response.content_part.done","item_id":"msg_0","output_index":0,"content_index":0,"part":{"type":"output_text","text":" \n\t"}}`,
				`{"type":"response.completed","response":{"status":"completed","output":[{"id":"msg_0","type":"message","content":[{"type":"output_text","text":" \n\t"}]}]}}`,
			}, " \n\t",
		},
		{
			"mixed_text_and_refusal_parts",
			[]string{
				`{"type":"response.output_text.delta","item_id":"msg_0","output_index":0,"content_index":0,"delta":"The "}`,
				`{"type":"response.output_text.done","item_id":"msg_0","output_index":0,"content_index":0,"text":"The answer: "}`,
				`{"type":"response.refusal.delta","item_id":"msg_0","output_index":0,"content_index":1,"delta":"cannot "}`,
				`{"type":"response.completed","response":{"status":"completed","output":[{"id":"msg_0","type":"message","content":[{"type":"output_text","text":"The answer: "},{"type":"refusal","refusal":"cannot comply. "}]}]}}`,
			}, "The answer: cannot comply. ",
		},
		{
			"interleaved_items",
			[]string{
				`{"type":"response.output_text.delta","item_id":"msg_a","output_index":0,"content_index":0,"delta":"A1"}`,
				`{"type":"response.output_text.delta","item_id":"msg_b","output_index":1,"content_index":0,"delta":"B1"}`,
				`{"type":"response.output_text.delta","item_id":"msg_a","output_index":0,"content_index":0,"delta":"A2"}`,
				`{"type":"response.output_text.done","item_id":"msg_a","output_index":0,"content_index":0,"text":"A1A2"}`,
				`{"type":"response.output_text.delta","item_id":"msg_b","output_index":1,"content_index":0,"delta":"B2"}`,
				`{"type":"response.completed","response":{"status":"completed","output":[{"id":"msg_a","type":"message","content":[{"type":"output_text","text":"A1A2"}]},{"id":"msg_b","type":"message","content":[{"type":"output_text","text":"B1B2"}]}]}}`,
			}, "A1B1A2B2",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			assertResponsesTextParts(t, tc.events, tc.want, false)
		})
	}
}

func TestResponsesTextSnapshotsPreserveIdentityCompatibility(t *testing.T) {
	const item = `{"id":"msg_text","type":"message","content":[{"type":"output_text","text":"Hello world! \n"}]}`
	const reasoning = `{"id":"rs_before","type":"reasoning","summary":[]}`
	for _, tc := range []struct {
		name, identity string
		outputIndex    int
		output         string
	}{
		{"anonymous_after_reasoning", "", 1, reasoning + "," + item},
		{"id_only", `"item_id":"msg_text",`, 0, item},
		{"index_only", `"output_index":0,"content_index":0,`, 0, item},
		{"full_identity", `"item_id":"msg_text","output_index":0,"content_index":0,`, 0, item},
	} {
		t.Run(tc.name, func(t *testing.T) {
			assertResponsesTextParts(t, []string{
				`{"type":"response.output_text.delta",` + tc.identity + `"delta":"Hello "}`,
				`{"type":"response.output_text.done",` + tc.identity + `"text":"Hello world! \n"}`,
				fmt.Sprintf(`{"type":"response.output_item.done","output_index":%d,"item":%s}`, tc.outputIndex, item),
				`{"type":"response.completed","response":{"status":"completed","output":[` + tc.output + `]}}`,
			}, "Hello world! \n", false)
		})
	}
}

func TestResponsesTextSnapshotsTopLevelCompatibility(t *testing.T) {
	for _, tc := range []struct {
		name, final, want string
		events            []string
	}{
		{"without_deltas", "Only  \n", "Only  \n", nil},
		{"anonymous_suffix", "prefix ", "prefix ", []string{`{"type":"response.output_text.delta","delta":"pre"}`}},
		{"single_nonzero_part", "named suffix", "named suffix", []string{`{"type":"response.output_text.delta","item_id":"msg_1","output_index":1,"content_index":2,"delta":"named "}`}},
		{"multiple_parts_aggregate", "AB", "AB", []string{
			`{"type":"response.output_text.delta","item_id":"msg_a","output_index":0,"content_index":0,"delta":"A"}`,
			`{"type":"response.output_text.delta","item_id":"msg_b","output_index":1,"content_index":0,"delta":"B"}`,
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			events := append(append([]string(nil), tc.events...), fmt.Sprintf(`{"type":"response.completed","response":{"status":"completed","output_text":%q}}`, tc.final))
			assertResponsesTextParts(t, events, tc.want, false)
		})
	}
}

func TestResponsesTextSnapshotsRejectConflictsButIgnoreAbsentFields(t *testing.T) {
	for _, tc := range []struct {
		name, snapshot string
		wantError      bool
	}{
		{"different_done", `{"type":"response.output_text.done","text":"different"}`, true},
		{"empty_done", `{"type":"response.output_text.done","text":""}`, true},
		{"whitespace_done", `{"type":"response.output_text.done","text":" "}`, true},
		{"empty_part", `{"type":"response.content_part.done","part":{"type":"output_text","text":""}}`, true},
		{"different_item", `{"type":"response.output_item.done","output_index":0,"item":{"id":"msg_0","type":"message","content":[{"type":"output_text","text":"different"}]}}`, true},
		{"different_terminal", `{"type":"response.completed","response":{"status":"completed","output":[{"id":"msg_0","type":"message","content":[{"type":"output_text","text":"different"}]}]}}`, true},
		{"absent_done", `{"type":"response.output_text.done"}`, false},
		{"absent_part", `{"type":"response.content_part.done","part":{"type":"output_text"}}`, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			assertResponsesTextParts(t, []string{
				`{"type":"response.output_text.delta","delta":"original"}`,
				tc.snapshot,
				`{"type":"response.completed","response":{"status":"completed","output":[]}}`,
			}, "original", tc.wantError)
		})
	}
}

func assertResponsesTextParts(t *testing.T, events []string, wantText string, wantError bool) {
	t.Helper()
	server := openAIStreamFixture(t, events...)
	defer server.Close()
	client := &ResponsesClient{BaseURL: server.URL, ModelName: "fixture", MaxRetries: 1}
	stream, err := client.InvokeStream(t.Context(), llm.InvokeRequest{Messages: []llm.Message{llm.NewUserMessage("inspect")}})
	if err != nil {
		t.Fatal(err)
	}
	text, done, errors := "", 0, 0
	for _, event := range collectOpenAIStream(stream) {
		switch event := event.(type) {
		case llm.StreamTextDeltaEvent:
			text += event.Delta
		case llm.StreamDoneEvent:
			done++
		case llm.StreamErrorEvent:
			errors++
			if !wantError || !strings.Contains(event.AsError().Error(), "conflicts") {
				t.Fatal(event.AsError())
			}
		}
	}
	wantDone, wantErrors := 1, 0
	if wantError {
		wantDone, wantErrors = 0, 1
	}
	if text != wantText || done != wantDone || errors != wantErrors {
		t.Fatalf("text=%q done=%d errors=%d; want %q / %d / %d", text, done, errors, wantText, wantDone, wantErrors)
	}
}
