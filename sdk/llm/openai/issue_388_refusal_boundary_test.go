package openai

import (
	"fmt"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

func TestRound2RefusalFinalSnapshotsValidateEmptyAndWhitespace(t *testing.T) {
	const identity = `"item_id":"msg_boundary","output_index":0,"content_index":0`
	item := func(text string) string {
		return fmt.Sprintf(`{"id":"msg_boundary","type":"message","role":"assistant","content":[{"type":"refusal","refusal":%q}]}`, text)
	}
	for _, stage := range []string{"refusal_done", "part_done", "item_done", "response_completed", "top_level"} {
		t.Run(stage, func(t *testing.T) {
			events := []string{`{"type":"response.refusal.delta",` + identity + `,"delta":"already emitted"}`}
			switch stage {
			case "refusal_done":
				events = append(events, `{"type":"response.refusal.done",`+identity+`,"refusal":""}`)
			case "item_done":
				events = append(events, `{"type":"response.output_item.done","output_index":0,"item":`+item("")+`}`)
			case "part_done":
				events = append(events, `{"type":"response.content_part.done",`+identity+`,"part":{"type":"refusal","refusal":""}}`)
			}
			if stage == "top_level" {
				events = append(events, `{"type":"response.completed","response":{"status":"completed","refusal":""}}`)
			} else {
				events = append(events, `{"type":"response.completed","response":{"status":"completed","output":[`+item("")+`]}}`)
			}
			server := openAIStreamFixture(t, events...)
			defer server.Close()
			client := &ResponsesClient{BaseURL: server.URL, ModelName: "fixture", MaxRetries: 1}
			stream, err := client.InvokeStream(t.Context(), llm.InvokeRequest{Messages: []llm.Message{llm.NewUserMessage("inspect")}})
			if err != nil {
				t.Fatal(err)
			}
			done, failures := 0, 0
			for _, event := range collectOpenAIStream(stream) {
				switch event.(type) {
				case llm.StreamDoneEvent:
					done++
				case llm.StreamErrorEvent:
					failures++
				}
			}
			if done != 0 || failures != 1 {
				t.Fatalf("empty final snapshot contradicted visible prefix: done=%d errors=%d", done, failures)
			}
		})
	}
	t.Run("whitespace_suffix", func(t *testing.T) {
		server := openAIStreamFixture(t,
			`{"type":"response.refusal.delta",`+identity+`,"delta":" "}`,
			`{"type":"response.refusal.done",`+identity+`,"refusal":"  "}`,
			`{"type":"response.completed","response":{"status":"completed","output":[`+item("  ")+`]}}`,
		)
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
			case llm.StreamDoneEvent:
				done++
			case llm.StreamErrorEvent:
				t.Fatal(event.AsError())
			}
		}
		if text != "  " || done != 1 {
			t.Fatalf("whitespace final: text=%q done=%d", text, done)
		}
	})
}

func TestRefusalSnapshotsDistinguishMissingFromEmpty(t *testing.T) {
	for _, tc := range []struct {
		name, prefix, middle, final, want string
	}{
		{"empty_without_prefix", "", `{"type":"response.refusal.done","refusal":""}`, `{"status":"completed","refusal":""}`, ""},
		{"missing_done_field", "retained", `{"type":"response.refusal.done"}`, `{"status":"completed"}`, "retained"},
		{"missing_part_field", "retained", `{"type":"response.content_part.done","part":{"type":"refusal"}}`, `{"status":"completed"}`, "retained"},
		{"exact_whitespace", "\n Cannot", `{"type":"response.refusal.done","refusal":"\n Cannot \t"}`, `{"status":"completed","refusal":"\n Cannot \t"}`, "\n Cannot \t"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			server := openAIStreamFixture(t,
				fmt.Sprintf(`{"type":"response.refusal.delta","delta":%q}`, tc.prefix),
				tc.middle,
				`{"type":"response.completed","response":`+tc.final+`}`,
			)
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
				case llm.StreamDoneEvent:
					done++
				case llm.StreamErrorEvent:
					t.Fatal(event.AsError())
				}
			}
			if text != tc.want || done != 1 {
				t.Fatalf("text=%q done=%d, want %q and one terminal", text, done, tc.want)
			}
		})
	}
}
