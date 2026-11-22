package openai

import (
	"fmt"
	"strings"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

func TestResponsesRefusalPrefixesAreScopedToContentParts(t *testing.T) {
	for _, sameItem := range []bool{false, true} {
		t.Run(fmt.Sprintf("same_item_%t", sameItem), func(t *testing.T) {
			texts := []string{"First refusal. ", "Second refusal."}
			var events, items []string
			for i, text := range texts {
				outputIndex, contentIndex := i, 0
				if sameItem {
					outputIndex, contentIndex = 0, i
				}
				identity := fmt.Sprintf(`"item_id":"msg_%d","output_index":%d,"content_index":%d`, outputIndex, outputIndex, contentIndex)
				events = append(events,
					fmt.Sprintf(`{"type":"response.refusal.delta",%s,"delta":%q}`, identity, text),
					fmt.Sprintf(`{"type":"response.refusal.done",%s,"refusal":%q}`, identity, text),
				)
			}
			if sameItem {
				items = []string{fmt.Sprintf(`{"id":"msg_0","type":"message","role":"assistant","content":[{"type":"refusal","refusal":%q},{"type":"refusal","refusal":%q}]}`, texts[0], texts[1])}
			} else {
				for i, text := range texts {
					items = append(items, fmt.Sprintf(`{"id":"msg_%d","type":"message","role":"assistant","content":[{"type":"refusal","refusal":%q}]}`, i, text))
				}
			}
			for i, item := range items {
				events = append(events, fmt.Sprintf(`{"type":"response.output_item.done","output_index":%d,"item":%s}`, i, item))
			}
			events = append(events, `{"type":"response.completed","response":{"id":"resp_refusal","status":"completed","output":[`+strings.Join(items, ",")+`]}}`)
			server := openAIStreamFixture(t, events...)
			defer server.Close()
			client := &ResponsesClient{BaseURL: server.URL, ModelName: "fixture", MaxRetries: 1}
			stream, err := client.InvokeStream(t.Context(), llm.InvokeRequest{Messages: []llm.Message{llm.NewUserMessage("inspect")}})
			if err != nil {
				t.Fatal(err)
			}
			var output strings.Builder
			var terminalError error
			done := 0
			for _, event := range collectOpenAIStream(stream) {
				switch event := event.(type) {
				case llm.StreamTextDeltaEvent:
					output.WriteString(event.Delta)
				case llm.StreamErrorEvent:
					terminalError = event.AsError()
				case llm.StreamDoneEvent:
					done++
				}
			}
			if terminalError != nil || done != 1 || output.String() != strings.Join(texts, "") {
				t.Fatalf("text=%q done=%d error=%v; want both refusal parts once and normal completion", output.String(), done, terminalError)
			}
		})
	}
}

func TestResponsesRefusalIdentityCompatibility(t *testing.T) {
	const item = `{"id":"msg_refusal","type":"message","role":"assistant","content":[{"type":"refusal","refusal":"Cannot comply. "}]}`
	const reasoning = `{"id":"rs_before","type":"reasoning","summary":[]}`
	for _, tc := range []struct {
		name, identity string
		outputIndex    int
		output         string
	}{
		{"anonymous_after_reasoning", "", 1, reasoning + "," + item},
		{"id_only", `"item_id":"msg_refusal",`, 0, item},
		{"index_only", `"output_index":0,"content_index":0,`, 0, item},
		{"full_identity", `"item_id":"msg_refusal","output_index":0,"content_index":0,`, 0, item},
	} {
		t.Run(tc.name, func(t *testing.T) {
			server := openAIStreamFixture(t,
				`{"type":"response.refusal.delta",`+tc.identity+`"delta":"Cannot "}`,
				`{"type":"response.refusal.done",`+tc.identity+`"refusal":"Cannot comply. "}`,
				fmt.Sprintf(`{"type":"response.output_item.done","output_index":%d,"item":%s}`, tc.outputIndex, item),
				`{"type":"response.completed","response":{"status":"completed","output":[`+tc.output+`]}}`,
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
				case llm.StreamErrorEvent:
					t.Fatal(event.AsError())
				case llm.StreamDoneEvent:
					done++
				}
			}
			if text != "Cannot comply. " || done != 1 {
				t.Fatalf("text=%q done=%d; want exact refusal once", text, done)
			}
		})
	}
}

func TestResponsesRefusalConflictWithinOnePartStillFails(t *testing.T) {
	server := openAIStreamFixture(t,
		`{"type":"response.refusal.delta","item_id":"msg_1","output_index":0,"content_index":0,"delta":"original"}`,
		`{"type":"response.refusal.done","item_id":"msg_1","output_index":0,"content_index":0,"refusal":"different"}`,
		`{"type":"response.completed","response":{"status":"completed","output":[]}}`,
	)
	defer server.Close()
	client := &ResponsesClient{BaseURL: server.URL, ModelName: "fixture", MaxRetries: 1}
	stream, err := client.InvokeStream(t.Context(), llm.InvokeRequest{Messages: []llm.Message{llm.NewUserMessage("inspect")}})
	if err != nil {
		t.Fatal(err)
	}
	errors, done := 0, 0
	for _, event := range collectOpenAIStream(stream) {
		switch event := event.(type) {
		case llm.StreamErrorEvent:
			if !strings.Contains(event.AsError().Error(), "conflicts") {
				t.Fatal(event.AsError())
			}
			errors++
		case llm.StreamDoneEvent:
			done++
		}
	}
	if errors != 1 || done != 0 {
		t.Fatalf("errors=%d done=%d; expected one conflict", errors, done)
	}
}
