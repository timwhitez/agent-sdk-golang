package openai

import (
	"fmt"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

func TestR3RefusalFinalValidationPreservesSiblingTextAndPublicationBoundary(t *testing.T) {
	const textIdentity = `"item_id":"msg_mixed","output_index":0,"content_index":0`
	const refusalIdentity = `"item_id":"msg_mixed","output_index":0,"content_index":1`
	for _, stage := range []string{"whitespace_valid", "empty_part", "empty_item", "empty_terminal"} {
		t.Run(stage, func(t *testing.T) {
			refusal := ""
			if stage == "whitespace_valid" {
				refusal = " \n\t"
			}
			item := fmt.Sprintf(`{"id":"msg_mixed","type":"message","role":"assistant","content":[{"type":"output_text","text":"context: "},{"type":"refusal","refusal":%q}]}`, refusal)
			events := []string{
				`{"type":"response.content_part.added",` + textIdentity + `,"part":{"type":"output_text","text":""}}`,
				`{"type":"response.content_part.added",` + refusalIdentity + `,"part":{"type":"refusal","refusal":""}}`,
				`{"type":"response.output_text.delta",` + textIdentity + `,"delta":"context: "}`,
				`{"type":"response.refusal.delta",` + refusalIdentity + `,"delta":" "}`,
			}
			switch stage {
			case "whitespace_valid", "empty_part":
				events = append(events, fmt.Sprintf(`{"type":"response.content_part.done",%s,"part":{"type":"refusal","refusal":%q}}`, refusalIdentity, refusal))
			case "empty_item":
				events = append(events, `{"type":"response.output_item.done","output_index":0,"item":`+item+`}`)
			}
			events = append(events, `{"type":"response.completed","response":{"status":"completed","output":[`+item+`]}}`)
			server := openAIStreamFixture(t, events...)
			defer server.Close()
			client := &ResponsesClient{BaseURL: server.URL, ModelName: "fixture", MaxRetries: 1}
			stream, err := client.InvokeStream(t.Context(), llm.InvokeRequest{Messages: []llm.Message{llm.NewUserMessage("inspect")}})
			if err != nil {
				t.Fatal(err)
			}
			text, done, failures, states := "", 0, 0, 0
			for _, event := range collectOpenAIStream(stream) {
				switch event := event.(type) {
				case llm.StreamTextDeltaEvent:
					text += event.Delta
				case llm.StreamDoneEvent:
					done++
				case llm.StreamErrorEvent:
					failures++
				case llm.StreamProviderStateEvent:
					states += len(event.State)
				}
			}
			if stage == "whitespace_valid" {
				if text != "context:  \n\t" || done != 1 || failures != 0 || states != 1 {
					t.Fatalf("exact sibling output or success publication changed: text=%q done/errors/states=%d/%d/%d", text, done, failures, states)
				}
			} else if text != "context:  " || done != 0 || failures != 1 || states != 0 {
				t.Fatalf("invalid refusal published a terminal or opaque state: text=%q done/errors/states=%d/%d/%d", text, done, failures, states)
			}
		})
	}
}
