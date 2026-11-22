package openai

import (
	"fmt"
	"strings"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

func TestResponsesOpaqueSnapshotsCompareExactNumbers(t *testing.T) {
	for _, tc := range []struct {
		name, first, final string
		wantError          bool
	}{
		{"identical_large_exponent", "1e400", "1e400", false},
		{"different_adjacent_integers", "9007199254740992", "9007199254740993", true},
		{"identical_large_integer", "9007199254740993", "9007199254740993", false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			item := func(number string) string {
				return fmt.Sprintf(`{"id":"prog_1","type":"program_output","status":"completed","result":"output","future_counter":%s}`, number)
			}
			server := openAIStreamFixture(t,
				`{"type":"response.output_item.done","output_index":0,"item":`+item(tc.first)+`}`,
				`{"type":"response.completed","response":{"id":"resp_numbers","status":"completed","output":[`+item(tc.final)+`]}}`,
			)
			defer server.Close()
			client := &ResponsesClient{BaseURL: server.URL, ModelName: "fixture", MaxRetries: 1}
			stream, err := client.InvokeStream(t.Context(), llm.InvokeRequest{Messages: []llm.Message{llm.NewUserMessage("inspect")}})
			if err != nil {
				t.Fatal(err)
			}
			var terminalError error
			done, states := 0, 0
			for _, event := range collectOpenAIStream(stream) {
				switch event := event.(type) {
				case llm.StreamErrorEvent:
					terminalError = event.AsError()
				case llm.StreamDoneEvent:
					done++
				case llm.StreamProviderStateEvent:
					states += len(event.State)
					if len(event.State) > 0 && !strings.Contains(string(event.State[0].Data), tc.final) {
						t.Fatalf("opaque counter was rewritten: %s", event.State[0].Data)
					}
				}
			}
			if tc.wantError {
				if terminalError == nil || done != 0 || states != 0 {
					t.Fatalf("changed numeric state: error=%v done=%d state=%d; want conflict without publication", terminalError, done, states)
				}
			} else if terminalError != nil || done != 1 || states != 1 {
				t.Fatalf("identical numeric state: error=%v done=%d state=%d; want success", terminalError, done, states)
			}
		})
	}
}

func TestResponsesOpaqueEqualityStillIgnoresFormatting(t *testing.T) {
	left := []byte(`{"type":"program_output","counter":9007199254740993,"nested":{"value":1e400}}`)
	right := []byte(`{ "nested": { "value": 1e400 }, "counter": 9007199254740993, "type": "program_output" }`)
	if !responsesJSONEqual(left, right) {
		t.Fatal("key order or insignificant whitespace changed equality")
	}
	for _, invalid := range []string{`{"counter":1} {"counter":1}`, `{"counter":`, `false trailing`} {
		if responsesJSONEqual([]byte(invalid), []byte(invalid)) {
			t.Fatalf("invalid JSON was accepted: %q", invalid)
		}
	}
}

func TestResponsesOpaqueEqualityNormalizesNumbersWithoutFloat64(t *testing.T) {
	for _, pair := range [][2]string{
		{"1", "1.0"}, {"1", "1e0"}, {"-0", "0"},
		{"0.0100", "1e-2"}, {"1000", "1e3"}, {"-1.25", "-125e-2"},
		{"1e400", "10e399"}, {"1e1000000", "10e999999"},
		{"1e-100000000000000000000", "10e-100000000000000000001"},
	} {
		t.Run(pair[0]+"_"+pair[1], func(t *testing.T) {
			left := []byte(`{"nested":[` + pair[0] + `]}`)
			right := []byte(`{"nested":[` + pair[1] + `]}`)
			if !responsesJSONEqual(left, right) {
				t.Fatalf("equivalent numbers compared unequal: %s / %s", pair[0], pair[1])
			}
		})
	}
	if responsesJSONEqual([]byte(`1`), []byte(`"1"`)) {
		t.Fatal("number and string were conflated")
	}
}
