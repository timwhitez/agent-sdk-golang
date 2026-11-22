package openai

import (
	"fmt"
	"testing"
)

func TestR3TextPartsUseAnnouncedItemIdentityAcrossAliases(t *testing.T) {
	for _, kind := range []string{"output_text", "refusal"} {
		for _, idFirst := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/id_first_%t", kind, idFirst), func(t *testing.T) {
				first, second := `"output_index":0`, `"item_id":"msg_alias"`
				if idFirst {
					first, second = second, first
				}
				field := "text"
				if kind == "refusal" {
					field = "refusal"
				}
				item := fmt.Sprintf(`{"id":"msg_alias","type":"message","role":"assistant","content":[{"type":%q,%q:"AB"}]}`, kind, field)
				assertResponsesTextParts(t, []string{
					`{"type":"response.output_item.added","output_index":0,"item":{"id":"msg_alias","type":"message","role":"assistant","content":[]}}`,
					fmt.Sprintf(`{"type":"response.%s.delta",%s,"content_index":0,"delta":"A"}`, kind, first),
					fmt.Sprintf(`{"type":"response.%s.delta",%s,"content_index":0,"delta":"B"}`, kind, second),
					fmt.Sprintf(`{"type":"response.%s.done","item_id":"msg_alias","output_index":0,"content_index":0,%q:"AB"}`, kind, field),
					`{"type":"response.output_item.done","output_index":0,"item":` + item + `}`,
					`{"type":"response.completed","response":{"status":"completed","output":[` + item + `]}}`,
				}, "AB", false)
			})
		}
	}
}

func TestR3TextAndRefusalAliasesShareExplicitMessageBindings(t *testing.T) {
	for _, header := range []bool{false, true} {
		t.Run(fmt.Sprintf("added_header_%t", header), func(t *testing.T) {
			var events []string
			if header {
				events = append(events, `{"type":"response.output_item.added","output_index":0,"item":{"id":"msg_mix","type":"message","content":[]}}`)
			}
			events = append(events,
				`{"type":"response.output_text.delta","item_id":"msg_mix","output_index":0,"content_index":0,"delta":"text:"}`,
				`{"type":"response.refusal.delta","item_id":"msg_mix","content_index":1,"delta":"A"}`,
				`{"type":"response.refusal.delta","output_index":0,"content_index":1,"delta":"B"}`,
				`{"type":"response.output_text.delta","output_index":0,"content_index":2,"delta":"C"}`,
				`{"type":"response.output_text.delta","item_id":"msg_mix","content_index":2,"delta":"D"}`,
				`{"type":"response.completed","response":{"status":"completed","output":[{"id":"msg_mix","type":"message","role":"assistant","content":[{"type":"output_text","text":"text:"},{"type":"refusal","refusal":"AB"},{"type":"output_text","text":"CD"}]}]}}`,
			)
			assertResponsesTextParts(t, events, "text:ABCD", false)
		})
	}
}

func TestR3ConflictingTextAliasesDoNotOverwriteEmittedPrefixes(t *testing.T) {
	for _, tc := range []struct {
		name   string
		events []string
		want   string
	}{
		{"changed_item_id", []string{
			`{"type":"response.output_item.added","output_index":0,"item":{"id":"msg_first","type":"message","content":[]}}`,
			`{"type":"response.output_text.delta","item_id":"msg_first","content_index":0,"delta":"A"}`,
			`{"type":"response.output_text.delta","item_id":"msg_other","output_index":0,"content_index":0,"delta":"B"}`,
		}, "A"},
		{"changed_output_index", []string{
			`{"type":"response.output_item.added","output_index":0,"item":{"id":"msg_first","type":"message","content":[]}}`,
			`{"type":"response.refusal.delta","item_id":"msg_first","content_index":0,"delta":"A"}`,
			`{"type":"response.refusal.done","item_id":"msg_first","output_index":1,"content_index":0,"refusal":"AB"}`,
		}, "A"},
		{"changed_header", []string{
			`{"type":"response.output_item.added","output_index":0,"item":{"id":"msg_first","type":"message","content":[]}}`,
			`{"type":"response.output_text.delta","output_index":0,"content_index":0,"delta":"A"}`,
			`{"type":"response.output_item.added","output_index":0,"item":{"id":"msg_other","type":"message","content":[]}}`,
		}, "A"},
		{"ambiguous_late_binding", []string{
			`{"type":"response.output_text.delta","item_id":"msg_first","content_index":0,"delta":"A"}`,
			`{"type":"response.output_text.delta","output_index":0,"content_index":0,"delta":"B"}`,
			`{"type":"response.output_text.done","item_id":"msg_first","output_index":0,"content_index":0,"text":"AB"}`,
		}, "AB"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			events := append(tc.events, `{"type":"response.completed","response":{"status":"completed","output":[]}}`)
			assertResponsesTextParts(t, events, tc.want, true)
		})
	}
}
