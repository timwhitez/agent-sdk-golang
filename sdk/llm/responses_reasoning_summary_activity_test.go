package llm_test

import (
	"encoding/json"
	"fmt"
	"github.com/timwhitez/agent-sdk-golang/sdk/agent"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm/openai"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"
)

func TestAuditResponsesReasoningSummaryActivity(t *testing.T) {
	for _, typ := range []string{"response.output_text.delta", "response.reasoning_summary_text.delta"} {
		t.Run(typ, func(t *testing.T) {
			var flushed atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var body struct{ Reasoning struct{ Summary string } }
				if err := json.NewDecoder(r.Body).Decode(&body); err != nil || body.Reasoning.Summary != "auto" {
					http.Error(w, "missing requested reasoning summary", 400)
					return
				}
				w.Header().Set("Content-Type", "text/event-stream")
				flush := w.(http.Flusher)
				emit := func(s string) { fmt.Fprint(w, "data: "+s+"\n\n"); flush.Flush() }
				emit(`{"type":"response.created","response":{"id":"resp_work","status":"in_progress","output":[]},"sequence_number":0}`)
				if typ == "response.reasoning_summary_text.delta" {
					emit(`{"type":"response.output_item.added","output_index":0,"item":{"id":"rs_work","type":"reasoning","summary":[]},"sequence_number":1}`)
					emit(`{"type":"response.reasoning_summary_part.added","item_id":"rs_work","output_index":0,"summary_index":0,"part":{"type":"summary_text","text":""},"sequence_number":2}`)
				}
				for i := 0; i < 10; i++ {
					select {
					case <-r.Context().Done():
						return
					case <-time.After(25 * time.Millisecond):
					}
					if typ == "response.reasoning_summary_text.delta" {
						emit(fmt.Sprintf(`{"type":%q,"item_id":"rs_work","output_index":0,"summary_index":0,"delta":"a","sequence_number":%d}`, typ, 3+i))
					} else {
						emit(fmt.Sprintf(`{"type":%q,"item_id":"msg_work","output_index":0,"content_index":0,"delta":"a","sequence_number":%d}`, typ, 3+i))
					}
					flushed.Add(1)
				}
				out := `[{"id":"msg_work","type":"message","role":"assistant","content":[{"type":"output_text","text":"aaaaaaaaaa"}]}]`
				if typ == "response.reasoning_summary_text.delta" {
					item := `{"id":"rs_work","type":"reasoning","summary":[{"type":"summary_text","text":"aaaaaaaaaa"}]}`
					emit(`{"type":"response.output_item.done","output_index":0,"item":` + item + `,"sequence_number":13}`)
					out = `[` + item + `,{"id":"msg_work","type":"message","role":"assistant","content":[{"type":"output_text","text":"ok"}]}]`
				}
				emit(`{"type":"response.completed","response":{"id":"resp_work","status":"completed","output":` + out + `},"sequence_number":14}`)
			}))
			defer server.Close()
			ag, err := agent.New(agent.Config{LLM: &openai.ResponsesClient{ModelName: "fixture", BaseURL: server.URL, MaxRetries: 1, Extra: map[string]any{"reasoning": map[string]any{"summary": "auto"}}}, StreamIdleTimeout: 120 * time.Millisecond, StreamIdleMaxRecoveries: 0, InvokeRetryMaxAttempts: 1, Warningf: func(string, ...any) {}})
			if err != nil {
				t.Fatal(err)
			}
			got, err := ag.Query(t.Context(), "Calculate the result")
			if err != nil {
				t.Fatalf("%d provider progress deltas flushed; query=%q; err=%v", flushed.Load(), got, err)
			}
			want := "ok"
			if typ == "response.output_text.delta" {
				want = "aaaaaaaaaa"
			}
			if got != want {
				t.Fatalf("answer=%q, want %q", got, want)
			}
		})
	}
}
