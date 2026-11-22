package openai

import (
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"testing"
)

func TestAuditTopLevelOverflowAgentRecovery(t *testing.T) {
	for _, tc := range []struct{ name, body string }{
		{"nested_control", `{"type":"error","error":{"code":"context_length_exceeded","message":"Input too long"}}`},
		{"published_shape", `{"type":"error","code":"context_length_exceeded","message":"Input too long","param":"input","sequence_number":1}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			run := runOverflowNegative(t, true, []negativeStep{sseStep("", "data: "+tc.body+"\n\n")}, false, t.Context())
			if run.main != 2 || run.summary != 1 || run.compactions != 1 || run.finals != 1 || run.errors != 0 {
				t.Fatalf("main=%d summary=%d compactions=%d final=%d errors=%d last_error=%q", run.main, run.summary, run.compactions, run.finals, run.errors, run.lastErrorMessage)
			}
		})
	}
}

func TestResponsesTopLevelErrorDoesNotInferOverflow(t *testing.T) {
	for _, code := range []any{nil, "invalid_request_error", "context_window_too_big", 400} {
		err := parseResponsesStreamEventError("gateway", map[string]any{"type": "error", "code": code, "message": "context length exceeded", "status": 400})
		if llm.IsContextOverflow(err) {
			t.Fatalf("code=%v inferred overflow: %v", code, err)
		}
	}
}
