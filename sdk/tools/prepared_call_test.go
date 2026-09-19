package tools

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

func TestPreparedCallOwnsArgumentsAndCapturedHandler(t *testing.T) {
	const raw = "```json\n{\"value\":\"safe\"}\n```"
	calls := 0
	tool := Tool{Name: "fixture", Handler: func(ctx context.Context, args json.RawMessage, _ *Container) (llm.Content, error) {
		calls++
		if ctx != nil || string(args) != `{"value":"safe"}` {
			t.Errorf("ctx=%v args=%s", ctx, args)
		}
		args[0] = '!'
		return llm.TextContent("original"), nil
	}}
	prepared, view := tool.PrepareCall(raw)
	copy(view.Normalized, []byte(`{"value":"evil"}`))
	view.Display["value"] = "evil"
	view.Meta["args_raw"] = "evil"
	tool.Handler = func(context.Context, json.RawMessage, *Container) (llm.Content, error) {
		return llm.TextContent("replacement"), nil
	}
	for i := 0; i < 2; i++ {
		out, err := prepared.Execute(nil, nil)
		if err != nil || out.PlainText() != "original" {
			t.Fatalf("prepared result=%s err=%v", out.PlainText(), err)
		}
	}
	out, err := tool.Execute(nil, raw, nil)
	if err != nil || out.PlainText() != "replacement" || calls != 2 {
		t.Fatalf("replacement result=%s err=%v original calls=%d", out.PlainText(), err, calls)
	}
	out, err = (Tool{Name: "missing"}).Execute(nil, "{", nil)
	if err == nil || !strings.Contains(err.Error(), "missing handler") || !out.IsEmpty() {
		t.Fatalf("missing handler result=%s err=%v", out.PlainText(), err)
	}
}
