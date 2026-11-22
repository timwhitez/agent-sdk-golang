package tools_test

import (
	"context"
	"encoding/json"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
	"testing"
)

func TestRepairPreservesOpenObjectValues(t *testing.T) {
	type args struct {
		Payload any            `json:"payload"`
		Labels  map[string]any `json:"labels"`
	}
	const want = `{"payload":{"name":"review"},"labels":{"color":{"value":"blue"}}}`
	for _, tc := range []struct{ name, raw string }{
		{"clean", want},
		{"extra-sibling", `{"payload":{"name":"review"},"labels":{"color":{"value":"blue"}},"extra":true}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var got string
			t1 := tools.Func[args]("record", "Record payload", func(_ context.Context, a args, _ *tools.Container) (any, error) {
				b, err := json.Marshal(a)
				got = string(b)
				return "ok", err
			})
			p, _ := t1.PrepareCall(tc.raw)
			view, ok := p.FinalArgs()
			_, err := p.Execute(context.Background(), tools.NewContainer())
			if err != nil || !ok || got != want || string(view) != want {
				t.Errorf("final=%s, handler=%s, err=%v; want %s", view, got, err, want)
			}
		})
	}
}

func TestRepairPreservesOpenArraysAndNestedClosedObjects(t *testing.T) {
	type item struct {
		Value string `json:"value"`
	}
	type args struct {
		Open  any             `json:"open"`
		Typed map[string]item `json:"typed"`
	}
	const raw = `{"open":[{"free":{"name":"kept"}},[1,{"nested":true}]],"typed":{"entry":{"value":"kept","extra":"remove"}},"extra":true}`
	const want = `{"open":[{"free":{"name":"kept"}},[1,{"nested":true}]],"typed":{"entry":{"value":"kept"}}}`
	calls := 0
	tool := tools.Func[args]("record", "", func(_ context.Context, a args, _ *tools.Container) (any, error) {
		calls++
		b, err := json.Marshal(a)
		if string(b) != want {
			t.Fatalf("handler=%s want=%s", b, want)
		}
		return nil, err
	})
	p, _ := tool.PrepareCall(raw)
	final, ok := p.FinalArgs()
	if !ok || string(final) != want {
		t.Fatalf("final=%s ok=%t", final, ok)
	}
	if _, err := p.Execute(context.Background(), nil); err != nil || calls != 1 {
		t.Fatalf("err=%v calls=%d", err, calls)
	}
}
