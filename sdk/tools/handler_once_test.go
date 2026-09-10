package tools

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

func TestHandlerEffectMustNotReplayOnUnknownField(t *testing.T) {
	effects := 0
	tool := Tool{Name: "custom", Schema: map[string]any{"type": "object", "properties": map[string]any{"value": map[string]any{"type": "string"}}, "additionalProperties": false}, Handler: func(_ context.Context, raw json.RawMessage, _ *Container) (llm.Content, error) {
		effects++
		var args map[string]any
		_ = json.Unmarshal(raw, &args)
		if _, unknown := args["extra"]; unknown {
			return llm.TextContent("effect committed"), errors.New("downstream response: unknown field extra")
		}
		return llm.TextContent("second effect committed"), nil
	}}
	out, err := tool.Execute(context.Background(), `{"value":"ok","extra":true}`, NewContainer())
	if effects != 1 {
		t.Fatalf("handler side effects replayed: effects=%d error=%v output=%s", effects, err, out.PlainText())
	}
	if err == nil || out.PlainText() != "effect committed" {
		t.Fatal("first execution outcome was replaced", err)
	}
}

var decodeCalls int
var decodedRaw string

type flexibleNested struct {
	Value string `json:"value"`
}

func (v *flexibleNested) UnmarshalJSON(raw []byte) error {
	var input map[string]string
	if err := json.Unmarshal(raw, &input); err != nil {
		return err
	}
	v.Value = input["flattened_alias"]
	return nil
}
func TestTypedNestedCustomDecoderPreservesWiderInput(t *testing.T) {
	type args struct {
		Nested flexibleNested `json:"nested"`
	}
	tool := Func[args]("fixture", "fixture", func(_ context.Context, a args, _ *Container) (any, error) { return a.Nested.Value, nil })
	out, err := tool.Execute(context.Background(), `{"nested":{"flattened_alias":"preserved"}}`, NewContainer())
	if err != nil || out.PlainText() != "preserved" {
		t.Fatal("custom nested input stripped", out.PlainText(), err)
	}
}

type textDecoded string

func (v *textDecoded) UnmarshalText(raw []byte) error { *v = textDecoded(raw); return nil }

type failingTextDecoded string

func (*failingTextDecoded) UnmarshalText([]byte) error {
	decodeCalls++
	return errors.New("text decoder effect then unknown field")
}
func TestTypedTextDecoderErrorIsNotRetried(t *testing.T) {
	type args struct {
		Value failingTextDecoded `json:"value"`
	}
	decodeCalls = 0
	business := 0
	tool := Func[args]("fixture", "fixture", func(context.Context, args, *Container) (any, error) { business++; return "wrong", nil })
	_, err := tool.Execute(context.Background(), `{"value":"ok","extra":true}`, NewContainer())
	if decodeCalls != 1 || business != 0 || err == nil || err.Error() != "text decoder effect then unknown field" {
		t.Fatal("text decoder repeated", decodeCalls, business, err)
	}
}
func TestTypedPreparationSkipsCustomDecoderGraphs(t *testing.T) {
	type recursive struct {
		Next   *recursive
		Custom map[textDecoded][]flexibleNested
	}
	raw := json.RawMessage(`{ "unknown" : "retain" }`)
	schema := map[string]any{"type": "object", "properties": map[string]any{"value": map[string]any{"type": "string"}}}
	if !hasCustomArgumentDecoder(reflect.TypeOf(recursive{}), make(map[reflect.Type]bool)) {
		t.Fatal("nested decoder graph not detected")
	}
	if got, err := DecodeTypedToolArgs[map[textDecoded]string](context.Background(), "fixture", schema, raw); err != nil || got["unknown"] != "retain" {
		t.Fatal("text decoder graph normalized", got, err)
	}
}

func TestTypedValidCaseInsensitiveArgsAvoidRepair(t *testing.T) {
	type args struct {
		Value string `json:"value"`
	}
	ctx := WithToolResultMetadata(context.Background())
	tool := Func[args]("fixture", "fixture", func(_ context.Context, a args, _ *Container) (any, error) { return a.Value, nil })
	out, err := tool.Execute(ctx, `{ "VALUE" : "ok" }`, NewContainer())
	if err != nil || out.PlainText() != "ok" || argsRepaired(ToolResultMetadataSnapshot(ctx)) {
		t.Fatal("valid arguments changed", err, ToolResultMetadataSnapshot(ctx))
	}
}

func TestTypedRepairPreservesOriginalSpellingMetadata(t *testing.T) {
	type args struct {
		Value string `json:"value"`
	}
	tool := Func[args]("fixture", "fixture", func(_ context.Context, a args, _ *Container) (any, error) { return a.Value, nil })
	for _, raw := range []string{" \n {\"value\":\"ok\",\"extra\":true}\t ", " \n {\"value\":\"ok\"}\t "} {
		ctx := WithToolResultMetadata(context.Background())
		if _, err := tool.Execute(ctx, raw, NewContainer()); err != nil {
			t.Fatal(err)
		}
		meta := ToolResultMetadataSnapshot(ctx)
		if strings.Contains(raw, "extra") {
			if meta["args_raw"] != raw {
				t.Fatal("original spelling lost", meta)
			}
		} else if _, ok := meta["args_raw"]; ok {
			t.Fatal("untouched raw newly exposed", meta)
		}
	}
}

type onceDecodedArgs struct {
	Value string `json:"value"`
}

func (a *onceDecodedArgs) UnmarshalJSON(raw []byte) error {
	decodeCalls++
	decodedRaw = string(raw)
	return errors.New("decoder effect then unknown field")
}

func TestTypedDecoderRunsOnceBeforeBusiness(t *testing.T) {
	for _, raw := range []string{`{ "value" : "ok" }`, `{"value":"ok","extra":true}`} {
		decodeCalls, decodedRaw = 0, ""
		business := 0
		tool := Func[onceDecodedArgs]("custom", "fixture", func(context.Context, onceDecodedArgs, *Container) (any, error) { business++; return "wrong", nil })
		_, err := tool.Execute(context.Background(), raw, NewContainer())
		if decodeCalls != 1 || business != 0 || err == nil || err.Error() != "decoder effect then unknown field" {
			t.Fatal("decoder replayed or lost error", decodeCalls, business, err)
		}
		if raw[1] == ' ' && decodedRaw != raw {
			t.Fatal("valid raw bytes changed", decodedRaw)
		}
		if decodedRaw != raw {
			t.Fatal("custom decoder input changed", decodedRaw)
		}
	}
}
