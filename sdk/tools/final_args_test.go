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

type finalArgsFixture struct {
	Value  string            `json:"value"`
	Labels map[string]string `json:"labels,omitempty"`
}

func finalArgsTool(calls *int, seen *finalArgsFixture) Tool {
	return Func[finalArgsFixture]("final", "fixture", func(_ context.Context, args finalArgsFixture, _ *Container) (any, error) {
		*calls++
		*seen = args
		return "ok", nil
	})
}

func mapIdentity(m map[string]string) uintptr {
	return reflect.ValueOf(m).Pointer()
}

// Planning and execution use one decoded object: the business function
// receives the very value decoded at prepare time, not a second decode.
func TestFinalArgsConsumedByExecution(t *testing.T) {
	calls := 0
	var seen finalArgsFixture
	tool := finalArgsTool(&calls, &seen)
	prepared, _ := tool.PrepareCall(`{"value":"x","labels":{"k":"v"}}`)
	view, ok := prepared.FinalArgs()
	if !ok || string(view) != `{"value":"x","labels":{"k":"v"}}` {
		t.Fatalf("view=%s ok=%v", view, ok)
	}
	planned := prepared.final.args.(finalArgsFixture)
	if _, err := prepared.Execute(context.Background(), NewContainer()); err != nil {
		t.Fatal(err)
	}
	if calls != 1 || seen.Value != "x" || mapIdentity(seen.Labels) != mapIdentity(planned.Labels) {
		t.Fatalf("calls=%d seen=%+v: execution did not consume the prepared object", calls, seen)
	}
	if got := prepared.FinalArgsOutcome(); got != FinalArgsConsumed {
		t.Fatalf("outcome=%s", got)
	}
}

// The planning view is an owned copy: mutating it cannot change execution or
// later views.
func TestFinalArgsViewIsOwned(t *testing.T) {
	calls := 0
	var seen finalArgsFixture
	tool := finalArgsTool(&calls, &seen)
	prepared, _ := tool.PrepareCall(`{"value":"safe"}`)
	view, _ := prepared.FinalArgs()
	copy(view, []byte(`{"value":"evil"}`))
	again, _ := prepared.FinalArgs()
	if string(again) != `{"value":"safe"}` {
		t.Fatalf("view aliased internal state: %s", again)
	}
	if _, err := prepared.Execute(context.Background(), NewContainer()); err != nil || seen.Value != "safe" {
		t.Fatalf("err=%v seen=%+v", err, seen)
	}
}

// A wrapper that rewrites the forwarded bytes is never bypassed: the business
// function sees the rewritten arguments and the prepared object is diverged.
func TestFinalArgsWrapperRewriteDiverges(t *testing.T) {
	calls := 0
	var seen finalArgsFixture
	tool := finalArgsTool(&calls, &seen)
	inner := tool.Handler
	tool.Handler = func(ctx context.Context, raw json.RawMessage, deps *Container) (llm.Content, error) {
		return inner(ctx, json.RawMessage(`{"value":"rewritten"}`), deps)
	}
	prepared, _ := tool.PrepareCall(`{"value":"planned"}`)
	if _, err := prepared.Execute(context.Background(), NewContainer()); err != nil {
		t.Fatal(err)
	}
	if calls != 1 || seen.Value != "rewritten" {
		t.Fatalf("calls=%d seen=%+v: wrapper was bypassed", calls, seen)
	}
	if got := prepared.FinalArgsOutcome(); got != FinalArgsDiverged {
		t.Fatalf("outcome=%s", got)
	}
}

// A wrapper that refuses (for example a denied confirmation) never reaches
// the business function; the prepared object stays unused.
func TestFinalArgsWrapperDenialNeverRunsBusiness(t *testing.T) {
	calls := 0
	var seen finalArgsFixture
	tool := finalArgsTool(&calls, &seen)
	tool.Handler = func(context.Context, json.RawMessage, *Container) (llm.Content, error) {
		return llm.TextContent("denied"), errors.New("confirmation denied")
	}
	prepared, _ := tool.PrepareCall(`{"value":"x"}`)
	if _, err := prepared.Execute(context.Background(), NewContainer()); err == nil || calls != 0 {
		t.Fatalf("err=%v calls=%d", err, calls)
	}
	if got := prepared.FinalArgsOutcome(); got != FinalArgsUnused {
		t.Fatalf("outcome=%s", got)
	}
}

// Schema-key repair happens once at prepare time; its metadata is published
// only on the execution context, exactly as the legacy adapter did.
func TestFinalArgsTypedRepairPublishesAtExecution(t *testing.T) {
	calls := 0
	var seen finalArgsFixture
	tool := finalArgsTool(&calls, &seen)
	const raw = `{"Value":"x","extra":true}`
	prepared, _ := tool.PrepareCall(raw)
	view, ok := prepared.FinalArgs()
	if !ok || string(view) != `{"value":"x"}` {
		t.Fatalf("view=%s ok=%v", view, ok)
	}
	ctx := WithToolResultMetadata(context.Background())
	if meta := ToolResultMetadataSnapshot(ctx); len(meta) != 0 {
		t.Fatalf("planning published metadata: %v", meta)
	}
	if _, err := prepared.Execute(ctx, NewContainer()); err != nil || calls != 1 || seen.Value != "x" {
		t.Fatalf("err=%v calls=%d seen=%+v", err, calls, seen)
	}
	meta := ToolResultMetadataSnapshot(ctx)
	if kind, _ := meta["args_repair_kind"].(string); !strings.Contains(kind, "schema_key") || meta["args_raw"] != raw {
		t.Fatalf("metadata=%v", meta)
	}
	if prepared.FinalArgsOutcome() != FinalArgsConsumed {
		t.Fatalf("outcome=%s", prepared.FinalArgsOutcome())
	}

	// Legacy parity: the direct adapter path publishes the same metadata.
	legacyCtx := WithToolResultMetadata(context.Background())
	if _, err := tool.Execute(legacyCtx, raw, NewContainer()); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(ToolResultMetadataSnapshot(legacyCtx), meta) {
		t.Fatalf("metadata diverged from legacy: %v vs %v", ToolResultMetadataSnapshot(legacyCtx), meta)
	}
}

// Invalid canonical types never produce a final view and never run business.
func TestFinalArgsInvalidCanonicalIsUnavailable(t *testing.T) {
	calls := 0
	var seen finalArgsFixture
	tool := finalArgsTool(&calls, &seen)
	prepared, _ := tool.PrepareCall(`{"value":42}`)
	if _, ok := prepared.FinalArgs(); ok {
		t.Fatal("invalid arguments produced a final view")
	}
	if _, err := prepared.Execute(context.Background(), NewContainer()); err == nil || calls != 0 {
		t.Fatalf("err=%v calls=%d", err, calls)
	}
	if prepared.FinalArgsOutcome() != FinalArgsUnavailable {
		t.Fatalf("outcome=%s", prepared.FinalArgsOutcome())
	}
}

type finalArgsCountingDecoder struct{ Value string }

var finalArgsDecoderCalls int

func (d *finalArgsCountingDecoder) UnmarshalJSON(b []byte) error {
	finalArgsDecoderCalls++
	return json.Unmarshal(b, &d.Value)
}

// Custom decoders run user code, so they are never decoded at prepare time.
func TestFinalArgsCustomDecoderStaysLegacy(t *testing.T) {
	type customArgs struct {
		Value finalArgsCountingDecoder `json:"value"`
	}
	calls := 0
	tool := Func[customArgs]("custom", "fixture", func(context.Context, customArgs, *Container) (any, error) {
		calls++
		return "ok", nil
	})
	finalArgsDecoderCalls = 0
	prepared, _ := tool.PrepareCall(`{"value":"x"}`)
	if _, ok := prepared.FinalArgs(); ok || finalArgsDecoderCalls != 0 {
		t.Fatalf("custom decoder ran at prepare time: calls=%d", finalArgsDecoderCalls)
	}
	if _, err := prepared.Execute(context.Background(), NewContainer()); err != nil || calls != 1 || finalArgsDecoderCalls != 1 {
		t.Fatalf("err=%v business=%d decoder=%d", err, calls, finalArgsDecoderCalls)
	}
}

// Business errors and panics after consumption are not retried.
func TestFinalArgsBusinessFailureRunsOnce(t *testing.T) {
	calls := 0
	tool := Func[finalArgsFixture]("final", "fixture", func(context.Context, finalArgsFixture, *Container) (any, error) {
		calls++
		return nil, errors.New("unknown field in business logic")
	})
	prepared, _ := tool.PrepareCall(`{"value":"x"}`)
	if _, err := prepared.Execute(context.Background(), NewContainer()); err == nil || calls != 1 {
		t.Fatalf("err=%v calls=%d", err, calls)
	}
	panicking := Func[finalArgsFixture]("final", "fixture", func(context.Context, finalArgsFixture, *Container) (any, error) {
		calls++
		panic("boom")
	})
	prepared, _ = panicking.PrepareCall(`{"value":"x"}`)
	func() {
		defer func() { _ = recover() }()
		_, _ = prepared.Execute(context.Background(), NewContainer())
	}()
	if calls != 2 || prepared.FinalArgsOutcome() != FinalArgsConsumed {
		t.Fatalf("calls=%d outcome=%s", calls, prepared.FinalArgsOutcome())
	}
}

// Tools without a sealed decoder (hand-written Handlers) have no final view.
func TestFinalArgsUnavailableForHandWrittenHandler(t *testing.T) {
	tool := Tool{Name: "raw", Handler: func(context.Context, json.RawMessage, *Container) (llm.Content, error) {
		return llm.TextContent("ok"), nil
	}}
	prepared, _ := tool.PrepareCall(`{"a":1}`)
	if _, ok := prepared.FinalArgs(); ok || prepared.FinalArgsOutcome() != FinalArgsUnavailable {
		t.Fatal("hand-written handler claimed final arguments")
	}
}
