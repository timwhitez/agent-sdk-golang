package tools

import (
	"context"
	"encoding/json"
	"errors"

	"sync/atomic"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

var encoderEffects atomic.Int32

type effectEncoder struct{ V string }

func (e effectEncoder) MarshalJSON() ([]byte, error) {
	encoderEffects.Add(1)
	return json.Marshal(e.V)
}

type effectTextKey string

func (k effectTextKey) MarshalText() ([]byte, error) {
	encoderEffects.Add(1)
	return []byte(k), nil
}

type panicEncoder struct{ V string }

func (panicEncoder) MarshalJSON() ([]byte, error) { panic("encoder ran at prepare time") }

type effectRoot struct {
	Value string `json:"value"`
}

func (r effectRoot) MarshalJSON() ([]byte, error) {
	encoderEffects.Add(1)
	return json.Marshal(map[string]string{"value": r.Value})
}

type nestedEncoderArgs struct {
	Inner struct {
		E effectEncoder `json:"e"`
	} `json:"inner"`
}
type sliceEncoderArgs struct {
	Items []effectEncoder `json:"items"`
}
type pointerEncoderArgs struct {
	E *effectEncoder `json:"e"`
}
type mapKeyEncoderArgs struct {
	M map[effectTextKey]string `json:"m"`
}
type mapValueEncoderArgs struct {
	M map[string]effectEncoder `json:"m"`
}
type panicEncoderArgs struct {
	P panicEncoder `json:"p"`
}

// Types whose encoding runs user code are never prepared: no MarshalJSON or
// MarshalText runs before the (possibly wrapped) Handler chain.
func TestFinalArgsCustomEncodersAreNotPrepared(t *testing.T) {
	check := func(name string, tool Tool, raw string) {
		t.Run(name, func(t *testing.T) {
			encoderEffects.Store(0)
			prepared, _ := tool.PrepareCall(raw)
			if _, ok := prepared.FinalArgs(); ok {
				t.Fatal("type with a custom encoder produced a prepared view")
			}
			if got := encoderEffects.Load(); got != 0 {
				t.Fatalf("encoder ran %d time(s) at prepare time", got)
			}
			if prepared.FinalArgsOutcome() != FinalArgsUnavailable {
				t.Fatalf("outcome=%s", prepared.FinalArgsOutcome())
			}
		})
	}
	check("root", Func[effectRoot]("t", "t", func(context.Context, effectRoot, *Container) (any, error) { return "ok", nil }), `{"value":"x"}`)
	check("nested", Func[nestedEncoderArgs]("t", "t", func(context.Context, nestedEncoderArgs, *Container) (any, error) { return "ok", nil }), `{"inner":{"e":{"V":"x"}}}`)
	check("slice", Func[sliceEncoderArgs]("t", "t", func(context.Context, sliceEncoderArgs, *Container) (any, error) { return "ok", nil }), `{"items":[{"V":"x"}]}`)
	check("pointer", Func[pointerEncoderArgs]("t", "t", func(context.Context, pointerEncoderArgs, *Container) (any, error) { return "ok", nil }), `{"e":{"V":"x"}}`)
	check("map key", Func[mapKeyEncoderArgs]("t", "t", func(context.Context, mapKeyEncoderArgs, *Container) (any, error) { return "ok", nil }), `{"m":{"k":"v"}}`)
	check("map value", Func[mapValueEncoderArgs]("t", "t", func(context.Context, mapValueEncoderArgs, *Container) (any, error) { return "ok", nil }), `{"m":{"k":{"V":"x"}}}`)
	t.Run("panicking encoder", func(t *testing.T) {
		tool := Func[panicEncoderArgs]("t", "t", func(context.Context, panicEncoderArgs, *Container) (any, error) { return "ok", nil })
		prepared, _ := tool.PrepareCall(`{"p":{"V":"x"}}`)
		if _, ok := prepared.FinalArgs(); ok {
			t.Fatal("panicking encoder type was prepared")
		}
	})
}

// With a refusing outer wrapper, a custom encoder type never runs user code
// at all: nothing before the wrapper, and the wrapper stops the chain.
func TestFinalArgsCustomEncoderWithRefusingWrapperRunsNoUserCode(t *testing.T) {
	encoderEffects.Store(0)
	business := 0
	tool := Func[nestedEncoderArgs]("t", "t", func(context.Context, nestedEncoderArgs, *Container) (any, error) {
		business++
		return "ok", nil
	})
	tool.Handler = func(context.Context, json.RawMessage, *Container) (llm.Content, error) {
		return llm.TextContent("denied"), errors.New("confirmation denied")
	}
	prepared, _ := tool.PrepareCall(`{"inner":{"e":{"V":"x"}}}`)
	if _, err := prepared.Execute(context.Background(), NewContainer()); err == nil {
		t.Fatal("expected refusal")
	}
	if business != 0 || encoderEffects.Load() != 0 {
		t.Fatalf("business=%d encoder effects=%d", business, encoderEffects.Load())
	}
}

// #155's ambiguity rule is kept: an ambiguous repair never produces a
// prepared view and still fails before business execution.
func TestFinalArgsKeepsAmbiguityRejection(t *testing.T) {
	business := 0
	tool := Func[issue153ContentArgs]("t", "t", func(context.Context, issue153ContentArgs, *Container) (any, error) {
		business++
		return "ok", nil
	})
	prepared, _ := tool.PrepareCall(`{"text":"A","body":"B"}`)
	if _, ok := prepared.FinalArgs(); ok {
		t.Fatal("ambiguous arguments were prepared")
	}
	if _, err := prepared.Execute(context.Background(), NewContainer()); !errors.Is(err, errAmbiguousToolArguments) || business != 0 {
		t.Fatalf("err=%v business=%d", err, business)
	}
}
