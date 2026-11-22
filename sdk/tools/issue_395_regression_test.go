package tools_test

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

var issue395CodecCalls atomic.Int32

type issue395JSONDecoder struct {
	Value string `json:"value"`
}

func (v *issue395JSONDecoder) UnmarshalJSON(raw []byte) error {
	issue395CodecCalls.Add(1)
	type plain issue395JSONDecoder
	return json.Unmarshal(raw, (*plain)(v))
}

type issue395JSONEncoder struct {
	Value string `json:"value"`
}

func (v issue395JSONEncoder) MarshalJSON() ([]byte, error) {
	issue395CodecCalls.Add(1)
	return json.Marshal(v.Value)
}

type issue395TextDecoder string

func (v *issue395TextDecoder) UnmarshalText(raw []byte) error {
	issue395CodecCalls.Add(1)
	*v = issue395TextDecoder(raw)
	return nil
}

type issue395TextEncoder string

func (v issue395TextEncoder) MarshalText() ([]byte, error) {
	issue395CodecCalls.Add(1)
	return []byte(v), nil
}

type issue395ZeroValue string

func (v issue395ZeroValue) IsZero() bool {
	issue395CodecCalls.Add(1)
	return v == ""
}

func issue395CheckField[Args any](t *testing.T, raw string, wantFinal bool, wantExecutionCodecs int32) {
	t.Helper()
	issue395CodecCalls.Store(0)
	var business atomic.Int32
	base := tools.Func[Args]("fixture", "fixture", func(context.Context, Args, *tools.Container) (any, error) {
		business.Add(1)
		return "executed", nil
	})
	denied := errors.New("wrapper denied execution")
	wrapped := base
	wrapped.Handler = func(context.Context, json.RawMessage, *tools.Container) (llm.Content, error) {
		return llm.TextContent("denied"), denied
	}
	prepared, _ := wrapped.PrepareCall(raw)
	required := prepared.RequireFinalArgs()
	_, finalAvailable := required.FinalArgs()
	_, err := required.Execute(t.Context(), nil)
	if !errors.Is(err, denied) || business.Load() != 0 || issue395CodecCalls.Load() != 0 || finalAvailable != wantFinal {
		t.Fatalf("denied preparation: err=%v business=%d codecs=%d final_available=%t, want %t", err, business.Load(), issue395CodecCalls.Load(), finalAvailable, wantFinal)
	}
	if result, err := base.Execute(t.Context(), raw, nil); err != nil || result.PlainText() != "executed" || business.Load() != 1 || issue395CodecCalls.Load() != wantExecutionCodecs {
		t.Fatalf("ordinary execution changed: result=%q err=%v business=%d codecs=%d, want %d", result.PlainText(), err, business.Load(), issue395CodecCalls.Load(), wantExecutionCodecs)
	}
}

func TestNamedHyphenFieldsDoNotHideCustomCodecs(t *testing.T) {
	t.Run("json_decoder", func(t *testing.T) {
		issue395CheckField[struct {
			Field issue395JSONDecoder `json:"-,"`
		}](t, `{"-":{"value":"present"}}`, false, 1)
	})
	t.Run("json_decoder_omitempty", func(t *testing.T) {
		issue395CheckField[struct {
			Field issue395JSONDecoder `json:"-,omitempty"`
		}](t, `{"-":{"value":"present"}}`, false, 1)
	})
	t.Run("json_encoder", func(t *testing.T) {
		issue395CheckField[struct {
			Field issue395JSONEncoder `json:"-,"`
		}](t, `{"-":{"value":"present"}}`, false, 0)
	})
	t.Run("text_decoder", func(t *testing.T) {
		issue395CheckField[struct {
			Field issue395TextDecoder `json:"-,"`
		}](t, `{"-":"present"}`, false, 1)
	})
	t.Run("text_encoder", func(t *testing.T) {
		issue395CheckField[struct {
			Field issue395TextEncoder `json:"-,"`
		}](t, `{"-":"present"}`, false, 0)
	})
	t.Run("omitzero_hook", func(t *testing.T) {
		issue395CheckField[struct {
			Field issue395ZeroValue `json:"-,omitzero"`
		}](t, `{"-":"present"}`, false, 0)
	})
}

func TestNamedHyphenFieldPlainAndIgnoredControls(t *testing.T) {
	t.Run("plain_named_hyphen", func(t *testing.T) {
		issue395CheckField[struct {
			Field string `json:"-,"`
		}](t, `{"-":"present"}`, true, 0)
	})
	t.Run("ignored_json_decoder", func(t *testing.T) {
		issue395CheckField[struct {
			Field issue395JSONDecoder `json:"-"`
		}](t, `{}`, true, 0)
	})
	t.Run("ignored_json_encoder", func(t *testing.T) {
		issue395CheckField[struct {
			Field issue395JSONEncoder `json:"-"`
		}](t, `{}`, true, 0)
	})
	t.Run("ignored_text_decoder", func(t *testing.T) {
		issue395CheckField[struct {
			Field issue395TextDecoder `json:"-"`
		}](t, `{}`, true, 0)
	})
	t.Run("ignored_text_encoder", func(t *testing.T) {
		issue395CheckField[struct {
			Field issue395TextEncoder `json:"-"`
		}](t, `{}`, true, 0)
	})
}

func TestNamedHyphenCustomDecoderIsNotRetriedAfterUnknownField(t *testing.T) {
	type args struct {
		Field issue395JSONDecoder `json:"-,"`
	}
	issue395CodecCalls.Store(0)
	_, err := tools.DecodeTypedToolArgs[args](t.Context(), "fixture", tools.SchemaFor[args](), json.RawMessage(`{"-":{"value":"present"},"unknown":true}`))
	if err == nil || !strings.Contains(err.Error(), "unknown field") || issue395CodecCalls.Load() != 1 {
		t.Fatalf("custom decoder retried after unknown-field error: err=%v codecs=%d", err, issue395CodecCalls.Load())
	}
}
