package tools_test

import (
	"context"
	"encoding/json"
	"errors"
	"sync/atomic"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

var issue394ZeroCalls atomic.Int32

type issue394ValueZero struct {
	Value string `json:"value"`
}

func (v issue394ValueZero) IsZero() bool {
	issue394ZeroCalls.Add(1)
	return v.Value == ""
}

type issue394PointerZero struct {
	Value string `json:"value"`
}

func (v *issue394PointerZero) IsZero() bool {
	issue394ZeroCalls.Add(1)
	return v.Value == ""
}

type issue394HiddenZero string

func (v issue394HiddenZero) IsZero() bool {
	issue394ZeroCalls.Add(1)
	return v == ""
}

func issue394CheckPreparation[Args any](t *testing.T, raw string, wantFinal bool) {
	t.Helper()
	issue394ZeroCalls.Store(0)
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
	if !errors.Is(err, denied) || business.Load() != 0 || issue394ZeroCalls.Load() != 0 || finalAvailable != wantFinal {
		t.Fatalf("denied preparation: err=%v business=%d IsZero=%d final_available=%t, want %t", err, business.Load(), issue394ZeroCalls.Load(), finalAvailable, wantFinal)
	}
	if !wantFinal && required.FinalArgsOutcome() != tools.FinalArgsUnavailable {
		t.Fatalf("custom IsZero outcome=%s", required.FinalArgsOutcome())
	}
	// Unplanned execution still decodes once and runs the complete Handler.
	// There is no reason to encode these arguments, so IsZero stays unused.
	if result, err := base.Execute(t.Context(), raw, nil); err != nil || result.PlainText() != "executed" || business.Load() != 1 || issue394ZeroCalls.Load() != 0 {
		t.Fatalf("ordinary execution changed: result=%q err=%v business=%d IsZero=%d", result.PlainText(), err, business.Load(), issue394ZeroCalls.Load())
	}
}

func TestFinalArgsOmitZeroHooksAreNotPrepared(t *testing.T) {
	t.Run("value_method", func(t *testing.T) {
		issue394CheckPreparation[struct {
			Value issue394ValueZero `json:"value,omitzero"`
		}](t, `{"value":{"value":"present"}}`, false)
	})
	t.Run("pointer_method_on_value", func(t *testing.T) {
		issue394CheckPreparation[struct {
			Value issue394PointerZero `json:"value,omitzero"`
		}](t, `{"value":{"value":"present"}}`, false)
	})
	t.Run("pointer_method_on_pointer", func(t *testing.T) {
		issue394CheckPreparation[struct {
			Value *issue394PointerZero `json:"value,omitzero"`
		}](t, `{"value":{"value":"present"}}`, false)
	})
	t.Run("combined_options", func(t *testing.T) {
		issue394CheckPreparation[struct {
			Value issue394ValueZero `json:"value,omitempty,omitzero"`
		}](t, `{"value":{"value":"present"}}`, false)
	})
	t.Run("nested", func(t *testing.T) {
		issue394CheckPreparation[struct {
			Items []struct {
				Value issue394ValueZero `json:"value,omitzero"`
			} `json:"items"`
		}](t, `{"items":[{"value":{"value":"present"}}]}`, false)
	})
	t.Run("same_type_previously_seen_without_option", func(t *testing.T) {
		issue394CheckPreparation[struct {
			Plain issue394ValueZero `json:"plain"`
			Value issue394ValueZero `json:"value,omitzero"`
		}](t, `{"plain":{"value":"plain"},"value":{"value":"present"}}`, false)
	})
	t.Run("plain_decoder_still_repairs_unknown_keys", func(t *testing.T) {
		issue394CheckPreparation[struct {
			Value issue394ValueZero `json:"value,omitzero"`
		}](t, `{"value":{"value":"present"},"unknown":true}`, false)
	})
	t.Run("named_embedding_uses_its_option", func(t *testing.T) {
		issue394CheckPreparation[struct {
			issue394ValueZero `json:"named,omitzero"`
		}](t, `{"named":{"value":"present"}}`, false)
	})
}

func TestFinalArgsOmitZeroPlainControlsRemainPrepared(t *testing.T) {
	t.Run("method_without_option", func(t *testing.T) {
		issue394CheckPreparation[struct {
			Value issue394ValueZero `json:"value"`
		}](t, `{"value":{"value":"present"}}`, true)
	})
	t.Run("omitempty_does_not_call_method", func(t *testing.T) {
		issue394CheckPreparation[struct {
			Value issue394PointerZero `json:"value,omitempty"`
		}](t, `{"value":{"value":"present"}}`, true)
	})
	t.Run("option_substring_is_not_omitzero", func(t *testing.T) {
		issue394CheckPreparation[struct {
			Value issue394ValueZero `json:"value,notomitzero"`
		}](t, `{"value":{"value":"present"}}`, true)
	})
	t.Run("omitzero_as_name_is_not_option", func(t *testing.T) {
		issue394CheckPreparation[struct {
			Value issue394ValueZero `json:"omitzero"`
		}](t, `{"omitzero":{"value":"present"}}`, true)
	})
	t.Run("plain_zero_field", func(t *testing.T) {
		issue394CheckPreparation[struct {
			Value string `json:"value,omitzero"`
		}](t, `{"value":""}`, true)
	})
	t.Run("ignored_field", func(t *testing.T) {
		issue394CheckPreparation[struct {
			Value issue394ValueZero `json:"-"`
		}](t, `{}`, true)
	})
	t.Run("unexported_field", func(t *testing.T) {
		issue394CheckPreparation[struct {
			value issue394ValueZero `json:"value,omitzero"`
		}](t, `{}`, true)
	})
	t.Run("flattened_embedding_ignores_its_options", func(t *testing.T) {
		issue394CheckPreparation[struct {
			issue394ValueZero `json:",omitzero"`
		}](t, `{"value":"present"}`, true)
	})
	t.Run("invalid_name_on_embedding_is_flattened", func(t *testing.T) {
		issue394CheckPreparation[struct {
			issue394ValueZero `json:"invalid\\name,omitzero"`
		}](t, `{"value":"present"}`, true)
	})
	t.Run("unexported_nonstruct_embedding_is_ignored", func(t *testing.T) {
		issue394CheckPreparation[struct {
			issue394HiddenZero `json:"hidden,omitzero"`
		}](t, `{}`, true)
	})
}
