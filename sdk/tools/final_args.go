package tools

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"sync/atomic"
)

// typedArgsBinding is the sealed decoder half of a Func tool. Func creates one
// per tool and its Handler closure holds the same pointer, so a prepared
// object can only be consumed by the adapter that belongs to its decoder.
// Bindings exist only for plain argument types: decoding and encoding them
// runs no user code, so preparing final arguments has no side effects.
type typedArgsBinding struct {
	decode func(json.RawMessage) (any, bool, error)
}

// finalArgsState records what happened to a prepared object during execution.
type finalArgsState struct {
	consumed atomic.Bool
	diverged atomic.Bool
}

// preparedTypedArgs is the final typed argument object accepted at prepare
// time. It is owned by one PreparedCall and never exposed to consumers.
type preparedTypedArgs struct {
	binding  *typedArgsBinding
	raw      json.RawMessage
	args     any
	repaired bool
	view     json.RawMessage
	state    *finalArgsState
	// required: the adapter refuses bytes other than the prepared ones
	// instead of decoding them anew (see PreparedCall.RequireFinalArgs).
	required bool
}

// ErrFinalArgsChanged refuses a Func adapter reached inside a call whose
// prepared final arguments were required, when it would run on other bytes
// (a wrapper rewrote them, directly or by re-entering Tool.Execute or
// PreparedCall.Execute) or is another Func tool. The typed function was not
// run; whatever the wrapper itself did before is not undone or replayed.
var ErrFinalArgsChanged = errors.New("tool arguments changed after they were prepared for concurrent execution; the tool function was not run on them")

type preparedTypedArgsKey struct{}

// Final argument outcomes reported by PreparedCall.FinalArgsOutcome.
const (
	FinalArgsUnavailable = "unavailable" // no sealed binding, custom decoder or decode failure
	FinalArgsUnused      = "unused"      // prepared, but the adapter was never reached
	FinalArgsConsumed    = "consumed"    // the adapter executed the prepared object itself
	FinalArgsDiverged    = "diverged"    // a wrapper changed the bytes; the adapter decoded them anew
)

func newTypedArgsBinding[Args any](name string, schema map[string]any) *typedArgsBinding {
	// Preparing decodes and encodes the arguments; both must be free of user
	// code (custom JSON/Text decoders or encoders anywhere in the type).
	if hasCustomArgumentCodec(reflect.TypeOf((*Args)(nil)).Elem(), make(map[reflect.Type]bool), true) {
		return nil
	}
	return &typedArgsBinding{decode: func(raw json.RawMessage) (any, bool, error) {
		args, repaired, err := decodeTypedArgs[Args](name, schema, raw)
		return args, repaired, err
	}}
}

// prepareTypedArgs decodes final arguments once for a sealed binding.
func prepareTypedArgs(binding *typedArgsBinding, normalized json.RawMessage) *preparedTypedArgs {
	if binding == nil || normalized == nil {
		return nil
	}
	args, repaired, err := binding.decode(normalized)
	if err != nil {
		return nil
	}
	view, err := json.Marshal(args)
	if err != nil {
		return nil
	}
	return &preparedTypedArgs{binding: binding, raw: bytes.Clone(normalized), args: args, repaired: repaired, view: view, state: &finalArgsState{}}
}

// consumeTypedArgs lets the Func adapter execute the prepared object when it
// receives exactly the prepared bytes for its own binding. Any other input,
// such as bytes rewritten by a wrapper, is decoded by the legacy path and the
// prepared object is marked diverged. A prepared object is consumed once.
func consumeTypedArgs[Args any](ctx context.Context, binding *typedArgsBinding, name string, schema map[string]any, raw json.RawMessage) (Args, error) {
	if binding != nil && ctx != nil {
		prepared, _ := ctx.Value(preparedTypedArgsKey{}).(*preparedTypedArgs)
		if prepared != nil && prepared.required && prepared.binding != binding {
			// Another Func tool reached inside a call whose arguments are
			// required: its effect was not planned, so it is refused.
			prepared.state.diverged.Store(true)
			var zero Args
			return zero, ErrFinalArgsChanged
		}
		if prepared != nil && prepared.binding == binding {
			if prepared.required && !bytes.Equal(prepared.raw, raw) {
				prepared.state.diverged.Store(true)
				var zero Args
				return zero, ErrFinalArgsChanged
			}
			if bytes.Equal(prepared.raw, raw) && prepared.state.consumed.CompareAndSwap(false, true) {
				if args, ok := prepared.args.(Args); ok {
					if prepared.repaired {
						publishSchemaKeyRepair(ctx, raw)
					}
					return args, nil
				}
			} else if !bytes.Equal(prepared.raw, raw) {
				prepared.state.diverged.Store(true)
			}
		}
	}
	return DecodeTypedToolArgs[Args](ctx, name, schema, raw)
}
