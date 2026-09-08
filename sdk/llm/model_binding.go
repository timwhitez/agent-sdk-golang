package llm

import (
	"context"
	"errors"
	"reflect"
)

// FrameModelBinder optionally captures a callable model configuration for one
// logical frame, including its SDK retries. A successful binding must preserve
// invocation/streaming behavior and own mutable semantic configuration and
// capability inputs. Transport and diagnostic handles may remain shared; this
// is not a claim about a remote endpoint or mutable state inside those handles.
//
// Implementations must be bounded, respect context, and not invoke the model or
// mutate host state. Return bound=false when the complete configured chain
// cannot provide this guarantee; the driver then retains legacy dispatch.
type FrameModelBinder interface {
	BindFrameModel(context.Context) (model ChatModel, bound bool, err error)
}

// BindFrameModel calls the explicit capability once, without inspecting model
// names or unwrapping arbitrary models. The bool distinguishes known binding
// from legacy/unknown. Callers must not concurrently mutate input configuration.
// A different concrete wrapper type is treated as unknown: an embedded client's
// promoted method must not accidentally strip the outer wrapper's behavior.
func BindFrameModel(ctx context.Context, model ChatModel) (ChatModel, bool, error) {
	if err := ctx.Err(); err != nil {
		return nil, false, err
	}
	if nilFrameModel(model) {
		return nil, false, errors.New("nil frame model")
	}
	binder, ok := model.(FrameModelBinder)
	if !ok {
		return model, false, nil
	}
	bound, known, err := binder.BindFrameModel(ctx)
	if canceled := ctx.Err(); canceled != nil {
		return nil, false, canceled
	}
	if err != nil {
		return nil, false, err
	}
	if !known {
		return model, false, nil
	}
	if nilFrameModel(bound) {
		return nil, false, errors.New("nil frame model binding")
	}
	_, before := model.(StreamingChatModel)
	_, after := bound.(StreamingChatModel)
	if before != after {
		return nil, false, errors.New("frame binding changed streaming support")
	}
	if reflect.TypeOf(model) != reflect.TypeOf(bound) {
		return model, false, nil
	}
	return bound, true, nil
}

func nilFrameModel(model ChatModel) bool {
	v := reflect.ValueOf(model)
	if !v.IsValid() {
		return true
	}
	switch v.Kind() {
	case reflect.Pointer, reflect.Map, reflect.Slice, reflect.Func, reflect.Chan, reflect.Interface:
		return v.IsNil()
	}
	return false
}
