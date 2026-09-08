package llm

import (
	"context"
	"errors"
	"testing"
)

type bindingModel struct {
	bind func(context.Context) (ChatModel, bool, error)
}

func (*bindingModel) Provider() string { panic("binding must not inspect names") }
func (*bindingModel) Model() string    { panic("binding must not inspect names") }
func (*bindingModel) Invoke(context.Context, InvokeRequest) (*Completion, error) {
	panic("binding must not invoke")
}
func (m *bindingModel) BindFrameModel(ctx context.Context) (ChatModel, bool, error) {
	return m.bind(ctx)
}

type bindingStreamModel struct{ *bindingModel }

func (*bindingStreamModel) InvokeStream(context.Context, InvokeRequest) (<-chan StreamEvent, error) {
	panic("must not stream")
}

type unboundModel struct{}

func (unboundModel) Provider() string { panic("must not inspect") }
func (unboundModel) Model() string    { panic("must not inspect") }
func (unboundModel) Invoke(context.Context, InvokeRequest) (*Completion, error) {
	panic("must not invoke")
}

func TestBindFrameModelCapabilityAndFailureBoundaries(t *testing.T) {
	legacy := unboundModel{}
	if model, known, err := BindFrameModel(context.Background(), legacy); err != nil || known || model != legacy {
		t.Fatal(model, known, err)
	}
	target := &bindingModel{}
	var nilTarget *bindingModel
	sentinel := errors.New("private binding failure")
	for _, tc := range []struct {
		name    string
		model   ChatModel
		known   bool
		err     error
		wantErr bool
	}{
		{"bound", target, true, nil, false},
		{"unknown_ignores_candidate", target, false, nil, false},
		{"error_even_if_unknown", target, false, sentinel, true},
		{"nil", nil, true, nil, true},
		{"typed_nil", nilTarget, true, nil, true},
		{"stream_upgrade", &bindingStreamModel{target}, true, nil, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			calls := 0
			source := &bindingModel{bind: func(context.Context) (ChatModel, bool, error) { calls++; return tc.model, tc.known, tc.err }}
			got, known, err := BindFrameModel(context.Background(), source)
			if calls != 1 || (err != nil) != tc.wantErr {
				t.Fatal(calls, known, err)
			}
			if err != nil {
				if got != nil || known {
					t.Fatal("partial binding returned")
				}
				return
			}
			want := ChatModel(source)
			if tc.known {
				want = target
			}
			if got != want || known != tc.known {
				t.Fatal("unknown/success ownership changed")
			}
		})
	}
	source := &bindingStreamModel{&bindingModel{bind: func(context.Context) (ChatModel, bool, error) { return target, true, nil }}}
	if _, _, err := BindFrameModel(context.Background(), source); err == nil {
		t.Fatal("stream downgrade accepted")
	}
	ctx, cancel := context.WithCancel(context.Background())
	source.bindingModel.bind = func(context.Context) (ChatModel, bool, error) { cancel(); return nil, true, sentinel }
	if _, _, err := BindFrameModel(ctx, source); !errors.Is(err, context.Canceled) {
		t.Fatal("cancellation lost precedence", err)
	}
	source.bindingModel.bind = func(context.Context) (ChatModel, bool, error) {
		t.Error("canceled entry called binder")
		return nil, false, nil
	}
	_, _, _ = BindFrameModel(ctx, source)
}
