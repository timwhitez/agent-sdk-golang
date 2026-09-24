package tools

import (
	"context"
	"encoding/json"
	"errors"
	"sync/atomic"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

type reentryArgs struct {
	FilePath string `json:"file_path"`
}

// reentryTools returns a Func tool and a wrapper that re-enters it through
// one public path with the given arguments.
func reentryTools(inner *atomic.Int32, got *atomic.Value, path string, forwarded string) (Tool, Tool) {
	base := Func[reentryArgs]("read", "read", func(_ context.Context, a reentryArgs, _ *Container) (any, error) {
		inner.Add(1)
		got.Store(a.FilePath)
		return "read " + a.FilePath, nil
	})
	wrapped := base
	wrapped.Handler = func(ctx context.Context, raw json.RawMessage, deps *Container) (llm.Content, error) {
		switch path {
		case "handler":
			return base.Handler(ctx, json.RawMessage(forwarded), deps)
		case "tool-execute":
			return base.Execute(ctx, forwarded, deps)
		default: // "prepared-execute"
			p, _ := base.PrepareCall(forwarded)
			return p.Execute(ctx, deps)
		}
	}
	return base, wrapped
}

// N03R1: with required final arguments, none of the three public re-entry
// paths reaches the inner function with other arguments; without the
// requirement each keeps its legacy rewrite behavior.
func TestN03R1RequiredArgumentsSurviveReentry(t *testing.T) {
	for _, path := range []string{"handler", "tool-execute", "prepared-execute"} {
		t.Run(path, func(t *testing.T) {
			var inner atomic.Int32
			var got atomic.Value
			_, wrapped := reentryTools(&inner, &got, path, `{"file_path":"same.txt"}`)
			p, _ := wrapped.PrepareCall(`{"file_path":"a.txt"}`)
			if _, ok := p.FinalArgs(); !ok {
				t.Fatal("no final arguments")
			}
			_, err := p.RequireFinalArgs().Execute(context.Background(), nil)
			if !errors.Is(err, ErrFinalArgsChanged) || inner.Load() != 0 {
				t.Fatalf("required: err=%v inner=%d got=%v", err, inner.Load(), got.Load())
			}
			// Legacy: not required, the rewrite runs as before.
			p, _ = wrapped.PrepareCall(`{"file_path":"a.txt"}`)
			if _, err := p.Execute(context.Background(), nil); err != nil || inner.Load() != 1 || got.Load() != "same.txt" {
				t.Fatalf("legacy: err=%v inner=%d got=%v", err, inner.Load(), got.Load())
			}
		})
	}
}

// N03R3: re-entering with the planned arguments stays allowed on every path;
// a canceled call and a wrapper's own refusal keep their results.
func TestN03R3ReentryWithPlannedArgumentsIsAllowed(t *testing.T) {
	for _, path := range []string{"handler", "tool-execute", "prepared-execute"} {
		t.Run(path, func(t *testing.T) {
			var inner atomic.Int32
			var got atomic.Value
			_, wrapped := reentryTools(&inner, &got, path, `{"file_path":"a.txt"}`)
			p, _ := wrapped.PrepareCall(`{"file_path":"a.txt"}`)
			if _, err := p.RequireFinalArgs().Execute(context.Background(), nil); err != nil || inner.Load() != 1 || got.Load() != "a.txt" {
				t.Fatalf("err=%v inner=%d got=%v", err, inner.Load(), got.Load())
			}
		})
	}
	var inner atomic.Int32
	base := Func[reentryArgs]("read", "read", func(context.Context, reentryArgs, *Container) (any, error) {
		inner.Add(1)
		return "x", nil
	})
	denying := base
	denying.Handler = func(context.Context, json.RawMessage, *Container) (llm.Content, error) {
		return llm.TextContent("denied"), errors.New("permission denied")
	}
	p, _ := denying.PrepareCall(`{"file_path":"a.txt"}`)
	if _, err := p.RequireFinalArgs().Execute(context.Background(), nil); err == nil || errors.Is(err, ErrFinalArgsChanged) || inner.Load() != 0 {
		t.Fatalf("wrapper refusal changed: err=%v inner=%d", err, inner.Load())
	}
}

// A different Func tool reached inside a required call is an unplanned
// effect: refused when required, run as before otherwise.
func TestN03R1OtherToolInsideRequiredCallIsRefused(t *testing.T) {
	var helper atomic.Int32
	other := Func[reentryArgs]("write", "write", func(context.Context, reentryArgs, *Container) (any, error) {
		helper.Add(1)
		return "wrote", nil
	})
	base := Func[reentryArgs]("read", "read", func(context.Context, reentryArgs, *Container) (any, error) { return "read", nil })
	wrapped := base
	wrapped.Handler = func(ctx context.Context, raw json.RawMessage, deps *Container) (llm.Content, error) {
		return other.Execute(ctx, `{"file_path":"x.txt"}`, deps)
	}
	p, _ := wrapped.PrepareCall(`{"file_path":"a.txt"}`)
	if _, err := p.RequireFinalArgs().Execute(context.Background(), nil); !errors.Is(err, ErrFinalArgsChanged) || helper.Load() != 0 {
		t.Fatalf("required: err=%v helper=%d", err, helper.Load())
	}
	p, _ = wrapped.PrepareCall(`{"file_path":"a.txt"}`)
	if _, err := p.Execute(context.Background(), nil); err != nil || helper.Load() != 1 {
		t.Fatalf("legacy: err=%v helper=%d", err, helper.Load())
	}
}
