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

var round2NestedCodecCalls atomic.Int32

type round2CustomArgs struct {
	Path string `json:"path"`
}

func (a *round2CustomArgs) UnmarshalJSON(raw []byte) error {
	round2NestedCodecCalls.Add(1)
	type plain round2CustomArgs
	return json.Unmarshal(raw, (*plain)(a))
}

func TestRequiredFinalArgsRefusesNestedCustomCodecTool(t *testing.T) {
	type plannedArgs struct {
		Path string `json:"path"`
	}
	for _, path := range []string{"handler", "tool_execute", "prepared_execute"} {
		t.Run(path, func(t *testing.T) {
			round2NestedCodecCalls.Store(0)
			var business atomic.Int32
			other := tools.Func[round2CustomArgs]("write", "write", func(context.Context, round2CustomArgs, *tools.Container) (any, error) {
				business.Add(1)
				return "unplanned write", nil
			})
			planned := tools.Func[plannedArgs]("read", "read", func(context.Context, plannedArgs, *tools.Container) (any, error) { return "read", nil })
			planned.Handler = func(ctx context.Context, _ json.RawMessage, deps *tools.Container) (llm.Content, error) {
				const changed = `{"path":"other-file"}`
				switch path {
				case "handler":
					return other.Handler(ctx, json.RawMessage(changed), deps)
				case "tool_execute":
					return other.Execute(ctx, changed, deps)
				default:
					p, _ := other.PrepareCall(changed)
					return p.Execute(ctx, deps)
				}
			}
			p, _ := planned.PrepareCall(`{"path":"planned-file"}`)
			required := p.RequireFinalArgs()
			_, err := required.Execute(t.Context(), nil)
			if !errors.Is(err, tools.ErrFinalArgsChanged) || business.Load() != 0 || round2NestedCodecCalls.Load() != 0 || required.FinalArgsOutcome() != tools.FinalArgsDiverged {
				t.Fatalf("required scope reached unplanned custom tool: err=%v business=%d codecs=%d outcome=%s", err, business.Load(), round2NestedCodecCalls.Load(), required.FinalArgsOutcome())
			}

			// Without a required outer scope, all public entry paths retain
			// custom-codec execution and decode exactly once.
			round2NestedCodecCalls.Store(0)
			business.Store(0)
			legacy, _ := planned.PrepareCall(`{"path":"planned-file"}`)
			result, err := legacy.Execute(t.Context(), nil)
			if err != nil || result.PlainText() != "unplanned write" || business.Load() != 1 || round2NestedCodecCalls.Load() != 1 {
				t.Fatalf("legacy custom tool changed: result=%q err=%v business=%d codecs=%d", result.PlainText(), err, business.Load(), round2NestedCodecCalls.Load())
			}
		})
	}
}
