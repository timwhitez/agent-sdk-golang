package agent

import (
	"bytes"
	"encoding/json"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

// ToolParallelism lets the native tool loop run calls the host declares
// Concurrent in bounded waves (see BlockParallelism): handlers of one wave
// run at once, while admission, results, history and events stay in model
// order through the single block owner.
//
// The SDK never infers concurrency from tool names. It only narrows the
// host's plan: a call is offered to Plan only when it resolved exactly to a
// registered tool with valid arguments and is an evidence-family call (the
// progress ledger treats any other successful tool as a possible mutation of
// what later reads observe, so those always run Exclusive). Calls on the
// same evidence target never share a wave, so evidence suppression sees the
// same history it would sequentially. Once a wave is admitted, a steering
// message or a stop after one call does not recall later calls of that wave:
// they settle with their real outcome.
type ToolParallelism struct {
	// MaxWorkers bounds concurrently running handlers (capped at
	// MaxBlockWorkers); values below 2 disable waves.
	MaxWorkers int
	// Plan decides one call. It must be pure and read only its input.
	Plan func(ToolCallPlanInput) BlockCallPlan
}

// ToolCallPlanInput is what Plan may read about one call: its ordinal, the
// registered tool name it resolved to and its normalized arguments (an owned
// copy of the arguments Admit executes).
type ToolCallPlanInput struct {
	Ordinal   int
	Tool      string
	Arguments json.RawMessage
}

func cloneToolParallelism(p *ToolParallelism) *ToolParallelism {
	if p == nil {
		return nil
	}
	owned := *p
	return &owned
}

// nativePreparedCall is one call's resolution and preparation, computed once
// and used by both planning and Admit.
type nativePreparedCall struct {
	tool            tools.Tool
	resolvedName    string
	found           bool
	normalizedAlias bool
	execArgs        string
	prepared        tools.PreparedCall
	norm            tools.ToolArgsNormalization
}

func prepareNativeCall(tc llm.ToolCall, exact, normalized map[string]tools.Tool) *nativePreparedCall {
	p := &nativePreparedCall{execArgs: tc.Function.Arguments}
	p.tool, p.resolvedName, p.found, p.normalizedAlias = resolveToolByName(tc.Function.Name, exact, normalized)
	if !p.found {
		p.resolvedName = "invalid"
		p.execArgs = wrapInvalidToolArgs(tc.Function.Name, tc.Function.Arguments)
		if inv, ok := exact["invalid"]; ok {
			p.tool = inv
		} else {
			p.tool = autoInvalidTool()
		}
	}
	p.prepared, p.norm = p.tool.PrepareCall(p.execArgs)
	return p
}

// planNativeCall narrows the host's plan for one prepared call.
func (a *Agent) planNativeCall(idx int, p *nativePreparedCall) BlockCallPlan {
	if a.toolParallelism == nil || a.toolParallelism.Plan == nil || !p.found || p.normalizedAlias || p.norm.Err != nil {
		return BlockCallPlan{}
	}
	req, evidence := newEvidenceRequest(p.resolvedName, p.norm.Normalized, p.execArgs, a.deps)
	if !evidence {
		return BlockCallPlan{}
	}
	plan := a.toolParallelism.Plan(ToolCallPlanInput{Ordinal: idx, Tool: p.resolvedName, Arguments: bytes.Clone(p.norm.Normalized)})
	if !plan.Concurrent {
		return BlockCallPlan{}
	}
	resources := append(append([]string(nil), plan.Resources...), "sdk.evidence:"+req.family+"|"+req.target)
	return BlockCallPlan{Concurrent: true, Resources: resources}
}
