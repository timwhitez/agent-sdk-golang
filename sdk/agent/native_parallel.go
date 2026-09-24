package agent

import (
	"bytes"
	"encoding/json"
	"path/filepath"
	"strings"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

// ToolParallelism lets the native tool loop run calls the host declares
// Concurrent in bounded waves (see BlockParallelism): handlers of one wave
// run at once, while admission, results, history and events stay in model
// order through the single block owner.
//
// The SDK never infers concurrency from tool names; it only narrows the
// host's plan, and planning is pure (see planNativeCall). Plan is consulted
// once per call, only for an exact-resolved evidence-family call whose final
// arguments are known, and sees an owned copy of them. Calls naming one
// target never share a wave. A wave call must execute exactly the planned
// arguments. Once a wave is admitted, a steering message or a stop after one
// call does not recall later calls of that wave: they settle with their real
// outcome.
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

// planNativeCall narrows the host's plan for one prepared call. Planning is
// pure: it reads only the call's own preparation (SDK normalization and, for
// a Func tool whose argument type has no custom codec, the sealed decode), so
// it resolves no dependency, touches no file system and runs no user decoder,
// encoder, Handler or confirmation. The host sees, and the SDK resource is
// derived from, the final arguments the Func adapter will execute; a call
// without such a view is Exclusive. Inside a wave the call must consume
// exactly that view (see RequireFinalArgs).
func (a *Agent) planNativeCall(idx int, p *nativePreparedCall) BlockCallPlan {
	if a.toolParallelism == nil || a.toolParallelism.Plan == nil || !p.found || p.normalizedAlias || p.norm.Err != nil {
		return BlockCallPlan{}
	}
	if _, unproven := a.nativeUnproven.Load(p.resolvedName); unproven {
		return BlockCallPlan{}
	}
	family := evidenceFamily(p.resolvedName)
	if family == "" {
		return BlockCallPlan{}
	}
	view, ok := p.prepared.FinalArgs()
	if !ok {
		return BlockCallPlan{}
	}
	targets, ok := evidenceTargetsFromFinalView(family, view)
	if !ok {
		return BlockCallPlan{}
	}
	plan := a.toolParallelism.Plan(ToolCallPlanInput{Ordinal: idx, Tool: p.resolvedName, Arguments: bytes.Clone(view)})
	if !plan.Concurrent {
		return BlockCallPlan{}
	}
	resources := append([]string(nil), plan.Resources...)
	for _, target := range targets {
		resources = append(resources, "sdk.evidence:"+family+"|"+target)
	}
	return BlockCallPlan{Concurrent: true, Resources: resources}
}

// evidenceTargetsFromFinalView returns every lexically cleaned target the
// final arguments could name, without any file-system or dependency access.
// Every candidate key counts, so no field precedence is guessed; a view that
// is not a JSON object is not planned. Lexical keys do not prove that two
// different paths are different files: a symlinked or otherwise aliased path
// can share a wave, where the progress ledger (sampled at admission) may run
// a read it would have suppressed sequentially.
func evidenceTargetsFromFinalView(family string, view json.RawMessage) ([]string, bool) {
	var args map[string]any
	if err := json.Unmarshal(view, &args); err != nil || args == nil {
		return nil, false
	}
	keys := []string{"filePath", "file_path", "path"}
	seen := map[string]bool{}
	var targets []string
	for _, key := range keys {
		value := strings.TrimSpace(stringArg(args, key))
		if value == "" {
			continue
		}
		value = filepath.Clean(value)
		if !seen[value] {
			seen[value] = true
			targets = append(targets, value)
		}
	}
	if len(targets) == 0 {
		if family == "read" {
			return nil, false
		}
		targets = []string{"."}
	}
	return targets, true
}
