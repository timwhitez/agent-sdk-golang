package llm

import (
	"context"
	"fmt"
)

// Bind creates an owned plan tied to this original immutable request view.
// Cloning a request/plan preserves the binding. Mutating a clone's request
// requires a fresh plan for that new request, not silently rebinding the old one.
func (view *CacheTargetView) Bind(directives []CacheDirective) (*CachePlan, error) {
	if view == nil || view.request == nil {
		return nil, cachePlanError("missing_view", -1)
	}
	plan := CloneCachePlan(&CachePlan{SchemaVersion: CachePlanSchemaVersion, Directives: directives})
	if err := view.Validate(*view.request, plan); err != nil {
		return nil, err
	}
	// The exported view value can be overwritten wholesale by its owner even
	// though its fields are private. Keep an unreachable copy of that value.
	binding := *view
	plan.view = &binding
	return plan, nil
}

// AdmitCachePlan is the shared before-network boundary for built-in clients.
// Nil/empty plans preserve the legacy path. Nonempty plans require an original
// view binding and an owned request clone; no view is rebuilt during admission.
// The returned request is isolated from caller mutation before asynchronous
// streaming or retry. Callers must exclusively own input graphs during this call.
//
// Current clients have no explicit mapper: required intent fails, best-effort
// intent is skipped with bounded diagnostics while legacy cache flags remain.
// An accepted directive fails as unmapped rather than silently reaching a client
// whose serializer cannot implement it. Future mappers must extend this boundary.
// Warning callbacks see fixed metadata only; buffered clients also return these
// diagnostics in Completion.Diagnostics. No new event or invocation path is used.
func AdmitCachePlan(ctx context.Context, request InvokeRequest, model ChatModel, warnf func(string, ...any)) (InvokeRequest, []Diagnostic, error) {
	if request.CachePlan == nil || len(request.CachePlan.Directives) == 0 {
		return request, nil, nil
	}
	if err := ctx.Err(); err != nil {
		return InvokeRequest{}, nil, err
	}
	owned, err := CloneInvokeRequest(request)
	if err != nil {
		return InvokeRequest{}, nil, cachePlanError("uncloneable_request", -1)
	}
	if owned.CachePlan.view == nil {
		return InvokeRequest{}, nil, cachePlanError("unbound_plan", -1)
	}
	decision, err := owned.CachePlan.view.Decide(owned, owned.CachePlan, model)
	if ctxErr := ctx.Err(); ctxErr != nil {
		return InvokeRequest{}, nil, ctxErr
	}
	if err != nil {
		return InvokeRequest{}, nil, err
	}
	if len(decision.Plan.Directives) != 0 {
		return InvokeRequest{}, nil, cachePlanError("unmapped_plan", -1)
	}
	const diagnosticLimit = 32
	count := len(decision.Directives)
	if count > diagnosticLimit {
		count = diagnosticLimit
	}
	diagnostics := make([]Diagnostic, 0, count+1)
	for _, d := range decision.Directives[:count] {
		diagnostics = append(diagnostics, Diagnostic{Kind: "cache_plan_skipped", Message: fmt.Sprintf("directive %d: %s", d.DirectiveIndex, d.Reason)})
	}
	if count < len(decision.Directives) {
		diagnostics = append(diagnostics, Diagnostic{Kind: "cache_plan_skipped_summary", Message: fmt.Sprintf("%d additional directives skipped", len(decision.Directives)-count)})
	}
	owned.CachePlan = nil
	if warnf != nil {
		for _, d := range diagnostics {
			warnf("%s: %s", d.Kind, d.Message)
		}
	}
	return owned, diagnostics, nil
}
