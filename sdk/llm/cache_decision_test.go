package llm_test

import (
	"context"
	"encoding/json"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm/anthropic"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm/openai"
)

type cacheDecisionNoCapabilities struct{}

func (cacheDecisionNoCapabilities) Provider() string { panic("must not infer capabilities from names") }
func (cacheDecisionNoCapabilities) Model() string    { panic("must not infer capabilities from names") }
func (cacheDecisionNoCapabilities) Invoke(context.Context, llm.InvokeRequest) (*llm.Completion, error) {
	panic("decision must not invoke model")
}

type cacheDecisionModel struct {
	cacheDecisionNoCapabilities
	caps   llm.PromptCacheCapabilities
	reads  atomic.Int32
	onRead func()
}

func (m *cacheDecisionModel) PromptCacheCapabilities() llm.PromptCacheCapabilities {
	m.reads.Add(1)
	if m.onRead != nil {
		m.onRead()
	}
	return m.caps
}

func cacheDecisionFixture(t *testing.T) (llm.InvokeRequest, *llm.CacheTargetView, *llm.CachePlan) {
	t.Helper()
	request := llm.InvokeRequest{Messages: []llm.Message{
		{Role: llm.RoleUser, Content: llm.TextContent("private-zero")},
		{Role: llm.RoleUser, Content: llm.TextContent("private-one")},
		{Role: llm.RoleUser, Content: llm.TextContent("private-two")},
	}, Tools: []llm.ToolDefinition{{Name: "private-tool"}}}
	view, err := llm.NewCacheTargetView(request)
	if err != nil {
		t.Fatal(err)
	}
	plan := &llm.CachePlan{SchemaVersion: llm.CachePlanSchemaVersion}
	for i := range request.Messages {
		plan.Directives = append(plan.Directives, llm.CacheDirective{Target: llm.CacheTarget{Kind: llm.CacheAfterMessage, MessageIndex: i}, Policy: llm.CacheBestEffort})
	}
	plan.Directives[2].Policy = llm.CacheRequired
	return request, view, plan
}

func TestCacheDecisionRequiredReservationGoldenAndOwnership(t *testing.T) {
	request, view, plan := cacheDecisionFixture(t)
	before := llm.CloneCachePlan(plan)
	model := &cacheDecisionModel{caps: llm.PromptCacheCapabilities{ExplicitMessageBoundary: true, MaxBreakpoints: 2, SupportedTTLs: []llm.CacheTTL{llm.CacheTTL1Hour}}}
	result, err := view.Decide(request, plan, model)
	if err != nil {
		t.Fatal(err)
	}
	want := []llm.CacheDirectiveDecision{{DirectiveIndex: 0, Accepted: true, Reason: "accepted"}, {DirectiveIndex: 1, Reason: "breakpoint_limit"}, {DirectiveIndex: 2, Accepted: true, Reason: "accepted"}}
	if !reflect.DeepEqual(result.Directives, want) || !reflect.DeepEqual(result.Plan.Directives, []llm.CacheDirective{before.Directives[0], before.Directives[2]}) {
		t.Fatalf("decision golden mismatch: %+v", result)
	}
	if model.reads.Load() != 1 || !reflect.DeepEqual(plan, before) {
		t.Fatal("capability reads or input mutation")
	}
	encoded, err := json.Marshal(result)
	if err != nil || strings.Contains(string(encoded), "private-") {
		t.Fatal("decision leaked content")
	}
	result.Plan.Directives[0].Target.MessageIndex = 99
	result.Capabilities.SupportedTTLs[0] = "changed"
	if !reflect.DeepEqual(plan, before) || model.caps.SupportedTTLs[0] != llm.CacheTTL1Hour {
		t.Fatal("result aliases caller/model storage")
	}
	// A capability callback holding the original plan cannot rewrite validated intent.
	model.onRead = func() { plan.Directives[0].Target.MessageIndex = 99 }
	result, err = view.Decide(request, plan, model)
	if err != nil || result.Plan.Directives[0].Target.MessageIndex != 0 {
		t.Fatal("callback changed owned plan", err)
	}
}

func TestCacheDecisionPolicyMatrix(t *testing.T) {
	for _, test := range []struct {
		name   string
		kind   llm.CacheTargetKind
		ttl    llm.CacheTTL
		caps   llm.PromptCacheCapabilities
		reason string
	}{
		{"unknown", llm.CacheAfterMessage, "", llm.PromptCacheCapabilities{}, "unsupported_target"},
		{"message", llm.CacheAfterMessage, "", llm.PromptCacheCapabilities{ExplicitMessageBoundary: true, MaxBreakpoints: 1}, "accepted"},
		{"block", llm.CacheAfterMessageBlock, "", llm.PromptCacheCapabilities{ExplicitContentBlock: true, MaxBreakpoints: 1}, "accepted"},
		{"definition", llm.CacheAfterToolDefinition, "", llm.PromptCacheCapabilities{ExplicitToolDefinition: true, MaxBreakpoints: 1}, "accepted"},
		{"wrong-target-flag", llm.CacheAfterMessageBlock, "", llm.PromptCacheCapabilities{ExplicitMessageBoundary: true, MaxBreakpoints: 1}, "unsupported_target"},
		{"ttl", llm.CacheAfterMessage, llm.CacheTTL5Minutes, llm.PromptCacheCapabilities{ExplicitMessageBoundary: true, MaxBreakpoints: 1}, "unsupported_ttl"},
		{"ttl-supported", llm.CacheAfterMessage, llm.CacheTTL1Hour, llm.PromptCacheCapabilities{ExplicitMessageBoundary: true, MaxBreakpoints: 1, SupportedTTLs: []llm.CacheTTL{llm.CacheTTL1Hour}}, "accepted"},
		{"zero-limit", llm.CacheAfterMessage, "", llm.PromptCacheCapabilities{ExplicitMessageBoundary: true}, "breakpoint_limit"},
	} {
		for _, policy := range []llm.CacheDirectivePolicy{llm.CacheRequired, llm.CacheBestEffort} {
			t.Run(test.name+"/"+string(policy), func(t *testing.T) {
				request, view, plan := cacheDecisionFixture(t)
				plan.Directives = []llm.CacheDirective{{Target: llm.CacheTarget{Kind: test.kind}, TTL: test.ttl, Policy: policy}}
				model := &cacheDecisionModel{caps: test.caps}
				result, err := view.Decide(request, plan, model)
				if test.reason != "accepted" && policy == llm.CacheRequired {
					assertCacheViewError(t, err, test.reason, 0)
					if result != nil {
						t.Fatal("required failure returned partial plan")
					}
					return
				}
				if err != nil {
					t.Fatal(err)
				}
				accepted := test.reason == "accepted"
				if len(result.Directives) != 1 || result.Directives[0].Reason != test.reason || result.Directives[0].Accepted != accepted || (len(result.Plan.Directives) == 1) != accepted {
					t.Fatalf("unexpected decision: %+v", result)
				}
			})
		}
	}
}

func TestCacheDecisionFailuresAndNoCapabilityGuessing(t *testing.T) {
	request, view, plan := cacheDecisionFixture(t)
	model := &cacheDecisionModel{caps: llm.PromptCacheCapabilities{ExplicitMessageBoundary: true, MaxBreakpoints: 1}}
	for i := range plan.Directives {
		plan.Directives[i].Policy = llm.CacheRequired
	}
	result, err := view.Decide(request, plan, model)
	assertCacheViewError(t, err, "breakpoint_limit", 1)
	if result != nil {
		t.Fatal("overflow returned partial plan")
	}
	for _, caps := range []llm.PromptCacheCapabilities{{MaxBreakpoints: -1}, {SupportedTTLs: []llm.CacheTTL{"private-invalid-ttl"}}} {
		model.caps = caps
		_, err := view.Decide(request, plan, model)
		assertCacheViewError(t, err, "invalid_capabilities", -1)
	}
	_, err = view.Decide(request, plan, cacheDecisionNoCapabilities{})
	assertCacheViewError(t, err, "unsupported_target", 0)
	var nilModel *cacheDecisionModel
	_, err = view.Decide(request, plan, nilModel)
	assertCacheViewError(t, err, "missing_model", -1)
	model.reads.Store(0)
	request.Messages[0].Content.Text = "changed"
	_, err = view.Decide(request, plan, model)
	assertCacheViewError(t, err, "stale_request", -1)
	if model.reads.Load() != 0 {
		t.Fatal("stale request reached capability callback")
	}
	var nilView *llm.CacheTargetView
	result, err = nilView.Decide(request, nil, nil)
	if err != nil || result.Plan != nil || result.Directives != nil {
		t.Fatal("nil plan compatibility")
	}
}

func TestCacheDecisionBuiltinCapabilitiesAreConservative(t *testing.T) {
	for _, model := range []llm.ChatModel{&anthropic.Client{}, &openai.ChatClient{}, &openai.ResponsesClient{}} {
		provider, ok := model.(llm.PromptCacheCapabilityProvider)
		want := llm.PromptCacheCapabilities{UsageTelemetry: true}
		if _, anthropicClient := model.(*anthropic.Client); anthropicClient {
			want.ExplicitMessageBoundary = true
			want.ExplicitToolDefinition, want.MaxBreakpoints = true, 4
			want.SupportedTTLs = []llm.CacheTTL{llm.CacheTTL5Minutes}
		}
		if !ok || !reflect.DeepEqual(provider.PromptCacheCapabilities(), want) {
			t.Fatal("builtin claims unimplemented explicit control")
		}
		request, view, plan := cacheDecisionFixture(t)
		for i := range plan.Directives {
			plan.Directives[i].Target.Kind = llm.CacheAfterMessageBlock
		}
		_, err := view.Decide(request, plan, model)
		assertCacheViewError(t, err, "unsupported_target", 2)
		plan.Directives[2].Policy = llm.CacheBestEffort
		result, err := view.Decide(request, plan, model)
		if err != nil || len(result.Plan.Directives) != 0 || len(result.Directives) != 3 {
			t.Fatal("builtin best-effort projection", err)
		}
	}
}

func TestCacheDecisionConcurrentReadOnly(t *testing.T) {
	request, view, plan := cacheDecisionFixture(t)
	model := &cacheDecisionModel{caps: llm.PromptCacheCapabilities{ExplicitMessageBoundary: true, MaxBreakpoints: 3}}
	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 20; j++ {
				if _, err := view.Decide(request, plan, model); err != nil {
					t.Error(err)
				}
			}
		}()
	}
	wg.Wait()
	if model.reads.Load() != 160 {
		t.Fatal("capability snapshot count")
	}
}

func BenchmarkCacheDecision(b *testing.B) {
	request := llm.InvokeRequest{Messages: []llm.Message{{Role: llm.RoleUser, Content: llm.TextContent(strings.Repeat("x", 1024))}}}
	view, err := llm.NewCacheTargetView(request)
	if err != nil {
		b.Fatal(err)
	}
	plan := &llm.CachePlan{SchemaVersion: llm.CachePlanSchemaVersion, Directives: []llm.CacheDirective{{Target: llm.CacheTarget{Kind: llm.CacheAfterMessage}, Policy: llm.CacheRequired}}}
	model := &cacheDecisionModel{caps: llm.PromptCacheCapabilities{ExplicitMessageBoundary: true, MaxBreakpoints: 1}}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := view.Decide(request, plan, model); err != nil {
			b.Fatal(err)
		}
	}
}

type cacheEligibilityModel struct {
	cacheDecisionModel
	hook func(llm.InvokeRequest, []llm.CacheTarget) []bool
}

func (m *cacheEligibilityModel) PromptCacheTargetEligibility(request llm.InvokeRequest, targets []llm.CacheTarget) []bool {
	return m.hook(request, targets)
}

func TestCacheEligibilitySnapshotOwnershipAndPreallocation(t *testing.T) {
	request, view, plan := cacheDecisionFixture(t)
	before, err := llm.CloneInvokeRequest(request)
	if err != nil {
		t.Fatal(err)
	}
	mask := []bool{false, true, true}
	model := &cacheEligibilityModel{cacheDecisionModel: cacheDecisionModel{caps: llm.PromptCacheCapabilities{ExplicitMessageBoundary: true, MaxBreakpoints: 2}}}
	model.hook = func(copy llm.InvokeRequest, targets []llm.CacheTarget) []bool {
		if !reflect.DeepEqual(copy, before) {
			t.Error("eligibility did not receive validated original snapshot")
		}
		copy.Messages[0].Content.Text = "private-mutated"
		copy.Tools[0].Name = "private-mutated"
		targets[1].MessageIndex = 99
		return mask
	}
	result, err := view.Decide(request, plan, model)
	if err != nil {
		t.Fatal(err)
	}
	if result.Directives[0].Reason != "unmappable_target" || len(result.Plan.Directives) != 2 || result.Plan.Directives[0].Target.MessageIndex != 1 || result.Plan.Directives[1].Target.MessageIndex != 2 {
		t.Fatal("eligibility was applied after capacity or exposed target aliases")
	}
	mask[1] = false
	if !result.Directives[1].Accepted || !reflect.DeepEqual(request, before) {
		t.Fatal("eligibility exposed caller or result aliases")
	}
	if err := view.Validate(request, plan); err != nil {
		t.Fatal("provider mutated retained snapshot", err)
	}
	changed, _ := llm.CloneInvokeRequest(request)
	changed.Messages[0].Content.Text = "new-view"
	fresh, err := llm.NewCacheTargetView(changed)
	if err != nil {
		t.Fatal(err)
	}
	model.onRead = func() { *view = *fresh }
	mask[1] = true
	if _, err := view.Decide(request, plan, model); err != nil {
		t.Fatal(err)
	}
}

func TestCacheEligibilityMalformedAndStale(t *testing.T) {
	request, view, plan := cacheDecisionFixture(t)
	model := &cacheEligibilityModel{cacheDecisionModel: cacheDecisionModel{caps: llm.PromptCacheCapabilities{ExplicitMessageBoundary: true, MaxBreakpoints: 4}}}
	var calls int
	model.hook = func(llm.InvokeRequest, []llm.CacheTarget) []bool { calls++; return []bool{true} }
	result, err := view.Decide(request, plan, model)
	assertCacheViewError(t, err, "invalid_capabilities", -1)
	if result != nil || calls != 1 {
		t.Fatal("malformed hook produced partial result")
	}
	request.Messages[0].Content.Text = "changed"
	_, err = view.Decide(request, plan, model)
	assertCacheViewError(t, err, "stale_request", -1)
	if calls != 1 {
		t.Fatal("stale request reached mapper")
	}
}
