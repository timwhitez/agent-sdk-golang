package llm

import "testing"

type cacheCapabilityFixture struct {
	capabilities PromptCacheCapabilities
}

var _ PromptCacheCapabilityProvider = cacheCapabilityFixture{}

func (fixture cacheCapabilityFixture) PromptCacheCapabilities() PromptCacheCapabilities {
	return fixture.capabilities.Clone()
}

func TestCloneCachePlanOwnsDirectives(t *testing.T) {
	plan := &CachePlan{
		SchemaVersion:      CachePlanSchemaVersion,
		RequestFingerprint: "request:v1:abc",
		Directives: []CacheDirective{{
			Target: CacheTarget{
				Kind:                      CacheAfterMessageBlock,
				MessageIndex:              2,
				BlockOrdinal:              1,
				ExpectedObjectFingerprint: "object:v1:def",
			},
			Policy: CacheRequired,
			TTL:    CacheTTL5Minutes,
		}},
	}
	cloned := CloneCachePlan(plan)
	plan.Directives[0].Target.MessageIndex = 9
	plan.Directives[0].Target.ExpectedObjectFingerprint = "mutated"
	if cloned == plan || cloned.Directives[0].Target.MessageIndex != 2 || cloned.Directives[0].Target.ExpectedObjectFingerprint != "object:v1:def" {
		t.Fatalf("clone shared mutable plan state: %#v", cloned)
	}
}

func TestCloneCachePlanPreservesNilAndEmpty(t *testing.T) {
	if CloneCachePlan(nil) != nil {
		t.Fatal("nil plan became non-nil")
	}
	cloned := CloneCachePlan(&CachePlan{Directives: []CacheDirective{}})
	if cloned.Directives == nil {
		t.Fatal("non-nil empty directives became nil")
	}
}

func TestPromptCacheCapabilitiesCloneOwnsTTLs(t *testing.T) {
	capabilities := PromptCacheCapabilities{
		ExplicitMessageBoundary: true,
		SupportedTTLs:           []CacheTTL{CacheTTL5Minutes, CacheTTL1Hour},
		MaxBreakpoints:          4,
		UsageTelemetry:          true,
	}
	cloned := capabilities.Clone()
	capabilities.SupportedTTLs[0] = "mutated"
	if cloned.SupportedTTLs[0] != CacheTTL5Minutes {
		t.Fatalf("capability clone shared TTL slice: %#v", cloned)
	}
	empty := (PromptCacheCapabilities{SupportedTTLs: []CacheTTL{}}).Clone()
	if empty.SupportedTTLs == nil {
		t.Fatal("non-nil empty SupportedTTLs became nil")
	}
}
