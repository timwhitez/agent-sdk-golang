package llm

import (
	"bytes"
	"encoding/json"
	"reflect"
	"testing"
)

func TestInvokeRequestCachePlanOwnershipAndJSONBoundary(t *testing.T) {
	for _, plan := range []*CachePlan{nil, {}, {Directives: []CacheDirective{}}, {
		SchemaVersion: CachePlanSchemaVersion, RequestFingerprint: "private-marker",
		Directives: []CacheDirective{{Target: CacheTarget{Kind: CacheAfterMessageBlock, MessageIndex: 3, BlockOrdinal: 2, ExpectedObjectFingerprint: "private-marker"}, Policy: CacheRequired, TTL: CacheTTL1Hour}},
	}} {
		request := InvokeRequest{Messages: []Message{NewUserMessage("fixture")}, CachePlan: plan}
		cloned, err := CloneInvokeRequest(request)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(cloned, request) {
			t.Fatal("clone changed nil/empty/plan fields")
		}
		if plan != nil {
			if cloned.CachePlan == plan {
				t.Fatal("shared plan pointer")
			}
			cloned.CachePlan.RequestFingerprint = "clone-only"
			if len(plan.Directives) > 0 {
				cloned.CachePlan.Directives[0].Target.MessageIndex = 99
				if plan.Directives[0].Target.MessageIndex != 3 {
					t.Fatal("shared directives")
				}
			}
			if plan.RequestFingerprint == "clone-only" {
				t.Fatal("shared plan scalar")
			}
		}
		with, err := json.Marshal(request)
		if err != nil {
			t.Fatal(err)
		}
		request.CachePlan = nil
		without, err := json.Marshal(request)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(with, without) || bytes.Contains(with, []byte("private-marker")) {
			t.Fatal("plan leaked into request JSON")
		}
		var restored InvokeRequest
		if err := json.Unmarshal([]byte(`{"CachePlan":{"RequestFingerprint":"private-marker"}}`), &restored); err != nil {
			t.Fatal(err)
		}
		if restored.CachePlan != nil {
			t.Fatal("JSON restored request-local plan")
		}
	}
}
