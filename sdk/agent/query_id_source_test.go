package agent

import "testing"

func TestQueryIDSourceDoesNotInferOrInvokeCustomGenerator(t *testing.T) {
	var missing *Agent
	if missing.UsesDefaultQueryIDGenerator() {
		t.Fatal("nil agent is not an ID source")
	}
	if !new(Agent).UsesDefaultQueryIDGenerator() {
		t.Fatal("default source not recognized")
	}
	for _, id := range []string{"", "query_0123456789abcdef0123456789abcdef", "PRIVATE_ID"} {
		calls := 0
		ag := &Agent{queryIDGenerator: func() string { calls++; return id }}
		if ag.UsesDefaultQueryIDGenerator() || calls != 0 {
			t.Fatal("custom generator trusted or invoked", id, calls)
		}
	}
}
