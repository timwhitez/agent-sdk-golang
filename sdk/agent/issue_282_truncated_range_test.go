package agent

import (
	"context"
	"fmt"
	"github.com/timwhitez/agent-sdk-golang/sdk/agent/compaction"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools/sandbox"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestIndependentTruncatedReadDoesNotCoverUnreturnedRange(t *testing.T) {
	for _, tc := range []struct {
		name      string
		lineChars int
		maxBytes  int
	}{
		{"sandbox_output_cap", 1000, 1 << 20},
		{"agent_output_cap", 100, 4096},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			if err := os.WriteFile(filepath.Join(dir, "report.txt"), []byte(strings.Repeat(strings.Repeat("x", tc.lineChars)+"\n", 1499)+"UNSEEN_RANGE_SENTINEL\n"+strings.Repeat(strings.Repeat("x", tc.lineChars)+"\n", 500)), 0600); err != nil {
				t.Fatal(err)
			}
			sb, err := sandbox.New(dir)
			if err != nil {
				t.Fatal(err)
			}
			deps := tools.NewContainer()
			tools.Provide(deps, sandbox.Key, func(context.Context) (*sandbox.Sandbox, error) { return sb, nil })
			model := &turnModel{}
			for i, limit := range []int{2000, 2000, 1} {
				offset := 1
				if i == 2 {
					offset = 1500
				}
				model.turns = append(model.turns, &llm.Completion{ToolCalls: []llm.ToolCall{issue282Call(fmt.Sprint(i), "read", fmt.Sprintf(`{"file_path":"report.txt","offset":%d,"limit":%d}`, offset, limit))}})
			}
			var read tools.Tool
			for _, tool := range sandbox.Tools() {
				if tool.Name == "read" {
					read = tool
				}
			}
			a, err := New(Config{LLM: model, Tools: []tools.Tool{read}, Deps: deps, MaxToolResultBytes: tc.maxBytes, Compaction: &compaction.Config{Enabled: false, ContextWindow: 10_000_000}})
			if err != nil {
				t.Fatal(err)
			}
			var results []ToolResultEvent
			for ev := range a.QueryStream(t.Context(), llm.TextContent("read the report")) {
				if r, ok := ev.(ToolResultEvent); ok {
					results = append(results, r)
				}
			}
			if len(results) != 3 || results[0].IsError || results[1].IsError {
				t.Fatalf("expected three successful results; received %d", len(results))
			}
			if strings.Contains(results[0].Result, "UNSEEN_RANGE_SENTINEL") || strings.Contains(results[1].Result, "UNSEEN_RANGE_SENTINEL") {
				t.Fatal("fixture returned requested target before third request")
			}
			if !strings.Contains(results[2].Result, "UNSEEN_RANGE_SENTINEL") {
				t.Fatalf("two truncated reads followed by unreturned offset=1500 yielded %q; want previously unseen line", results[2].Result)
			}
		})
	}
}

func TestTruncatedEvidenceStillDetectsIdenticalRequests(t *testing.T) {
	ledger := newEvidenceProgressLedger(nil, 0)
	req, ok := newEvidenceRequest(t.Context(), "read", nil, `{"file_path":"report.txt","offset":1,"limit":2000}`, nil)
	if !ok {
		t.Fatal("read not classified")
	}
	ledger.observe(req, "same truncated prefix", false, false)
	ledger.observe(req, "same truncated prefix", false, false)
	if decision := ledger.preflight(req, 0); !decision.suppress {
		t.Fatal("identical repeated truncated request was not suppressed")
	}
	next, ok := newEvidenceRequest(t.Context(), "read", nil, `{"file_path":"report.txt","offset":1500,"limit":1}`, nil)
	if !ok {
		t.Fatal("read not classified")
	}
	if decision := ledger.preflight(next, 0); decision.suppress {
		t.Fatal("unreturned range was marked as covered")
	}
}
