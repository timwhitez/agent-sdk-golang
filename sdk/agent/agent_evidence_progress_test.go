package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

type evidenceReadArgs struct {
	FilePath string `json:"filePath"`
	Offset   int    `json:"offset,omitempty"`
	Limit    int    `json:"limit,omitempty"`
}

type evidenceDoneArgs struct {
	Message string `json:"message"`
}

type evidenceFixtureModel struct {
	mu    sync.Mutex
	calls int
}

func (m *evidenceFixtureModel) Provider() string { return "fixture" }
func (m *evidenceFixtureModel) Model() string    { return "evidence-progress" }

func (m *evidenceFixtureModel) Invoke(context.Context, llm.InvokeRequest) (*llm.Completion, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.calls++
	call := func(id, name, args string) llm.ToolCall {
		return llm.ToolCall{ID: id, Type: "function", Function: llm.FunctionCall{Name: name, Arguments: args}}
	}
	switch m.calls {
	case 1:
		return &llm.Completion{ToolCalls: []llm.ToolCall{call("read-1", "read", `{"filePath":"a.go","offset":1,"limit":100}`)}}, nil
	case 2:
		return &llm.Completion{ToolCalls: []llm.ToolCall{call("read-2", "read_file", `{"filePath":"a.go","offset":1,"limit":100}`)}}, nil
	default:
		return &llm.Completion{ToolCalls: []llm.ToolCall{
			call("read-3", "read", `{"filePath":"a.go","offset":2,"limit":50}`),
			call("read-4", "read", `{"filePath":"a.go","offset":101,"limit":100}`),
			call("done-1", "done", `{"message":"complete"}`),
		}}, nil
	}
}

func TestEvidenceProgressSuppressesCoveredReadAndPreservesMixedBatch(t *testing.T) {
	model := &evidenceFixtureModel{}
	readExecutions := 0
	readTool := tools.Func[evidenceReadArgs]("read", "read", func(_ context.Context, args evidenceReadArgs, _ *tools.Container) (any, error) {
		readExecutions++
		if args.Offset >= 101 {
			return "101: new block", nil
		}
		return "1: same block", nil
	})
	readAlias := readTool
	readAlias.Name = "read_file"
	doneTool := tools.Func[evidenceDoneArgs]("done", "done", func(_ context.Context, args evidenceDoneArgs, _ *tools.Container) (any, error) {
		return nil, &tools.TaskCompleteError{Message: args.Message}
	})
	ag, err := New(Config{
		LLM: model, Tools: []tools.Tool{readTool, readAlias, doneTool},
		MaxIterations: -1, RequireDoneTool: true,
	})
	if err != nil {
		t.Fatal(err)
	}

	var warnings []WarnEvent
	var results []ToolResultEvent
	var final FinalResponseEvent
	for ev := range ag.QueryStream(context.Background(), llm.TextContent("inspect")) {
		switch e := ev.(type) {
		case WarnEvent:
			warnings = append(warnings, e)
		case ToolResultEvent:
			results = append(results, e)
		case FinalResponseEvent:
			final = e
		}
	}
	if readExecutions != 3 {
		t.Fatalf("read executions = %d, want 3 (two validation reads plus uncovered range)", readExecutions)
	}
	if final.Content != "complete" {
		t.Fatalf("final = %#v", final)
	}
	recoveryWarnings := 0
	var recoveryMetadata map[string]any
	suppressed := 0
	for _, warning := range warnings {
		if warning.Kind == "no_progress_recovery" {
			recoveryWarnings++
			recoveryMetadata = warning.Metadata
		}
	}
	for _, result := range results {
		if result.Metadata != nil && result.Metadata["no_progress_suppressed"] == true {
			suppressed++
			if result.IsError || !strings.Contains(result.Result, "already_observed") {
				t.Fatalf("suppressed result = %#v", result)
			}
		}
	}
	if recoveryWarnings != 1 || suppressed != 1 {
		t.Fatalf("recovery warnings=%d suppressed=%d", recoveryWarnings, suppressed)
	}
	if recoveryMetadata["evidence_family"] != "read" || recoveryMetadata["evidence_suppressed"] != 1 {
		t.Fatalf("recovery metadata = %#v", recoveryMetadata)
	}

	toolResults := map[string]bool{}
	for _, message := range ag.Messages() {
		if message.Role == llm.RoleTool {
			toolResults[message.ToolCallID] = true
		}
	}
	for _, id := range []string{"read-1", "read-2", "read-3", "read-4", "done-1"} {
		if !toolResults[id] {
			t.Fatalf("missing tool result for %s; results=%v", id, toolResults)
		}
	}
	foundRecoveryOrigin := false
	for _, message := range ag.Messages() {
		if message.Name == "sdk_internal_evidence_recovery" {
			foundRecoveryOrigin = true
			break
		}
	}
	if !foundRecoveryOrigin {
		t.Fatalf("history is missing named evidence recovery: %#v", ag.Messages())
	}
}

func TestEvidenceProgressDoesNotCacheStateChangingTools(t *testing.T) {
	ledger := newEvidenceProgressLedger(nil, 0)
	if family := evidenceFamily("write"); family != "" {
		t.Fatalf("write classified as evidence family %q", family)
	}
	ledger.targets["read|a.go"] = &evidenceTargetState{}
	ledger.invalidateAfter("write", false)
	if len(ledger.targets) != 0 {
		t.Fatalf("state-changing tool did not invalidate read evidence: %#v", ledger.targets)
	}
}

func TestEvidenceProgressAllowsOneRevalidationAfterCompaction(t *testing.T) {
	ledger := newEvidenceProgressLedger(nil, 0)
	raw := json.RawMessage(`{"filePath":"a.go","offset":1,"limit":10}`)
	req, ok := newEvidenceRequest("read", raw, string(raw), nil)
	if !ok {
		t.Fatal("read request not classified")
	}
	first := ledger.observe(req, "same", false)
	if first["evidence_disposition"] != "first_seen" {
		t.Fatalf("first evidence disposition = %#v", first)
	}
	repeated := ledger.observe(req, "same", false)
	if repeated["evidence_disposition"] != "repeated_same_generation" {
		t.Fatalf("same-generation evidence disposition = %#v", repeated)
	}
	if decision := ledger.preflight(req, 0); !decision.suppress {
		t.Fatal("expected covered request to be suppressed")
	}
	if decision := ledger.preflight(req, 1); decision.suppress {
		t.Fatalf("first post-compaction revalidation was suppressed: %#v", decision)
	}
	afterCompaction := ledger.observe(req, "same", false)
	if afterCompaction["evidence_disposition"] != "repeated_after_compaction" || afterCompaction["evidence_prior_generation"] != uint64(0) {
		t.Fatalf("post-compaction evidence disposition = %#v", afterCompaction)
	}
	if decision := ledger.preflight(req, 1); !decision.suppress {
		t.Fatal("second post-compaction repeat should be suppressed")
	} else if decision.metadata["evidence_disposition"] != "recovery_failed" {
		t.Fatalf("suppressed recovery disposition = %#v", decision.metadata)
	}
}

func TestEvidenceProgressCanonicalizesReadAliasesAndRanges(t *testing.T) {
	first, ok := newEvidenceRequest("read", json.RawMessage(`{"filePath":"./a.go","offset":1,"limit":20}`), "", nil)
	if !ok {
		t.Fatal("read not classified")
	}
	second, ok := newEvidenceRequest("read_file", json.RawMessage(`{"filePath":"a.go","offset":2,"limit":5}`), "", nil)
	if !ok {
		t.Fatal("read_file not classified")
	}
	if first.family != second.family || first.target != second.target {
		t.Fatalf("aliases not canonicalized: %#v vs %#v", first, second)
	}
	ledger := newEvidenceProgressLedger(nil, 0)
	ledger.observe(first, "content", false)
	ledger.observe(first, "content", false)
	decision := ledger.preflight(second, 0)
	if !decision.suppress {
		t.Fatalf("covered alias range was not suppressed: %s", fmt.Sprint(decision.metadata))
	}
}

func TestEvidenceProgressSeparatesByteRangeFromLineRange(t *testing.T) {
	line, ok := newEvidenceRequest("read", json.RawMessage(`{"filePath":"a.go","offset":1,"limit":2000}`), "", nil)
	if !ok {
		t.Fatal("line read not classified")
	}
	byteRange, ok := newEvidenceRequest("read", json.RawMessage(`{"filePath":"a.go","byte_offset":0,"byte_limit":64}`), "", nil)
	if !ok {
		t.Fatal("byte read not classified")
	}
	if line.key != "read|a.go|line" {
		t.Fatalf("line key = %q, want read|a.go|line", line.key)
	}
	if byteRange.key != "read|a.go|byte" {
		t.Fatalf("byte key = %q, want read|a.go|byte", byteRange.key)
	}

	ledger := newEvidenceProgressLedger(nil, 0)
	ledger.observe(line, "same visible prefix", false)
	ledger.observe(line, "same visible prefix", false)
	if decision := ledger.preflight(byteRange, 0); decision.suppress {
		t.Fatalf("line evidence covered a byte range: %#v", decision)
	}
}

func TestEvidenceProgressByteRangeCoverageAcrossAliases(t *testing.T) {
	first, ok := newEvidenceRequest("read", json.RawMessage(`{"filePath":"./a.go","byte_offset":0,"byte_limit":64}`), "", nil)
	if !ok {
		t.Fatal("first byte read not classified")
	}
	distinct, ok := newEvidenceRequest("read_file", json.RawMessage(`{"filePath":"a.go","byte_offset":64,"byte_limit":64}`), "", nil)
	if !ok {
		t.Fatal("distinct byte read alias not classified")
	}
	covered, ok := newEvidenceRequest("read_file", json.RawMessage(`{"filePath":"a.go","byte_offset":16,"byte_limit":16}`), "", nil)
	if !ok {
		t.Fatal("covered byte read alias not classified")
	}
	if first.key != "read|a.go|byte" || distinct.key != first.key || covered.key != first.key {
		t.Fatalf("byte aliases do not share coverage: first=%q distinct=%q covered=%q", first.key, distinct.key, covered.key)
	}

	ledger := newEvidenceProgressLedger(nil, 0)
	if decision := ledger.preflight(first, 0); decision.suppress {
		t.Fatalf("first byte range was suppressed: %#v", decision)
	}
	ledger.observe(first, "chunk-a", false)
	ledger.observe(first, "chunk-a", false)
	if decision := ledger.preflight(distinct, 0); decision.suppress {
		t.Fatalf("uncovered byte range was suppressed: %#v", decision)
	}
	ledger.observe(distinct, "chunk-b", false)
	ledger.observe(distinct, "chunk-b", false)
	if decision := ledger.preflight(covered, 0); !decision.suppress {
		t.Fatalf("covered byte subrange was not suppressed after no progress: %#v", decision)
	}
}

func TestEvidenceProgressByteRangeAllowsOneRevalidationAfterCompaction(t *testing.T) {
	ledger := newEvidenceProgressLedger(nil, 0)
	raw := json.RawMessage(`{"filePath":"a.go","byte_offset":1048576,"byte_limit":16384}`)
	req, ok := newEvidenceRequest("read_file", raw, string(raw), nil)
	if !ok {
		t.Fatal("byte read request not classified")
	}
	ledger.observe(req, "same byte window", false)
	ledger.observe(req, "same byte window", false)
	if decision := ledger.preflight(req, 0); !decision.suppress {
		t.Fatal("expected covered byte range to be suppressed")
	}
	if decision := ledger.preflight(req, 1); decision.suppress {
		t.Fatalf("first byte-range revalidation after compaction was suppressed: %#v", decision)
	}
	afterCompaction := ledger.observe(req, "same byte window", false)
	if afterCompaction["evidence_disposition"] != "repeated_after_compaction" || afterCompaction["evidence_prior_generation"] != uint64(0) {
		t.Fatalf("post-compaction byte evidence disposition = %#v", afterCompaction)
	}
	if decision := ledger.preflight(req, 1); !decision.suppress {
		t.Fatal("second post-compaction byte repeat should be suppressed")
	}
}

func TestEvidenceProgressByteRangeResetsAfterTargetStateChange(t *testing.T) {
	path := filepath.Join(t.TempDir(), "long.min.js")
	if err := os.WriteFile(path, []byte("first"), 0o600); err != nil {
		t.Fatal(err)
	}
	ledger := newEvidenceProgressLedger(nil, 0)
	raw := json.RawMessage(fmt.Sprintf(`{"filePath":%q,"byte_offset":0,"byte_limit":5}`, path))
	first, ok := newEvidenceRequest("read", raw, string(raw), nil)
	if !ok {
		t.Fatal("byte read request not classified")
	}
	ledger.observe(first, "first", false)
	ledger.observe(first, "first", false)
	if decision := ledger.preflight(first, 0); !decision.suppress {
		t.Fatal("expected unchanged byte range to be suppressed")
	}
	if err := os.WriteFile(path, []byte("second and larger"), 0o600); err != nil {
		t.Fatal(err)
	}
	changed, ok := newEvidenceRequest("read_file", raw, string(raw), nil)
	if !ok {
		t.Fatal("changed byte read alias not classified")
	}
	if changed.key != first.key {
		t.Fatalf("byte alias target key changed: first=%q changed=%q", first.key, changed.key)
	}
	if decision := ledger.preflight(changed, 0); decision.suppress {
		t.Fatalf("changed target byte range was suppressed: %#v", decision)
	}
}

func TestEvidenceProgressInvalidatesAfterUnknownSuccessfulTool(t *testing.T) {
	ledger := newEvidenceProgressLedger(nil, 0)
	ledger.targets["read|a.go"] = &evidenceTargetState{}
	ledger.invalidateAfter("custom_repository_mutator", false)
	if len(ledger.targets) != 0 {
		t.Fatalf("unknown successful tool did not invalidate read evidence: %#v", ledger.targets)
	}
}

func TestEvidenceProgressAllowsReadAfterExternalTargetChange(t *testing.T) {
	path := filepath.Join(t.TempDir(), "a.go")
	if err := os.WriteFile(path, []byte("one\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	ledger := newEvidenceProgressLedger(nil, 0)
	raw := json.RawMessage(fmt.Sprintf(`{"filePath":%q,"offset":1,"limit":10}`, path))
	first, ok := newEvidenceRequest("read", raw, string(raw), nil)
	if !ok {
		t.Fatal("read request not classified")
	}
	ledger.observe(first, "one", false)
	ledger.observe(first, "one", false)
	if decision := ledger.preflight(first, 0); !decision.suppress {
		t.Fatal("expected unchanged target to be suppressed")
	}
	if err := os.WriteFile(path, []byte("two and larger\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	changed, ok := newEvidenceRequest("read", raw, string(raw), nil)
	if !ok {
		t.Fatal("changed read request not classified")
	}
	if decision := ledger.preflight(changed, 0); decision.suppress {
		t.Fatalf("changed target was suppressed: %#v", decision)
	}
}
