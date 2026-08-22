package accounting

import (
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/agent/compaction"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

func testEstimator() Estimator {
	return Estimator{
		Name:       "goode_approx_tokens",
		Version:    "1",
		PolicyHash: "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
		EstimateTokens: func(text string) int {
			return len([]rune(text))
		},
	}
}

func TestProjectToolResultSeparatesLayersScanAndReturnCoverage(t *testing.T) {
	payload := ProjectToolResult(ToolResultInput{
		Tool:     "grep_files",
		Original: strings.Repeat("原", 200),
		Visible:  strings.Repeat("原", 20),
		Metadata: map[string]any{
			"output_bytes":            9000,
			"output_bytes_limit":      4096,
			"output_truncated":        true,
			"bytes":                   600,
			"truncated":               true,
			"result_original_bytes":   600,
			"result_bytes":            60,
			"result_max_bytes":        100,
			"result_truncated":        true,
			"scan_complete":           false,
			"return_truncated":        false,
			"matches_total_known":     false,
			"walk_entries_seen":       4096,
			"eligible_candidates":     200,
			"files_opened":            200,
			"files_matched":           3,
			"matches_returned":        3,
			"budget_exhausted_reason": "snapshot_walk_entries",
			"continuation":            "must-not-be-persisted",
			"pattern":                 "must-not-be-persisted",
		},
	}, testEstimator())

	if err := payload.Validate(); err != nil {
		t.Fatalf("payload validation: %v", err)
	}
	if payload.SchemaVersion != SchemaVersion || payload.EventKind != EventKindToolResult || payload.ToolKind != "grep" {
		t.Fatalf("payload identity = %#v", payload)
	}
	if payload.Measurements.OriginalBytes == nil || *payload.Measurements.OriginalBytes != 600 {
		t.Fatalf("original bytes = %#v", payload.Measurements.OriginalBytes)
	}
	if payload.Measurements.VisibleBytes == nil || *payload.Measurements.VisibleBytes != 60 {
		t.Fatalf("visible bytes = %#v", payload.Measurements.VisibleBytes)
	}
	if payload.Measurements.OriginalTokens == nil || *payload.Measurements.OriginalTokens != 200 {
		t.Fatalf("original tokens = %#v", payload.Measurements.OriginalTokens)
	}
	if payload.Measurements.VisibleTokens == nil || *payload.Measurements.VisibleTokens != 20 {
		t.Fatalf("visible tokens = %#v", payload.Measurements.VisibleTokens)
	}
	if len(payload.Layers) != 3 || payload.Layers[0].Layer != LayerProducer || payload.Layers[1].Layer != LayerWrapper || payload.Layers[2].Layer != LayerAgentBoundary {
		t.Fatalf("layers = %#v", payload.Layers)
	}
	if payload.Scan == nil || payload.Scan.ScanComplete == nil || *payload.Scan.ScanComplete {
		t.Fatalf("scan disposition = %#v", payload.Scan)
	}
	if payload.Scan.ReturnTruncated == nil || *payload.Scan.ReturnTruncated {
		t.Fatalf("return truncation must remain independent: %#v", payload.Scan)
	}
	if payload.Scan.BudgetReason != "snapshot_walk_entries" || !payload.Scan.ContinuationPresent {
		t.Fatalf("scan continuation = %#v", payload.Scan)
	}

	encoded, err := MarshalBounded(payload)
	if err != nil {
		t.Fatalf("marshal bounded: %v", err)
	}
	for _, forbidden := range []string{"must-not-be-persisted", "pattern", "continuation\""} {
		if strings.Contains(string(encoded), forbidden) {
			t.Fatalf("payload leaked %q: %s", forbidden, encoded)
		}
	}
}

func TestProjectToolResultRejectsArbitrarySecretAndRawMetadata(t *testing.T) {
	secret := "Bearer top-secret-value"
	raw := strings.Repeat("RAW_TOOL_OUTPUT", 100)
	payload := ProjectToolResult(ToolResultInput{
		Tool:     "bash",
		Original: "complete logical output",
		Visible:  "bounded output",
		Metadata: map[string]any{
			"authorization":        secret,
			"cookie":               "session=secret",
			"headers":              map[string]any{"Authorization": secret},
			"raw_output":           raw,
			"result_output_path":   "/tmp/secret-output",
			"result_output_ttl_ms": 900000,
		},
	}, testEstimator())

	encoded, err := json.Marshal(payload)
	if err != nil {
		t.Fatal(err)
	}
	for _, forbidden := range []string{secret, "session=secret", raw, "/tmp/secret-output", "authorization", "cookie", "headers", "raw_output"} {
		if strings.Contains(string(encoded), forbidden) {
			t.Fatalf("projection leaked %q: %s", forbidden, encoded)
		}
	}
	if payload.Artifact == nil || !payload.Artifact.LegacyPathPresent {
		t.Fatalf("legacy path presence was not projected safely: %#v", payload.Artifact)
	}
}

func TestProjectUsagePreservesProviderQualityAndUnknowns(t *testing.T) {
	rawPrompt := 0
	rawTotal := 7
	payload := ProjectUsage(llm.Usage{
		PromptTokens:          123,
		CompletionTokens:      7,
		TotalTokens:           130,
		ProviderPromptTokens:  &rawPrompt,
		ProviderTotalTokens:   &rawTotal,
		PromptTokensValid:     false,
		PromptTokensSource:    llm.PromptTokensSourceEstimate,
		PromptTokensSemantics: llm.PromptTokensSemanticsTotalInputV1,
	})
	if err := payload.Validate(); err != nil {
		t.Fatalf("payload validation: %v", err)
	}
	if payload.EventKind != EventKindProviderUsage || payload.Usage == nil {
		t.Fatalf("usage payload = %#v", payload)
	}
	if payload.Usage.EffectivePromptTokens == nil || *payload.Usage.EffectivePromptTokens != 123 {
		t.Fatalf("effective prompt = %#v", payload.Usage)
	}
	if payload.Usage.ProviderPromptTokens == nil || *payload.Usage.ProviderPromptTokens != 0 {
		t.Fatalf("raw provider prompt = %#v", payload.Usage)
	}
	if payload.Usage.PromptTokensValid || payload.Usage.PromptTokensSource != llm.PromptTokensSourceEstimate {
		t.Fatalf("usage quality = %#v", payload.Usage)
	}
	missing := ProjectUsage(llm.Usage{})
	if missing.Usage == nil || missing.Usage.EffectivePromptTokens != nil || missing.Usage.TotalTokens != nil {
		t.Fatalf("missing usage was fabricated as zero: %#v", missing.Usage)
	}
}

func TestProjectUsagePinsEstimatorForEstimatedPromptOnly(t *testing.T) {
	estimator := Estimator{
		Name:       "fixture_estimator",
		Version:    "2",
		PolicyHash: "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
		EstimateTokens: func(text string) int {
			return len([]rune(text))
		},
	}
	estimated := ProjectUsage(llm.Usage{
		PromptTokens:          123,
		CompletionTokens:      7,
		TotalTokens:           130,
		PromptTokensSource:    llm.PromptTokensSourceEstimate,
		PromptTokensSemantics: llm.PromptTokensSemanticsTotalInputV1,
	}, estimator)
	if estimated.Estimator == nil || estimated.Estimator.Name != estimator.Name ||
		estimated.Estimator.Version != estimator.Version || estimated.Estimator.PolicyHash != estimator.PolicyHash {
		t.Fatalf("estimated usage estimator = %#v, want pinned identity", estimated.Estimator)
	}

	provider := ProjectUsage(*llm.NewProviderUsage(50, 5, 55), estimator)
	if provider.Estimator != nil {
		t.Fatalf("provider usage inherited local estimator: %#v", provider.Estimator)
	}

	invalid := estimator
	invalid.PolicyHash = "not-a-policy-hash"
	unknown := ProjectUsage(llm.Usage{
		PromptTokens:          25,
		TotalTokens:           25,
		PromptTokensSource:    llm.PromptTokensSourceEstimate,
		PromptTokensSemantics: llm.PromptTokensSemanticsTotalInputV1,
	}, invalid)
	if unknown.Estimator != nil {
		t.Fatalf("invalid estimator identity was projected: %#v", unknown.Estimator)
	}
}

func TestProjectCompactionOmitsRawSummaryAndPaths(t *testing.T) {
	usage := llm.NewProviderUsage(900, 20, 920)
	payload := ProjectCompaction(compaction.Result{
		Compacted:          true,
		Trigger:            "usage",
		Watermark:          "summarize",
		Usage:              usage,
		OriginalTokens:     880,
		NewTokens:          120,
		TokenCountSource:   compaction.TokenCountSourceEstimate,
		TiersApplied:       []string{"snip", "prune", "summarize"},
		SnapshotPath:       "/tmp/private-snapshot",
		LedgerPath:         "/tmp/private-ledger",
		Warnings:           []string{"Bearer secret warning"},
		Summary:            "raw compacted transcript must not persist",
		CheckpointID:       "sha256:checkpoint",
		CheckpointMessages: 3,
	}, testEstimator())
	if err := payload.Validate(); err != nil {
		t.Fatalf("payload validation: %v", err)
	}
	if payload.Compaction == nil || !payload.Compaction.Compacted || payload.Compaction.GenerationDelta != 1 {
		t.Fatalf("compaction payload = %#v", payload.Compaction)
	}
	if payload.Compaction.WarningCount != 1 || !payload.Compaction.SummaryPresent {
		t.Fatalf("bounded compaction indicators = %#v", payload.Compaction)
	}
	encoded, _ := json.Marshal(payload)
	for _, forbidden := range []string{"private-snapshot", "private-ledger", "Bearer secret", "raw compacted transcript"} {
		if strings.Contains(string(encoded), forbidden) {
			t.Fatalf("compaction projection leaked %q: %s", forbidden, encoded)
		}
	}
}

func TestEstimatorContractRejectsUndeclaredAndPanickingEstimator(t *testing.T) {
	invalid := Estimator{EstimateTokens: func(string) int { return 99 }}
	payload := ProjectToolResult(ToolResultInput{Tool: "read", Original: "abc", Visible: "a"}, invalid)
	if payload.Measurements.OriginalTokens != nil || payload.Measurements.VisibleTokens != nil || payload.Estimator != nil {
		t.Fatalf("undeclared estimator created comparable tokens: %#v", payload)
	}

	panicking := testEstimator()
	panicking.EstimateTokens = func(string) int { panic("boom") }
	payload = ProjectToolResult(ToolResultInput{Tool: "read", Original: "abc", Visible: "a"}, panicking)
	if payload.Measurements.OriginalTokens != nil || payload.Measurements.VisibleTokens != nil {
		t.Fatalf("panicking estimator created token values: %#v", payload.Measurements)
	}

	tooLarge := Payload{SchemaVersion: SchemaVersion, EventKind: EventKindToolResult, Status: StatusUnknown, ToolKind: strings.Repeat("x", 20000)}
	if _, err := MarshalBounded(tooLarge); err == nil {
		t.Fatal("oversized accounting payload unexpectedly marshaled")
	}

	_ = time.Second // keep the fixture's time import pinned for expiry additions.
}

func TestProjectToolResultCarriesBoundedRepeatedAndArtifactRecoveryDisposition(t *testing.T) {
	payload := ProjectToolResult(ToolResultInput{
		Tool:     "artifact_read",
		Original: "bounded artifact response",
		Visible:  "bounded artifact response",
		Metadata: map[string]any{
			"artifact_recovery_disposition": "expired",
			"evidence_disposition":          "repeated_after_compaction",
			"evidence_fingerprint":          "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
			"evidence_prior_generation":     2,
			"evidence_repeat_count":         3,
			"artifact_error_text":           "must-not-persist",
		},
	}, testEstimator())

	if payload.Artifact == nil || payload.Artifact.RecoveryDisposition != "expired" {
		t.Fatalf("artifact recovery disposition = %#v", payload.Artifact)
	}
	if payload.RepeatedEvidence == nil || payload.RepeatedEvidence.Disposition != "repeated_after_compaction" ||
		payload.RepeatedEvidence.PriorGeneration == nil || *payload.RepeatedEvidence.PriorGeneration != 2 ||
		payload.RepeatedEvidence.RepeatCount == nil || *payload.RepeatedEvidence.RepeatCount != 3 {
		t.Fatalf("repeated evidence = %#v", payload.RepeatedEvidence)
	}
	encoded, err := MarshalBounded(payload)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(encoded), "must-not-persist") || strings.Contains(string(encoded), "artifact_error_text") {
		t.Fatalf("projector leaked arbitrary recovery source: %s", encoded)
	}
}
