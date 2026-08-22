package compaction

import (
	"strings"
	"testing"
)

func TestDefaultSummaryPromptIncludesEvidenceGuidance(t *testing.T) {
	required := []string{
		"Successful tool results and current filesystem/repository state",
		"Preserve exact paths, identifiers, commands, versions, error strings, status codes, and hashes when supplied",
	}
	for _, needle := range required {
		if !strings.Contains(DefaultSummaryPrompt, needle) {
			t.Fatalf("expected DefaultSummaryPrompt to include %q", needle)
		}
	}
}

func TestDefaultSummaryPromptUsesAdaptiveTokenBudget(t *testing.T) {
	if !strings.Contains(DefaultSummaryPrompt, "adaptive token budget supplied by the host") {
		t.Fatal("expected DefaultSummaryPrompt to use the host token budget")
	}
	for _, forbidden := range []string{"300-700 words", "500–1000 words", "500-1000 words"} {
		if strings.Contains(DefaultSummaryPrompt, forbidden) {
			t.Fatalf("fixed word target remains in prompt: %q", forbidden)
		}
	}
}

func TestDefaultSummaryPromptRequiresExactSingleStructuredBlock(t *testing.T) {
	required := []string{
		"Required sections, in this exact order",
		"exact level-2 Markdown heading",
		"Current Objective and Latest User Request",
		"Verification Already Run and Still Required",
		"Return exactly one <summary>...</summary> block and no text outside it",
	}
	for _, needle := range required {
		if !strings.Contains(DefaultSummaryPrompt, needle) {
			t.Fatalf("expected DefaultSummaryPrompt to include %q", needle)
		}
	}
}

func TestDefaultSummaryPromptMatchesValidatorHeadingSyntax(t *testing.T) {
	for _, section := range requiredSummarySections {
		want := "\n## " + section + "\n"
		if !strings.Contains(DefaultSummaryPrompt, want) {
			t.Fatalf("DefaultSummaryPrompt is missing validator-compatible heading %q", want)
		}
	}
}

func TestCompactionPromptDoesNotRequireEveryAnalyzedFile(t *testing.T) {
	if strings.Contains(strings.ToLower(DefaultSummaryPrompt), "list every file") {
		t.Fatalf("summary prompt still requires unavailable every-file material:\n%s", DefaultSummaryPrompt)
	}
	for _, want := range []string{"successful write/edit/delete/diff/status evidence proves it", "do not list every read or every analyzed file"} {
		if !strings.Contains(DefaultSummaryPrompt, want) {
			t.Fatalf("summary prompt is missing evidence-first file guidance %q", want)
		}
	}
}

func TestCompactionPromptUsesEvidenceTrustOrder(t *testing.T) {
	ordered := []string{
		"Successful tool results and current filesystem/repository state",
		"Explicit user messages and user-approved decisions",
		"Assistant statements only when corroborated",
	}
	last := -1
	for _, item := range ordered {
		idx := strings.Index(DefaultSummaryPrompt, item)
		if idx < 0 {
			t.Fatalf("summary prompt is missing trust-order item %q", item)
		}
		if idx <= last {
			t.Fatalf("trust-order item %q is out of order", item)
		}
		last = idx
	}
}

func TestCompactionPromptAllowsUnknownAndUnverified(t *testing.T) {
	for _, want := range []string{"UNKNOWN", "UNVERIFIED", "Do not infer missing facts"} {
		if !strings.Contains(DefaultSummaryPrompt, want) {
			t.Fatalf("summary prompt is missing uncertainty rule %q", want)
		}
	}
}

func TestResolveSummaryPrompt_UsesModelAwareFunction(t *testing.T) {
	fn := resolveSummaryPrompt(func(modelID string) string {
		if modelID == "small" {
			return "small prompt"
		}
		return "large prompt"
	})
	if got := fn("small"); got != "small prompt" {
		t.Fatalf("expected model-specific prompt, got %q", got)
	}
	if got := fn("large"); got != "large prompt" {
		t.Fatalf("expected fallback prompt, got %q", got)
	}
}

func TestResolveSummaryPrompt_UnsupportedTypeFallsBack(t *testing.T) {
	fn := resolveSummaryPrompt(123)
	if got := fn("any"); got != DefaultSummaryPrompt {
		t.Fatalf("expected default prompt fallback, got %q", got)
	}
}
