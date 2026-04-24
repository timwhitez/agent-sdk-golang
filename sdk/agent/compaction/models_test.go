package compaction

import (
	"strings"
	"testing"
)

func TestDefaultSummaryPromptIncludesToolStateCrossReferenceGuidance(t *testing.T) {
	required := []string{
		"Cross-reference tool results in Key Findings.",
		"\"Used git status (see Recent Tool Results) to confirm...\"",
		"\"The error in line X (from file read) indicates...\"",
	}
	for _, needle := range required {
		if !strings.Contains(DefaultSummaryPrompt, needle) {
			t.Fatalf("expected DefaultSummaryPrompt to include %q", needle)
		}
	}
}

func TestDefaultSummaryPromptUsesTighterWordTarget(t *testing.T) {
	if !strings.Contains(DefaultSummaryPrompt, "Target 300-700 words.") {
		t.Fatal("expected DefaultSummaryPrompt to require 300-700 words")
	}
	if strings.Contains(DefaultSummaryPrompt, "500–1000 words") || strings.Contains(DefaultSummaryPrompt, "500-1000 words") {
		t.Fatal("expected previous 500-1000 word target to be removed")
	}
}

func TestDefaultSummaryPromptIncludesFailureFallbackAndExactValueExamples(t *testing.T) {
	required := []string{
		"If unable to meaningfully summarize, respond with:",
		"<summary>UNABLE_TO_SUMMARIZE: [brief reason]</summary>",
		"File paths: /mnt/c/Users/.../file.go (not \"the file\")",
		"Error codes: HTTP 429, exit code 127 (not \"error\")",
		"Versions: v1.2.3, Python 3.10.5 (not \"latest\")",
		"Command lines: git commit -m \"msg\" (not \"git commit\")",
	}
	for _, needle := range required {
		if !strings.Contains(DefaultSummaryPrompt, needle) {
			t.Fatalf("expected DefaultSummaryPrompt to include %q", needle)
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
