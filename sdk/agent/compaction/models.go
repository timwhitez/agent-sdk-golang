package compaction

import (
	"log"
	"regexp"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

const (
	DefaultContextWindow          = 128_000
	DefaultThresholdRatio         = 0.85
	DefaultSnipThresholdRatio     = 0.70
	DefaultKeepRecentUserMessages = 3
	DefaultToolSnapshotMaxEntries = 6
	DefaultToolSnapshotMaxChars   = 2000
	DefaultCompactionRetries      = 1
	compactionSummaryMessageName  = "compaction_summary"
)

const (
	DefaultCompactionTimeout             = 30 * time.Second
	DefaultCompactionRetryBackoff        = 250 * time.Millisecond
	DefaultMinSummaryCharsForToolContext = 120
)

// DefaultSummaryPrefix mirrors Codex's summary prefix.
const DefaultSummaryPrefix = `Another language model started to solve this problem and produced a summary of its thinking process. You also have access to the state of the tools that were used by that language model. Use this to build on the work that has already been done and avoid duplicating work. Here is the summary produced by the other language model, use the information in this summary to assist with your own analysis:`

// DefaultSummaryPrompt is a coding-agent-optimized prompt for context compaction.
// Designed based on Anthropic context engineering docs, Factory.ai evaluation findings,
// Codex/Manus patterns, and common failure modes (Claude Code issue #14160).
//
// Key design decisions:
//   - "Modified Files" is a dedicated section because Factory.ai found file paths are
//     the #1 most commonly lost information type in summaries.
//   - "Errors & Failed Approaches" is separated from general findings because Jason Liu
//     and Anthropic both show this prevents the model from re-attempting failed strategies.
//   - "User Constraints" are elevated to section 1 because issue #14160 shows users
//     complain that preferences are lost after compaction.
//   - "DO NOT re-derive" instruction prevents decision regression post-compaction.
//   - "Verification" section preserves test commands which are commonly lost.
const DefaultSummaryPrompt = `You are performing a context checkpoint. Write a structured handoff summary that will REPLACE the full conversation history. Your future self will have NO access to prior messages — only this summary plus the most recent user messages and tool state.

## Required Sections

### 1. Task & User Constraints
- The user's original request and explicit success criteria
- User preferences, style requirements, or constraints they specified
- Commitments or promises made to the user that must be honored
- Response language or formatting requirements

### 2. Modified Files
List EVERY file created, modified, deleted, or analyzed — use EXACT absolute paths:
- ` + "`/path/to/file.go`" + ` — what was changed and why
- Include function names or line ranges when relevant for targeted re-reading
- Mark files as [created], [modified], [deleted], or [analyzed]

Cross-reference tool results in Key Findings. Example:
- "Used git status (see Recent Tool Results) to confirm..."
- "The error in line X (from file read) indicates..."

### 3. Completed Work
- Specific actions taken and their outcomes
- Commands executed and key results (preserve exact command strings if non-trivial)
- Current state: what exists now vs. what existed before

### 4. Key Decisions
- Technical decisions made and their rationale — state as FACTS, do not re-derive
- Architecture or design patterns adopted
- Project conventions or constraints discovered

### 5. Errors & Failed Approaches
- Approaches tried that DID NOT WORK — what was attempted and why it failed
- This section PREVENTS your future self from re-attempting the same dead ends
- Errors encountered → root cause → resolution applied
- Edge cases or gotchas discovered

### 6. Remaining Work
- Concrete next actions in priority order
- Blockers or open questions to resolve
- Incomplete operations that need finishing

### 7. Verification
- How to verify completed work (test commands, build commands, expected outputs)
- Known test failures, regressions, or warnings to watch for
- Working directory and environment details needed to run commands

## Rules
- Target 300-700 words. Every sentence must earn its place.
- Preserve VERBATIM: file paths, function/variable names, error messages, command invocations, and specific values.
- Critical to preserve exactly:
  - File paths: /mnt/c/Users/.../file.go (not "the file")
  - Error codes: HTTP 429, exit code 127 (not "error")
  - Versions: v1.2.3, Python 3.10.5 (not "latest")
  - Command lines: git commit -m "msg" (not "git commit")
- DO NOT reproduce file contents or full tool outputs — they can be re-read from disk.
- DO NOT re-derive or re-evaluate decisions already made — state them as settled facts.
- DO NOT explain what compaction is or add meta-commentary about the summarization process.
- Write as an operational briefing for your future self, not a narrative for a human reader.
- If unable to meaningfully summarize, respond with:
  <summary>UNABLE_TO_SUMMARIZE: [brief reason]</summary>

Wrap your summary in <summary></summary> tags.`

type Config struct {
	Enabled bool
	// ContextWindow is the model context window used for computing the compaction threshold.
	// If <=0, DefaultContextWindow is used.
	ContextWindow  int
	ThresholdRatio float64
	// SummaryPrompt accepts either a static string or a model-aware resolver function:
	//   - string
	//   - func(modelID string) string
	// Empty/invalid values fall back to DefaultSummaryPrompt.
	SummaryPrompt          any
	ReserveOutputTokens    int
	KeepRecentUserMessages int
	CompactionTimeout      time.Duration
	CompactionRetries      int
	CompactionRetryBackoff time.Duration
	// ProtectedTools pins important tool results in tool-context snapshots.
	// Names are matched case-insensitively.
	ProtectedTools []string
	// MinSummaryCharsForToolContext controls when recent tool context is appended.
	// Smaller summaries skip tool-context capture to avoid extra work for low-value compactions.
	MinSummaryCharsForToolContext int
	// ToolSnapshotMaxEntries limits how many recent tool-result entries are captured
	// in compaction summaries. Values <= 0 use DefaultToolSnapshotMaxEntries.
	ToolSnapshotMaxEntries int
	// ToolSnapshotMaxChars caps the total byte length of the tool snapshot section.
	// Values <= 0 use DefaultToolSnapshotMaxChars.
	ToolSnapshotMaxChars int
	// SnipThresholdRatio is the local Tier 1 watermark for tool-result snipping.
	// If <=0, DefaultSnipThresholdRatio is used. It is capped below ThresholdRatio.
	SnipThresholdRatio float64
	// ProtectedRecentMessages excludes the last N messages from local reducers.
	// If <=0, DefaultKeepRecentUserMessages is used as a conservative message-zone fallback.
	ProtectedRecentMessages int
	// SessionID and LedgerStore allow compaction replacements to be stable across turns/resume.
	SessionID   string
	LedgerStore LedgerStore
	LedgerPath  string
	// ToolArtifactWriter persists full tool outputs before local replacement.
	ToolArtifactWriter ArtifactWriter
}

// SummaryPromptFunc resolves a compaction summary prompt from the active model ID.
type SummaryPromptFunc func(modelID string) string

func DefaultConfig() Config {
	return Config{
		Enabled:                       true,
		ContextWindow:                 DefaultContextWindow,
		ThresholdRatio:                DefaultThresholdRatio,
		SummaryPrompt:                 DefaultSummaryPrompt,
		KeepRecentUserMessages:        DefaultKeepRecentUserMessages,
		CompactionTimeout:             DefaultCompactionTimeout,
		CompactionRetries:             DefaultCompactionRetries,
		CompactionRetryBackoff:        DefaultCompactionRetryBackoff,
		MinSummaryCharsForToolContext: DefaultMinSummaryCharsForToolContext,
		ToolSnapshotMaxEntries:        DefaultToolSnapshotMaxEntries,
		ToolSnapshotMaxChars:          DefaultToolSnapshotMaxChars,
		SnipThresholdRatio:            DefaultSnipThresholdRatio,
		ProtectedRecentMessages:       DefaultKeepRecentUserMessages,
	}
}

type Result struct {
	Compacted      bool       `json:"compacted"`
	Trigger        string     `json:"trigger,omitempty"`
	Watermark      string     `json:"watermark,omitempty"`
	Usage          *llm.Usage `json:"usage,omitempty"`
	OriginalTokens int        `json:"original_tokens,omitempty"`
	NewTokens      int        `json:"new_tokens,omitempty"`
	TiersApplied   []string   `json:"tiers_applied,omitempty"`
	SnapshotPath   string     `json:"snapshot_path,omitempty"`
	LedgerPath     string     `json:"ledger_path,omitempty"`
	Warnings       []string   `json:"warnings,omitempty"`
	Summary        string     `json:"summary,omitempty"`
}

var (
	summaryTagRe           = regexp.MustCompile(`(?s)<summary>(.*?)</summary>`)
	structuredSummaryTagRe = regexp.MustCompile(`(?s)<compaction_summary>(.*?)</compaction_summary>`)
)

type summaryCapture struct {
	start   int
	content string
}

func ExtractSummary(text string) string {
	captures := collectSummaryCaptures(text)
	if len(captures) == 0 {
		log.Printf("compaction: summary extraction failed: missing <summary> or <compaction_summary> tags")
		return ""
	}
	last := captures[0]
	for i := 1; i < len(captures); i++ {
		if captures[i].start > last.start {
			last = captures[i]
		}
	}
	summary := stringsTrim(last.content)
	if summary == "" {
		log.Printf("compaction: summary extraction failed: empty summary block")
		return ""
	}
	return summary
}

func collectSummaryCaptures(text string) []summaryCapture {
	out := make([]summaryCapture, 0, 4)
	out = appendSummaryCaptures(out, text, summaryTagRe)
	out = appendSummaryCaptures(out, text, structuredSummaryTagRe)
	return out
}

func appendSummaryCaptures(dst []summaryCapture, text string, re *regexp.Regexp) []summaryCapture {
	matches := re.FindAllStringSubmatchIndex(text, -1)
	for _, idx := range matches {
		if len(idx) < 4 {
			continue
		}
		dst = append(dst, summaryCapture{
			start:   idx[0],
			content: text[idx[2]:idx[3]],
		})
	}
	return dst
}

func stringsTrim(s string) string {
	// avoid importing strings in multiple files; tiny helper
	for len(s) > 0 {
		r := s[0]
		if r == ' ' || r == '\n' || r == '\t' || r == '\r' {
			s = s[1:]
			continue
		}
		break
	}
	for len(s) > 0 {
		r := s[len(s)-1]
		if r == ' ' || r == '\n' || r == '\t' || r == '\r' {
			s = s[:len(s)-1]
			continue
		}
		break
	}
	return s
}

func resolveSummaryPrompt(v any) SummaryPromptFunc {
	fallback := func(string) string { return DefaultSummaryPrompt }
	switch p := v.(type) {
	case nil:
		return fallback
	case string:
		prompt := stringsTrim(p)
		if prompt == "" {
			return fallback
		}
		return func(string) string { return prompt }
	case func(string) string:
		if p == nil {
			return fallback
		}
		return func(modelID string) string {
			prompt := stringsTrim(p(modelID))
			if prompt == "" {
				return DefaultSummaryPrompt
			}
			return prompt
		}
	case SummaryPromptFunc:
		if p == nil {
			return fallback
		}
		return func(modelID string) string {
			prompt := stringsTrim(p(modelID))
			if prompt == "" {
				return DefaultSummaryPrompt
			}
			return prompt
		}
	default:
		log.Printf("compaction: unsupported SummaryPrompt type %T; using default prompt", v)
		return fallback
	}
}
