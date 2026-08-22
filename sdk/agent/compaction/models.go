package compaction

import (
	"log"
	"regexp"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/agent/messageorigin"
	"github.com/timwhitez/agent-sdk-golang/sdk/artifact"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

const (
	DefaultContextWindow          = 128_000
	DefaultThresholdRatio         = 0.85
	DefaultSnipThresholdRatio     = 0.70
	DefaultPruneThresholdRatio    = 0.80
	DefaultKeepRecentUserMessages = 3
	DefaultToolSnapshotMaxEntries = 6
	DefaultToolSnapshotMaxChars   = 2000
	DefaultCheckpointMaxTokens    = 2000
	DefaultSummaryTargetTokens    = 1600
	DefaultCompactionRetries      = 1
	compactionSummaryMessageName  = messageorigin.CompactionSummaryName
	// TokenCountSourceEstimate identifies before/after counts produced by the
	// same host-injected local estimator. Provider usage remains available in
	// Result.Usage and must not be mixed into this comparable pair.
	TokenCountSourceEstimate = "estimate"
)

const (
	DefaultCompactionTimeout             = 300 * time.Second
	DefaultCompactionRetryBackoff        = 250 * time.Millisecond
	DefaultMinSummaryCharsForToolContext = 120
)

// DefaultSummaryPrefix mirrors Codex's summary prefix.
const DefaultSummaryPrefix = `Another language model started to solve this problem and produced a summary of its thinking process. You also have access to the state of the tools that were used by that language model. Use this to build on the work that has already been done and avoid duplicating work. Here is the summary produced by the other language model, use the information in this summary to assist with your own analysis:`

// DefaultSummaryPrompt defines the evidence-first handoff
// contract. Mandatory authority and the untrusted-material boundary are added
// separately by the request builder as a system message.
const DefaultSummaryPrompt = `Write an operational checkpoint for the next coding-agent turn.

Trust order:
1. Successful tool results and current filesystem/repository state.
2. Explicit user messages and user-approved decisions.
3. Assistant statements only when corroborated; otherwise label them UNVERIFIED.

Required sections, in this exact order. Inside <summary>, emit every title as
the exact level-2 Markdown heading shown below. Do not number, bold, rename,
translate, or add a trailing colon to these heading lines. Put a non-empty body
immediately below every heading.

## Current Objective and Latest User Request
## Authoritative Current State
## Completed Work
## In-Progress and Remaining Work
## Exact External State
## Errors, Failed Attempts, and Successful Recovery
## Verification Already Run and Still Required
## Conflicts, Uncertainty, and Facts That Must Be Re-read

Rules:
- Do not infer missing facts. Write UNKNOWN when authoritative state is unavailable.
- Write UNVERIFIED for assistant claims that lack successful tool/filesystem evidence.
- Do not claim a file changed unless successful write/edit/delete/diff/status evidence proves it.
- Include only analyzed files needed for remaining work; do not list every read or every analyzed file.
- Preserve exact paths, identifiers, commands, versions, error strings, status codes, and hashes when supplied.
- Preserve the original objective and newest user instruction when available.
- Do not convert internal recovery/reminder messages into user requirements.
- Prefer concise completeness within the adaptive token budget supplied by the host; do not target a fixed word count.
- Return exactly one <summary>...</summary> block and no text outside it.`

type Config struct {
	Enabled bool
	// Warningf receives non-fatal compaction diagnostics. Empty uses log.Printf.
	Warningf func(format string, args ...any)
	// ContextWindow is the raw model context window. Tier 1/2/3 watermarks and
	// the hard overflow boundary are computed from ContextWindow minus
	// ReserveOutputTokens. If <=0, DefaultContextWindow is used.
	ContextWindow  int
	ThresholdRatio float64
	// SummaryPrompt accepts either a static string or a model-aware resolver function:
	//   - string
	//   - func(modelID string) string
	// Empty/invalid values fall back to DefaultSummaryPrompt.
	SummaryPrompt any
	// ReserveOutputTokens keeps room for the next provider completion. It is
	// subtracted before every compaction watermark is calculated.
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
	// PruneThresholdRatio is the local Tier 2 watermark for tool-result and assistant-text pruning.
	// If <=0, DefaultPruneThresholdRatio is used. It is capped below ThresholdRatio.
	PruneThresholdRatio float64
	// ProtectedRecentMessages provides a trailing N-message fallback in addition
	// to complete-current-turn and unfinished-tool-topology protection. If <=0,
	// DefaultKeepRecentUserMessages is used.
	ProtectedRecentMessages int
	// ProtectedRecentTokens additionally protects a token-budgeted tail from
	// local reducers. It composes with the complete current-turn and open-tool
	// topology boundaries. Values <= 0 disable this optional extra budget.
	ProtectedRecentTokens int
	// EnableUserCodeMicrocompact allows local reducers to compact old markdown
	// fenced code blocks inside user messages outside the protected zone.
	EnableUserCodeMicrocompact bool
	// SessionID and LedgerStore allow compaction replacements to be stable across turns/resume.
	SessionID   string
	LedgerStore LedgerStore
	LedgerPath  string
	// ToolArtifactWriter persists full tool outputs before local replacement.
	// It is the legacy path-only compatibility writer. When any canonical
	// artifact field below is configured, local replacement requires the whole
	// canonical binding and does not promote this writer's path.
	ToolArtifactWriter ArtifactWriter
	// ArtifactOwnerProvider, ArtifactSink, ArtifactResolver,
	// ArtifactResolverCapability, and ArtifactEnvelopeCodec form one canonical
	// execution binding. Agent hosts should install the same binding used at the
	// tool-result boundary. A partial binding fails closed with a visible warning.
	ArtifactOwnerProvider      ArtifactOwnerProvider
	ArtifactSink               artifact.Sink
	ArtifactResolver           artifact.Resolver
	ArtifactResolverCapability artifact.ResolverCapability
	ArtifactEnvelopeCodec      artifact.EnvelopeCodec
	// SummarySourceWriter optionally persists the source history used
	// to produce a summary. SourceSnapshot is recorded only after this writer
	// returns a non-empty durable path.
	SummarySourceWriter ArtifactWriter
	// CheckpointWriter durably records the final provider history and telemetry
	// before an Agent replaces its in-memory history. Hosts own the storage
	// implementation; the SDK owns commit ordering and failure semantics.
	CheckpointWriter CompactionCheckpointWriter
	// CheckpointProvider supplies bounded host-owned task, workspace, evidence,
	// error, and validation state for summary material. The SDK defines the
	// portable schema; hosts remain responsible for collecting repository and
	// runtime state.
	CheckpointProvider CheckpointProvider
	// CheckpointMaxTokens bounds the rendered host checkpoint section. Values
	// <= 0 use DefaultCheckpointMaxTokens.
	CheckpointMaxTokens int
	// SummaryTargetTokens is the host-selected adaptive output budget described
	// to the summary model. Values <= 0 use DefaultSummaryTargetTokens.
	SummaryTargetTokens int
	// TokenEstimator estimates prompt tokens for a text fragment. When nil, a
	// naive (len+3)/4 heuristic is used. Hosts (e.g. Goode) inject a detailed
	// estimator so tier eligibility and reported token counts match the same
	// estimator used for prompt-budget decisions outside the compaction package.
	TokenEstimator func(text string) int
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
		CheckpointMaxTokens:           DefaultCheckpointMaxTokens,
		SummaryTargetTokens:           DefaultSummaryTargetTokens,
		SnipThresholdRatio:            DefaultSnipThresholdRatio,
		PruneThresholdRatio:           DefaultPruneThresholdRatio,
		ProtectedRecentMessages:       DefaultKeepRecentUserMessages,
	}
}

type Result struct {
	Compacted          bool       `json:"compacted"`
	Trigger            string     `json:"trigger,omitempty"`
	Watermark          string     `json:"watermark,omitempty"`
	Usage              *llm.Usage `json:"usage,omitempty"`
	OriginalTokens     int        `json:"original_tokens,omitempty"`
	NewTokens          int        `json:"new_tokens,omitempty"`
	TokenCountSource   string     `json:"token_count_source,omitempty"`
	TiersApplied       []string   `json:"tiers_applied,omitempty"`
	SnapshotPath       string     `json:"snapshot_path,omitempty"`
	LedgerPath         string     `json:"ledger_path,omitempty"`
	Warnings           []string   `json:"warnings,omitempty"`
	Summary            string     `json:"summary,omitempty"`
	CheckpointID       string     `json:"checkpoint_id,omitempty"`
	CheckpointMessages int        `json:"checkpoint_messages,omitempty"`

	// Ledger updates are deferred when a runtime checkpoint writer is configured.
	// The Agent commits this transaction immediately before the replayable
	// checkpoint, then rolls it back if checkpoint persistence fails.
	pendingLedger  *Ledger
	previousLedger *Ledger
}

// PipelineRequest describes one canonical compaction decision. Runtime entry
// points provide a trigger and usage/estimate; the service owns tier ordering,
// re-estimation, summary escalation, and telemetry merging.
type PipelineRequest struct {
	Trigger          string
	Usage            *llm.Usage
	EstimatedTokens  int
	AdditionalTokens int
	TargetWatermark  string
	AllowSummary     bool
	ForceSummary     bool
}

var (
	summaryTagRe           = regexp.MustCompile(`(?s)<summary>(.*?)</summary>`)
	structuredSummaryTagRe = regexp.MustCompile(`(?s)<compaction_summary>(.*?)</compaction_summary>`)
)

type summaryCapture struct {
	start   int
	end     int
	content string
}

func ExtractSummary(text string) string {
	return extractSummaryWithWarning(text, log.Printf)
}

func extractSummaryWithWarning(text string, warnf func(string, ...any)) string {
	if warnf == nil {
		warnf = func(string, ...any) {}
	}
	captures := collectSummaryCaptures(text)
	if len(captures) == 0 {
		warnf("compaction: summary extraction failed: missing <summary> or <compaction_summary> tags")
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
		warnf("compaction: summary extraction failed: empty summary block")
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
			end:     idx[1],
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
	return resolveSummaryPromptWithWarning(v, log.Printf)
}

func resolveSummaryPromptWithWarning(v any, warnf func(string, ...any)) SummaryPromptFunc {
	if warnf == nil {
		warnf = func(string, ...any) {}
	}
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
		warnf("compaction: unsupported SummaryPrompt type %T; using default prompt", v)
		return fallback
	}
}
