// Package messageorigin defines stable origin markers for framework-authored
// user-role messages and the real-user classification contract shared by the
// agent loop and compaction reducers.
package messageorigin

import (
	"fmt"
	"regexp"
	"strings"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

const (
	SDKInternalNamePrefix   = "sdk_internal_"
	CompactionSummaryName   = "compaction_summary"
	WideResearchContextName = "wide_research"
)

// Host namespaces scope the host-owned identifiers the SDK writes and
// recognizes: "<ns>_internal_*" and "<ns>_memory" message names, the
// "<ns>.compaction.material.v1" schema and the
// "<!-- <ns>-compaction-checkpoint-id=" summary marker.
const (
	// DefaultHostNamespace is used when a host configures none.
	DefaultHostNamespace = "agent"
	// LegacyHostNamespace is the namespace earlier SDK releases hard-coded.
	// Its forms stay recognized under every namespace so persisted
	// histories keep loading and classifying exactly as before.
	LegacyHostNamespace          = "goode"
	LegacyHostInternalNamePrefix = LegacyHostNamespace + "_internal_"
	LegacyHostMemoryName         = LegacyHostNamespace + "_memory"

	// Deprecated: use LegacyHostInternalNamePrefix.
	HostInternalNamePrefix = LegacyHostInternalNamePrefix
	// Deprecated: use LegacyHostMemoryName.
	GoodeMemoryName = LegacyHostMemoryName

	maxHostNamespaceLen = 32
)

// ValidateHostNamespace accepts 1-32 bytes of [a-z0-9] starting with a letter.
// The namespace is embedded in names and an HTML-comment marker that are
// parsed later, so the separators "_", "-", "." and anything that could close
// or extend the marker are excluded.
func ValidateHostNamespace(namespace string) error {
	if namespace == "" || len(namespace) > maxHostNamespaceLen {
		return fmt.Errorf("host namespace %q must be 1-%d characters", namespace, maxHostNamespaceLen)
	}
	for i := 0; i < len(namespace); i++ {
		c := namespace[i]
		if (c < 'a' || c > 'z') && (i == 0 || c < '0' || c > '9') {
			return fmt.Errorf("host namespace %q must match [a-z][a-z0-9]*", namespace)
		}
	}
	return nil
}

// ResolveHostNamespace maps an empty namespace to DefaultHostNamespace. An
// invalid namespace resolves to "" so that callers recognize only the SDK and
// legacy forms and never build an identifier from it.
func ResolveHostNamespace(namespace string) string {
	if namespace == "" {
		return DefaultHostNamespace
	}
	if ValidateHostNamespace(namespace) != nil {
		return ""
	}
	return namespace
}

type Kind string

const (
	KindRequireDone           Kind = "require_done"
	KindMaxTokensContinuation Kind = "max_tokens_continuation"
	KindStreamIdleRecovery    Kind = "stream_idle_recovery"
	KindToolCallContinuation  Kind = "tool_call_continuation"
	KindEarlyStop             Kind = "early_stop"
	KindLoopGuard             Kind = "loop_guard"
	KindEvidenceRecovery      Kind = "evidence_recovery"
)

const (
	RequireDoneReminderText = "Task completion must use the done tool. If work remains, call the next relevant tool now and continue. If the task is complete, call done with your final answer for the user: the actual result (the answer, summary, findings, or what changed and how it was verified), not a list of the steps you took. Do not respond with text-only planning or completion claims."
	// Earlier reminder texts stay recognized for persisted histories without
	// an internal message name.
	legacyRequireDoneReminderText   = "Task completion must use the done tool. If the task is complete, call done with a concise completion message. Do not end with text-only completion claims."
	legacyRequireDoneReminderTextV2 = "Task completion must use the done tool. If work remains, call the next relevant tool now and continue. If the task is complete, call done with a concise completion message. Do not respond with text-only planning or completion claims."
	legacyEarlyStopReminderText     = "You already used tools in this run. Before stopping, verify the task is complete and call done with a concise completion message."
	DefaultLoopGuardText            = "You are repeating the same tool call with identical arguments. Stop repeating, reuse prior results, adjust arguments, or call done if the task is complete."
	EarlyStopReminderText           = "You already used tools in this run. Before stopping, verify the task is complete and call done with your final answer for the user: the actual result, not a list of the steps you took."
	StreamIdleRecoveryText          = "The previous response stream stalled before completion. Continue from the current conversation state. Do not repeat completed tool calls unless needed. If you were mid-analysis or mid-sentence, continue exactly where you left off. If enough information is already available, complete the task."

	ResponseTruncatedContinuationText = "Your response was truncated. Please continue exactly where you left off."
	ToolCallContinuationLimitText     = "Your tool-call arguments were repeatedly truncated. Split the work into smaller tool calls and continue."
	InvalidToolCallContinuationText   = "Tool-call arguments are still invalid after continuation. Split the work into smaller tool calls and continue."
	RecycledToolResultRecoveryText    = "You are re-issuing a call whose earlier result was released to save context — re-reading it will not return the content. Change the offset/target or continue from your notes, or call done if the task is complete."
)

var legacyEvidenceRecoveryPattern = regexp.MustCompile(`^No-progress recovery: (?:read|search|list) evidence for .+ \(fingerprint [0-9a-f]{12}\) has already been observed\. Do not repeat covered reads\. Change target/range or action, use existing evidence, or call done if the task is complete\.$`)

var legacyExactInternalUserText = map[string]struct{}{
	RequireDoneReminderText:           {},
	legacyRequireDoneReminderText:     {},
	legacyRequireDoneReminderTextV2:   {},
	legacyEarlyStopReminderText:       {},
	DefaultLoopGuardText:              {},
	EarlyStopReminderText:             {},
	StreamIdleRecoveryText:            {},
	ResponseTruncatedContinuationText: {},
	ToolCallContinuationLimitText:     {},
	InvalidToolCallContinuationText:   {},
	RecycledToolResultRecoveryText:    {},
	"agent: plan":                     {},
	"agent: build":                    {},
	"[tool history omitted because its matching assistant tool call was trimmed]": {},
}

func Name(kind Kind) string {
	return SDKInternalNamePrefix + string(kind)
}

func NewInternalUserMessage(kind Kind, text string) llm.Message {
	return llm.Message{
		Role:    llm.RoleUser,
		Name:    Name(kind),
		Content: llm.TextContent(text),
	}
}

// IsInternalMessage classifies under DefaultHostNamespace; see IsInternalMessageIn.
func IsInternalMessage(message llm.Message) bool {
	return IsInternalMessageIn("", message)
}

// IsRealUserMessage classifies under DefaultHostNamespace; see IsRealUserMessageIn.
func IsRealUserMessage(message llm.Message) bool {
	return IsRealUserMessageIn("", message)
}

// IsInternalMessageIn reports whether a message is framework- or host-authored.
// Host-owned names are recognized for the given namespace (empty means
// DefaultHostNamespace) and for LegacyHostNamespace; other namespaces are not.
func IsInternalMessageIn(namespace string, message llm.Message) bool {
	name := strings.TrimSpace(message.Name)
	if isReservedInternalName(ResolveHostNamespace(namespace), name) {
		return true
	}
	return message.Role == llm.RoleUser && name == "" && isLegacyInternalUserText(message.Content.PlainText())
}

// IsRealUserMessageIn reports whether a message is a live user request under
// the given host namespace.
func IsRealUserMessageIn(namespace string, message llm.Message) bool {
	return message.Role == llm.RoleUser && !message.Destroyed && !IsInternalMessageIn(namespace, message)
}

func isReservedInternalName(namespace, name string) bool {
	if name == "" {
		return false
	}
	switch name {
	case CompactionSummaryName, LegacyHostMemoryName, WideResearchContextName:
		return true
	}
	if strings.HasPrefix(name, SDKInternalNamePrefix) || strings.HasPrefix(name, LegacyHostInternalNamePrefix) {
		return true
	}
	return namespace != "" && (name == namespace+"_memory" || strings.HasPrefix(name, namespace+"_internal_"))
}

func isLegacyInternalUserText(text string) bool {
	text = strings.TrimSpace(text)
	if _, ok := legacyExactInternalUserText[text]; ok {
		return true
	}
	return legacyEvidenceRecoveryPattern.MatchString(text)
}
