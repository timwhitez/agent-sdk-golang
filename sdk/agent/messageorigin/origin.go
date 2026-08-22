// Package messageorigin defines stable origin markers for framework-authored
// user-role messages and the real-user classification contract shared by the
// agent loop and compaction reducers.
package messageorigin

import (
	"regexp"
	"strings"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

const (
	SDKInternalNamePrefix   = "sdk_internal_"
	HostInternalNamePrefix  = "goode_internal_"
	CompactionSummaryName   = "compaction_summary"
	GoodeMemoryName         = "goode_memory"
	WideResearchContextName = "wide_research"
)

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
	RequireDoneReminderText       = "Task completion must use the done tool. If work remains, call the next relevant tool now and continue. If the task is complete, call done with a concise completion message. Do not respond with text-only planning or completion claims."
	legacyRequireDoneReminderText = "Task completion must use the done tool. If the task is complete, call done with a concise completion message. Do not end with text-only completion claims."
	DefaultLoopGuardText          = "You are repeating the same tool call with identical arguments. Stop repeating, reuse prior results, adjust arguments, or call done if the task is complete."
	EarlyStopReminderText         = "You already used tools in this run. Before stopping, verify the task is complete and call done with a concise completion message."
	StreamIdleRecoveryText        = "The previous response stream stalled before completion. Continue from the current conversation state. Do not repeat completed tool calls unless needed. If you were mid-analysis or mid-sentence, continue exactly where you left off. If enough information is already available, complete the task."

	ResponseTruncatedContinuationText = "Your response was truncated. Please continue exactly where you left off."
	ToolCallContinuationLimitText     = "Your tool-call arguments were repeatedly truncated. Split the work into smaller tool calls and continue."
	InvalidToolCallContinuationText   = "Tool-call arguments are still invalid after continuation. Split the work into smaller tool calls and continue."
	RecycledToolResultRecoveryText    = "You are re-issuing a call whose earlier result was released to save context — re-reading it will not return the content. Change the offset/target or continue from your notes, or call done if the task is complete."
)

var legacyEvidenceRecoveryPattern = regexp.MustCompile(`^No-progress recovery: (?:read|search|list) evidence for .+ \(fingerprint [0-9a-f]{12}\) has already been observed\. Do not repeat covered reads\. Change target/range or action, use existing evidence, or call done if the task is complete\.$`)

var legacyExactInternalUserText = map[string]struct{}{
	RequireDoneReminderText:           {},
	legacyRequireDoneReminderText:     {},
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

func IsInternalMessage(message llm.Message) bool {
	name := strings.TrimSpace(message.Name)
	if isReservedInternalName(name) {
		return true
	}
	return message.Role == llm.RoleUser && name == "" && isLegacyInternalUserText(message.Content.PlainText())
}

func IsRealUserMessage(message llm.Message) bool {
	return message.Role == llm.RoleUser && !message.Destroyed && !IsInternalMessage(message)
}

func isReservedInternalName(name string) bool {
	if name == "" {
		return false
	}
	switch name {
	case CompactionSummaryName, GoodeMemoryName, WideResearchContextName:
		return true
	default:
		return strings.HasPrefix(name, SDKInternalNamePrefix) || strings.HasPrefix(name, HostInternalNamePrefix)
	}
}

func isLegacyInternalUserText(text string) bool {
	text = strings.TrimSpace(text)
	if _, ok := legacyExactInternalUserText[text]; ok {
		return true
	}
	return legacyEvidenceRecoveryPattern.MatchString(text)
}
