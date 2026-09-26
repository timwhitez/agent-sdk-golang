package agent

import (
	"bytes"
	"encoding/json"
	"errors"
	"sync/atomic"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

// ErrActiveHistoryMutation means a replacement could invalidate the active
// query's conversation or an independent manual compaction publication.
// No mutation was applied.
var ErrActiveHistoryMutation = errors.New("agent: history replacement would invalidate an active query or manual compaction")

var errAssistantHistoryChanged = errors.New("agent: current assistant history changed before continuation finalization")

// ReplaceHistoryChecked replaces idle history, or applies a structure-preserving
// system-context update during a query. Non-system messages must retain their
// full JSON identity and any open tool/continuation tail must remain unchanged.
// System updates affect the next logical request, not its captured predecessor.
// An independent manual compaction rejects all replacements until publication
// completes, including System-only updates and callback attempts.
func (a *Agent) ReplaceHistoryChecked(messages []llm.Message) error {
	_, err := a.ReplaceHistoryCheckedRevision(messages)
	return err
}

// hostPublicationRevisions allocates HistoryPublication revisions for every
// Agent in the process, so one revision never names two publications.
var hostPublicationRevisions atomic.Uint64

// HistoryPublication identifies one successful host history publication.
// It is a process-local counter, not content, a session revision or proof
// that a request was delivered.
type HistoryPublication struct {
	// Revision is non-zero and unique among the publications of every Agent
	// in this process; later publications have larger revisions.
	Revision uint64
	// Replaced is the Revision whose system messages the replaced history
	// still carried without an SDK change to its system messages; zero is
	// unknown (no host publication yet, or the SDK changed them since).
	Replaced uint64
}

// ReplaceHistoryCheckedRevision is ReplaceHistoryChecked that also returns
// the publication it made, read under the same history lock. A Frame built
// afterward reports Revision as EventEnvelope.HostPublicationRevision until
// the host publishes again or the SDK changes the system messages itself
// (a configured SystemPrompt insertion or a compaction/trim installation).
func (a *Agent) ReplaceHistoryCheckedRevision(messages []llm.Message) (HistoryPublication, error) {
	a.mu.Lock()
	owned := llm.CloneMessages(messages)
	if a.manualCompactionActive || (a.turnActive.Load() && !activeHistoryReplacementSafe(a.messages, owned)) {
		a.mu.Unlock()
		return HistoryPublication{}, ErrActiveHistoryMutation
	}
	a.messages = owned
	publication := a.recordHostPublicationLocked()
	a.resetEphemeralTrackingLocked()
	a.mu.Unlock()
	a.cleanupToolResultDumps(toolResultDumpNow(), true)
	return publication, nil
}

// recordHostPublicationLocked makes the just-installed host history the
// current publication. Callers hold a.mu.
func (a *Agent) recordHostPublicationLocked() HistoryPublication {
	publication := HistoryPublication{Revision: hostPublicationRevisions.Add(1), Replaced: a.hostPublication}
	a.hostPublication = publication.Revision
	return publication
}

// messagesAndHostPublication returns an owned history, the publication
// whose system messages it carries (zero is unknown) and the compaction
// generation that installed it, read together.
func (a *Agent) messagesAndHostPublication() ([]llm.Message, uint64, uint64) {
	a.mu.Lock()
	defer a.mu.Unlock()
	return llm.CloneMessages(a.messages), a.hostPublication, a.compactionGeneration.Load()
}

// ClearHistoryChecked rejects destructive clearing during an active query.
func (a *Agent) ClearHistoryChecked() error { return a.ReplaceHistoryChecked(nil) }

func messageJSONEqual(left, right []llm.Message) bool {
	l, err := json.Marshal(left)
	if err != nil {
		return false
	}
	r, err := json.Marshal(right)
	return err == nil && bytes.Equal(l, r)
}

func activeHistoryReplacementSafe(current, candidate []llm.Message) bool {
	for _, m := range candidate {
		if m.Role == llm.RoleSystem && (len(m.ToolCalls) != 0 || m.ToolCallID != "" || m.ToolName != "" || m.IsError || m.Ephemeral || m.Destroyed) {
			return false
		}
	}
	nonSystem := func(messages []llm.Message) []llm.Message {
		body := make([]llm.Message, 0, len(messages))
		for _, m := range messages {
			if m.Role != llm.RoleSystem {
				body = append(body, m)
			}
		}
		return body
	}
	if !messageJSONEqual(nonSystem(current), nonSystem(candidate)) {
		return false
	}
	start := llm.OpenToolCallBlockStart(current)
	if start < 0 {
		_, changed, _ := repairToolCallPairsDetailed(candidate)
		return !changed
	}
	ordinal := 0
	for _, m := range current[:start] {
		if m.Role != llm.RoleSystem {
			ordinal++
		}
	}
	for i, m := range candidate {
		if m.Role == llm.RoleSystem {
			continue
		}
		if ordinal == 0 {
			if !messageJSONEqual(current[start:], candidate[i:]) {
				return false
			}
			// Read-only validation: do not introduce a System gap into an
			// already completed call/result block before the protected tail.
			_, changed, _ := repairToolCallPairsDetailed(candidate[:i])
			return !changed
		}
		ordinal--
	}
	return false
}

// Public checked replacement preserves non-system order. Locate the current
// assistant by complete JSON identity under the same history lock, not by an
// index captured before a legal system-prefix insertion/removal.
func assistantAnchorIndex(messages []llm.Message, current llm.Message) int {
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role == llm.RoleAssistant && messageJSONEqual(messages[i:i+1], []llm.Message{current}) {
			return i
		}
	}
	return -1
}
