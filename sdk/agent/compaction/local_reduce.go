package compaction

import (
	"context"
	"fmt"
	"regexp"
	"strings"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/agent/messageorigin"
	"github.com/timwhitez/agent-sdk-golang/sdk/artifact"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

const (
	tierSnip                             = "snip"
	tierPrune                            = "prune"
	tierMicrocompact                     = "microcompact"
	tierPlaceholderCleanup               = "placeholder_cleanup"
	defaultSnipMinTokens                 = 64
	defaultUserCodeMicrocompactMinTokens = 96
	userCodePreviewHeadLines             = 12
	userCodePreviewTailLines             = 8
)

var truncationArtifactRe = regexp.MustCompile(`(?i)(?:saved to\s+|full output:\s*|full_output=\s*)([^\]\s]+(?:/|\\)truncated(?:/|\\)[^\]\s]+)`)

func (s *Service) CompactLocal(ctx context.Context, messages []llm.Message, usage *llm.Usage) ([]llm.Message, Result, error) {
	return s.compactLocalWithWatermark(ctx, messages, usage, "")
}

// CompactDestroyedPlaceholders removes only already-destroyed tool-result
// blocks and repairs the associated assistant tool-call topology. It is a
// deterministic low-watermark cleanup: it never invokes the summary model and
// never deletes a mixed block that still contains a live tool result.
func (s *Service) CompactDestroyedPlaceholders(ctx context.Context, messages []llm.Message, usage *llm.Usage) ([]llm.Message, Result, error) {
	if s == nil || !s.Config.Enabled || len(messages) == 0 {
		return messages, Result{Compacted: false}, nil
	}
	if ctx != nil && ctx.Err() != nil {
		return messages, Result{Compacted: false}, ctx.Err()
	}
	originalTokens := s.approximateMessageTokens(messages)
	out, changed := removeDestroyedToolBlocks(messages)
	if !changed {
		return messages, Result{
			Compacted:        false,
			Watermark:        tierPlaceholderCleanup,
			Usage:            cloneUsage(usage),
			OriginalTokens:   originalTokens,
			NewTokens:        originalTokens,
			TokenCountSource: TokenCountSourceEstimate,
		}, nil
	}
	return out, Result{
		Compacted:        true,
		Watermark:        tierPlaceholderCleanup,
		Usage:            cloneUsage(usage),
		OriginalTokens:   originalTokens,
		NewTokens:        s.approximateMessageTokens(out),
		TokenCountSource: TokenCountSourceEstimate,
		TiersApplied:     []string{tierPlaceholderCleanup},
	}, nil
}

// removeDestroyedToolBlocks drops the tool results that were already recycled to
// the ephemeral placeholder and repairs the assistant tool-call topology that
// referenced them.
//
// A block's results are located by tool_call_id rather than by positional
// adjacency, and the search spans non-assistant messages: a framework-authored
// user reminder can end up between two results of the same block, and a
// positional scan would then stop halfway and treat the remaining results as
// unrelated messages — dropping the destroyed ones while keeping the live ones,
// which turns one malformed history into a second one (an assistant tool_use
// without its result, or a tool_result without its tool_use).
//
// The invariants are: a block is cleaned up only when every one of its results
// is destroyed, a block that still holds a live result is left completely
// untouched, and no live tool result is ever removed. Pre-existing orphans that
// belong to no block are not repaired here — the outgoing-request repair owns
// that — except for destroyed placeholders, which carry no information.
func removeDestroyedToolBlocks(messages []llm.Message) ([]llm.Message, bool) {
	dropped := make([]bool, len(messages))
	clearedCalls := make([]bool, len(messages))
	owned := make([]bool, len(messages))
	changed := false

	for i := 0; i < len(messages); i++ {
		msg := messages[i]
		if msg.Role != llm.RoleAssistant || len(msg.ToolCalls) == 0 {
			continue
		}
		members, destroyed, live := toolBlockMembers(messages, i)
		for _, j := range members {
			owned[j] = true
		}
		if destroyed == 0 || live > 0 {
			// Nothing recycled, or the block still carries evidence: leave the
			// whole block — assistant message and every result — as it is.
			continue
		}
		changed = true
		clearedCalls[i] = true
		if msg.Content.IsEmpty() {
			dropped[i] = true
		}
		for _, j := range members {
			dropped[j] = true
		}
	}

	for i, msg := range messages {
		if owned[i] || dropped[i] {
			continue
		}
		if msg.Role == llm.RoleTool && msg.Destroyed {
			// A destroyed placeholder that no surviving tool-call block claims.
			// It holds no information, so removing it cannot lose evidence.
			dropped[i] = true
			changed = true
		}
	}

	if !changed {
		return messages, false
	}
	out := make([]llm.Message, 0, len(messages))
	for i, msg := range messages {
		if dropped[i] {
			continue
		}
		if clearedCalls[i] {
			msg.ToolCalls = nil
			msg.Content = llm.WithoutProviderState(msg.Content)
		}
		out = append(out, msg)
	}
	return out, true
}

// toolBlockMembers returns the indexes of the tool results that answer the
// assistant tool-call block at assistantIndex, plus how many of them are
// destroyed and how many are still live. Results are matched by tool_call_id and
// the search continues across interleaved non-assistant messages, stopping at
// the next assistant or system message. Blocks whose tool calls cannot be
// matched unambiguously (an empty or duplicated ID) fall back to the contiguous
// positional run so that their historical handling is preserved.
func toolBlockMembers(messages []llm.Message, assistantIndex int) ([]int, int, int) {
	calls := messages[assistantIndex].ToolCalls
	pending := make(map[string]bool, len(calls))
	unambiguous := true
	for _, call := range calls {
		id := strings.TrimSpace(call.ID)
		if id == "" {
			unambiguous = false
			break
		}
		if _, exists := pending[id]; exists {
			unambiguous = false
			break
		}
		pending[id] = false
	}

	members := make([]int, 0, len(calls))
	destroyed := 0
	live := 0
	countMember := func(j int) {
		members = append(members, j)
		if messages[j].Destroyed {
			destroyed++
		} else {
			live++
		}
	}

	if !unambiguous {
		for j := assistantIndex + 1; j < len(messages) && messages[j].Role == llm.RoleTool; j++ {
			countMember(j)
		}
		return members, destroyed, live
	}

	for j := assistantIndex + 1; j < len(messages); j++ {
		role := messages[j].Role
		if role == llm.RoleAssistant || role == llm.RoleSystem {
			break
		}
		if role != llm.RoleTool {
			// An interleaved reminder does not close the block: the remaining
			// results of this same block may still follow it.
			continue
		}
		id := strings.TrimSpace(messages[j].ToolCallID)
		if answered, ok := pending[id]; !ok || answered {
			continue
		}
		pending[id] = true
		countMember(j)
	}
	return members, destroyed, live
}

func (s *Service) compactLocalWithWatermark(ctx context.Context, messages []llm.Message, usage *llm.Usage, forcedWatermark string) ([]llm.Message, Result, error) {
	if s == nil || !s.Config.Enabled {
		return messages, Result{Compacted: false}, nil
	}
	if ctx != nil && ctx.Err() != nil {
		return messages, Result{Compacted: false}, ctx.Err()
	}
	if len(messages) == 0 {
		return messages, Result{Compacted: false}, nil
	}
	sessionID := strings.TrimSpace(s.Config.SessionID)
	ledger, loadWarnings, err := s.loadLedger(ctx, sessionID)
	if err != nil {
		return messages, Result{Compacted: false}, err
	}
	originalTokens := s.approximateMessageTokens(messages)
	watermark := strings.TrimSpace(forcedWatermark)
	if watermark == "" {
		watermark = s.WatermarkForUsage(usage)
	}
	if watermark == "" || watermark == "overflow" || watermark == "summarize" {
		watermark = tierSnip
	}
	reducer := localReducer{
		service:       s,
		ctx:           ctx,
		sessionID:     sessionID,
		ledger:        ledger,
		replacements:  ledgerReplacementIndex(ledger),
		protectedFrom: s.protectedZoneStart(messages),
		latestUser:    latestRealUserIndex(messages),
		watermark:     watermark,
	}
	out, warnings, changed, ledgerChanged, tiers := reducer.reduce(messages)
	warnings = append(loadWarnings, warnings...)
	if changed == 0 {
		return messages, Result{
			Compacted:        false,
			Trigger:          "usage",
			Watermark:        watermark,
			Usage:            cloneUsage(usage),
			OriginalTokens:   originalTokens,
			NewTokens:        originalTokens,
			TokenCountSource: TokenCountSourceEstimate,
			Warnings:         warnings,
		}, nil
	}
	if ledgerChanged {
		ledger.UpdatedAt = time.Now().UTC()
		ledger.ContextWindow = s.contextWindow()
		if err := ledger.Validate(sessionID); err != nil {
			return messages, Result{Compacted: false, Warnings: warnings}, err
		}
		if err := s.saveLedger(ctx, sessionID, ledger); err != nil {
			return messages, Result{Compacted: false, Warnings: warnings}, err
		}
	}
	newTokens := s.approximateMessageTokens(out)
	res := Result{
		Compacted:        true,
		Trigger:          "usage",
		Watermark:        watermark,
		Usage:            cloneUsage(usage),
		OriginalTokens:   originalTokens,
		NewTokens:        newTokens,
		TokenCountSource: TokenCountSourceEstimate,
		TiersApplied:     tiers,
		Warnings:         warnings,
	}
	res.LedgerPath = strings.TrimSpace(s.Config.LedgerPath)
	return out, res, nil
}

type localReducer struct {
	service       *Service
	ctx           context.Context
	sessionID     string
	ledger        *Ledger
	replacements  map[string]LedgerReplacement
	protectedFrom int
	latestUser    int
	watermark     string
	ledgerChanged bool
}

func (r *localReducer) reduce(messages []llm.Message) ([]llm.Message, []string, int, bool, []string) {
	out := make([]llm.Message, len(messages))
	copy(out, messages)
	warnings := []string{}
	changed := 0
	applied := map[string]struct{}{}
	for i, msg := range messages {
		var repl LedgerReplacement
		var warning string
		var ok bool
		switch {
		case msg.Role == llm.RoleTool:
			repl, warning, ok = r.reduceToolMessage(i, msg)
		case r.isPrune() && msg.Role == llm.RoleAssistant:
			repl, warning, ok = r.reduceAssistantMessage(i, msg)
		case r.isPrune() && msg.Role == llm.RoleUser:
			repl, warning, ok = r.reduceUserCodeMessage(i, msg)
		default:
			continue
		}
		if warning != "" {
			warnings = append(warnings, warning)
		}
		if !ok {
			continue
		}
		if repl.ReplacementText == msg.Content.PlainText() {
			continue
		}
		msg.Content = llm.TextContent(repl.ReplacementText)
		out[i] = msg
		changed++
		applied[repl.Tier] = struct{}{}
	}
	return out, warnings, changed, r.ledgerChanged, orderedLocalTiers(applied)
}

func (r *localReducer) isPrune() bool {
	return r.watermark == tierPrune
}

func (r *localReducer) reduceToolMessage(index int, msg llm.Message) (LedgerReplacement, string, bool) {
	if r.isPrune() {
		text := msg.Content.PlainText()
		if parent, ok := r.findReplacementByText(text, msg); ok {
			if r.canonicalArtifactsConfigured() {
				validated, warning, valid := r.validateCanonicalReplacement(msg, parent)
				if !valid {
					return LedgerReplacement{}, warning, false
				}
				parent = validated
			}
			if parent.Tier == tierPrune {
				return parent, "", true
			}
			pruned := r.createToolPruneReplacement(msg, parent)
			pruneKey := replacementLookupKey(pruned.MessageKey, pruned.PartKey)
			if existing, exists := r.replacements[pruneKey]; exists && existing.Tier == tierPrune {
				return r.validateExistingPruneReplacement(msg, parent, existing)
			}
			r.recordReplacement(pruned)
			return pruned, "", true
		}
	}
	if !r.eligibleToolMessage(index, msg) {
		return LedgerReplacement{}, "", false
	}
	original := msg.Content.PlainText()
	key := StableMessageKey(MessageKeyInput{
		Role:           string(msg.Role),
		ToolCallID:     msg.ToolCallID,
		ToolName:       msg.ToolName,
		OriginalText:   original,
		FirstSeenIndex: index,
	})
	partKey := "content-0"
	lookupKey := replacementLookupKey(key, partKey)
	repl, ok := r.replacements[lookupKey]
	if !ok {
		created, warning, ok := r.createSnipReplacement(msg, key, partKey, original)
		if !ok {
			return LedgerReplacement{}, warning, false
		}
		repl = created
		r.recordReplacement(repl)
	} else if r.canonicalArtifactsConfigured() {
		if repl.CanonicalArtifact == nil {
			created, warning, migrated := r.createSnipReplacement(msg, key, partKey, original)
			if !migrated {
				return LedgerReplacement{}, warning, false
			}
			r.replaceReplacement(created)
			repl = created
		} else {
			validated, warning, valid := r.validateCanonicalReplacement(msg, repl)
			if !valid {
				return LedgerReplacement{}, warning, false
			}
			repl = validated
		}
	}
	if r.isPrune() && repl.Tier == tierSnip {
		pruned := r.createToolPruneReplacement(msg, repl)
		pruneKey := replacementLookupKey(pruned.MessageKey, pruned.PartKey)
		if existing, exists := r.replacements[pruneKey]; exists && existing.Tier == tierPrune {
			return r.validateExistingPruneReplacement(msg, repl, existing)
		}
		r.recordReplacement(pruned)
		return pruned, "", true
	}
	return repl, "", true
}

func (r *localReducer) findReplacementByText(text string, msg llm.Message) (LedgerReplacement, bool) {
	hash := ContentHash(text)
	for _, repl := range r.ledger.Replacements {
		if repl.ReplacementHash != hash {
			continue
		}
		if repl.Role != string(msg.Role) {
			continue
		}
		if strings.TrimSpace(repl.ToolName) != "" && strings.TrimSpace(repl.ToolName) != strings.TrimSpace(msg.ToolName) {
			continue
		}
		return repl, true
	}
	return LedgerReplacement{}, false
}

func (r *localReducer) eligibleToolMessage(index int, msg llm.Message) bool {
	if index >= r.protectedFrom {
		return false
	}
	if msg.Role != llm.RoleTool || msg.Destroyed {
		return false
	}
	if isProtectedTool(msg.ToolName, r.service.protectedTools) {
		return false
	}
	text := msg.Content.PlainText()
	if strings.TrimSpace(text) == "" {
		return false
	}
	if isToolCompactionStub(text) {
		return false
	}
	return r.service.estimateTextTokens(text) >= defaultSnipMinTokens
}

func (r *localReducer) reduceAssistantMessage(index int, msg llm.Message) (LedgerReplacement, string, bool) {
	if !r.eligibleAssistantMessage(index, msg) {
		return LedgerReplacement{}, "", false
	}
	original := msg.Content.PlainText()
	key := StableMessageKey(MessageKeyInput{
		Role:           string(msg.Role),
		OriginalText:   original,
		FirstSeenIndex: index,
	})
	partKey := "content-0"
	lookupKey := replacementLookupKey(key, partKey)
	if repl, ok := r.replacements[lookupKey]; ok {
		if !r.canonicalArtifactsConfigured() {
			return repl, "", true
		}
		if repl.CanonicalArtifact != nil {
			validated, warning, valid := r.validateCanonicalReplacement(msg, repl)
			return validated, warning, valid
		}
		created, warning, migrated := r.createAssistantReplacement(msg, key, partKey, original)
		if migrated {
			r.replaceReplacement(created)
		}
		return created, warning, migrated
	}
	created, warning, ok := r.createAssistantReplacement(msg, key, partKey, original)
	if ok {
		r.recordReplacement(created)
	}
	return created, warning, ok
}

func (r *localReducer) createAssistantReplacement(msg llm.Message, key, partKey, original string) (LedgerReplacement, string, bool) {
	if r.canonicalArtifactsConfigured() {
		manifest, stage, err := r.canonicalArtifactForContent(msg, artifact.ObjectKindCompactionMaterial, original)
		if err != nil {
			return LedgerReplacement{}, canonicalCompactionArtifactWarning(r.sessionID, "assistant", "", stage, err.Error()), false
		}
		text := canonicalAssistantPruneReplacementText(original, manifest, r.service.estimateTextTokens)
		repl := LedgerReplacement{
			MessageKey:        key,
			PartKey:           partKey,
			Role:              string(msg.Role),
			Tier:              tierPrune,
			OriginalHash:      ContentHash(original),
			ReplacementHash:   ContentHash(text),
			ReplacementText:   text,
			CanonicalArtifact: cloneCanonicalManifestPointer(manifest),
			CreatedAt:         time.Now().UTC(),
			OriginalText:      original,
		}
		return repl, "", true
	}
	if r.service.Config.ToolArtifactWriter == nil {
		return LedgerReplacement{}, compactionArtifactWarning(r.sessionID, "assistant", "", "write", "no artifact writer configured"), false
	}
	artifact, err := r.service.Config.ToolArtifactWriter.SaveCompactionArtifact(r.ctx, ArtifactRequest{
		SessionID:  r.sessionID,
		MessageKey: key,
		PartKey:    partKey,
		ToolName:   "assistant",
		Content:    original,
	})
	if err != nil {
		return LedgerReplacement{}, compactionArtifactWarning(r.sessionID, "assistant", "", "write", err.Error()), false
	}
	artifactPath := strings.TrimSpace(artifact.Path)
	if artifactPath == "" {
		return LedgerReplacement{}, compactionArtifactWarning(r.sessionID, "assistant", "", "write", "artifact writer returned empty path"), false
	}
	text := assistantPruneReplacementText(original, artifactPath, r.service.estimateTextTokens)
	repl := LedgerReplacement{
		MessageKey:      key,
		PartKey:         partKey,
		Role:            string(msg.Role),
		Tier:            tierPrune,
		OriginalHash:    ContentHash(original),
		ReplacementHash: ContentHash(text),
		ReplacementText: text,
		FullArtifact:    artifactPath,
		CreatedAt:       time.Now().UTC(),
		OriginalText:    original,
	}
	return repl, "", true
}

func (r *localReducer) eligibleAssistantMessage(index int, msg llm.Message) bool {
	if index >= r.protectedFrom {
		return false
	}
	if msg.Destroyed || len(msg.ToolCalls) > 0 {
		return false
	}
	text := msg.Content.PlainText()
	if strings.TrimSpace(text) == "" {
		return false
	}
	if isAssistantCompactionStub(text) {
		return false
	}
	return r.service.estimateTextTokens(text) >= defaultSnipMinTokens
}

func (r *localReducer) reduceUserCodeMessage(index int, msg llm.Message) (LedgerReplacement, string, bool) {
	if !r.eligibleUserCodeMessage(index, msg) {
		return LedgerReplacement{}, "", false
	}
	original := msg.Content.PlainText()
	key := StableMessageKey(MessageKeyInput{
		Role:           string(msg.Role),
		OriginalText:   original,
		FirstSeenIndex: index,
	})
	partKey := "content-0"
	lookupKey := replacementLookupKey(key, partKey)
	if repl, ok := r.replacements[lookupKey]; ok {
		if !r.canonicalArtifactsConfigured() {
			return repl, "", true
		}
		if repl.CanonicalArtifact != nil {
			validated, warning, valid := r.validateCanonicalReplacement(msg, repl)
			return validated, warning, valid
		}
		created, warning, migrated := r.createUserCodeReplacement(msg, key, partKey, original)
		if migrated {
			r.replaceReplacement(created)
		}
		return created, warning, migrated
	}
	created, warning, ok := r.createUserCodeReplacement(msg, key, partKey, original)
	if ok {
		r.recordReplacement(created)
	}
	return created, warning, ok
}

func (r *localReducer) createUserCodeReplacement(msg llm.Message, key, partKey, original string) (LedgerReplacement, string, bool) {
	if r.canonicalArtifactsConfigured() {
		manifest, stage, err := r.canonicalArtifactForContent(msg, artifact.ObjectKindCompactionMaterial, original)
		if err != nil {
			return LedgerReplacement{}, canonicalCompactionArtifactWarning(r.sessionID, "user_code", "", stage, err.Error()), false
		}
		text, ok := canonicalUserCodeMicrocompactReplacementText(original, manifest, r.service.estimateTextTokens)
		if !ok {
			return LedgerReplacement{}, "", false
		}
		repl := LedgerReplacement{
			MessageKey:        key,
			PartKey:           partKey,
			Role:              string(msg.Role),
			Tier:              tierMicrocompact,
			OriginalHash:      ContentHash(original),
			ReplacementHash:   ContentHash(text),
			ReplacementText:   text,
			CanonicalArtifact: cloneCanonicalManifestPointer(manifest),
			CreatedAt:         time.Now().UTC(),
			OriginalText:      original,
		}
		return repl, "", true
	}
	if r.service.Config.ToolArtifactWriter == nil {
		return LedgerReplacement{}, userCodeArtifactWarning(r.sessionID, "write", "no artifact writer configured"), false
	}
	artifact, err := r.service.Config.ToolArtifactWriter.SaveCompactionArtifact(r.ctx, ArtifactRequest{
		SessionID:  r.sessionID,
		MessageKey: key,
		PartKey:    partKey,
		ToolName:   "user_code",
		Content:    original,
	})
	if err != nil {
		return LedgerReplacement{}, userCodeArtifactWarning(r.sessionID, "write", err.Error()), false
	}
	artifactPath := strings.TrimSpace(artifact.Path)
	if artifactPath == "" {
		return LedgerReplacement{}, userCodeArtifactWarning(r.sessionID, "write", "artifact writer returned empty path"), false
	}
	text, ok := userCodeMicrocompactReplacementText(original, artifactPath, r.service.estimateTextTokens)
	if !ok {
		return LedgerReplacement{}, "", false
	}
	repl := LedgerReplacement{
		MessageKey:      key,
		PartKey:         partKey,
		Role:            string(msg.Role),
		Tier:            tierMicrocompact,
		OriginalHash:    ContentHash(original),
		ReplacementHash: ContentHash(text),
		ReplacementText: text,
		FullArtifact:    artifactPath,
		CreatedAt:       time.Now().UTC(),
		OriginalText:    original,
	}
	return repl, "", true
}

func (r *localReducer) recordReplacement(repl LedgerReplacement) {
	if r == nil || r.ledger == nil {
		return
	}
	r.ledger.Replacements = append(r.ledger.Replacements, repl)
	r.replacements[replacementLookupKey(repl.MessageKey, repl.PartKey)] = repl
	r.ledgerChanged = true
}

func (r *localReducer) replaceReplacement(repl LedgerReplacement) {
	if r == nil || r.ledger == nil {
		return
	}
	key := replacementLookupKey(repl.MessageKey, repl.PartKey)
	for i := range r.ledger.Replacements {
		if replacementLookupKey(r.ledger.Replacements[i].MessageKey, r.ledger.Replacements[i].PartKey) != key {
			continue
		}
		r.ledger.Replacements[i] = repl
		r.replacements[key] = repl
		r.ledgerChanged = true
		return
	}
	r.recordReplacement(repl)
}

func isToolCompactionStub(text string) bool {
	trimmed := strings.TrimSpace(text)
	return strings.HasPrefix(trimmed, "[Tool result snipped:") ||
		strings.HasPrefix(trimmed, "[Tool result pruned:")
}

func isAssistantCompactionStub(text string) bool {
	return strings.HasPrefix(strings.TrimSpace(text), "[Assistant text compacted:")
}

func (r *localReducer) eligibleUserCodeMessage(index int, msg llm.Message) bool {
	if !r.service.Config.EnableUserCodeMicrocompact {
		return false
	}
	if index >= r.protectedFrom || index == r.latestUser || !messageorigin.IsRealUserMessage(msg) {
		return false
	}
	text := msg.Content.PlainText()
	if strings.TrimSpace(text) == "" {
		return false
	}
	return userCodeMicrocompactCandidate(text, r.service.estimateTextTokens)
}

func (r *localReducer) createSnipReplacement(msg llm.Message, messageKey, partKey, original string) (LedgerReplacement, string, bool) {
	if r.canonicalArtifactsConfigured() {
		manifest, stage, err := r.canonicalArtifactForContent(msg, artifact.ObjectKindLogicalToolResult, original)
		if err != nil {
			return LedgerReplacement{}, canonicalCompactionArtifactWarning(r.sessionID, msg.ToolName, msg.ToolCallID, stage, err.Error()), false
		}
		replacementText := canonicalSnipReplacementText(msg, original, manifest)
		now := time.Now().UTC()
		return LedgerReplacement{
			MessageKey:        messageKey,
			PartKey:           partKey,
			Role:              string(msg.Role),
			ToolName:          strings.TrimSpace(msg.ToolName),
			Tier:              tierSnip,
			OriginalHash:      ContentHash(original),
			ReplacementHash:   ContentHash(replacementText),
			ReplacementText:   replacementText,
			CanonicalArtifact: cloneCanonicalManifestPointer(manifest),
			CreatedAt:         now,
			OriginalText:      original,
		}, "", true
	}
	artifactPath := extractTruncationArtifactPath(original)
	if artifactPath == "" {
		if r.service.Config.ToolArtifactWriter == nil {
			return LedgerReplacement{}, compactionArtifactWarning(r.sessionID, msg.ToolName, msg.ToolCallID, "write", "no artifact writer configured"), false
		}
		artifact, err := r.service.Config.ToolArtifactWriter.SaveCompactionArtifact(r.ctx, ArtifactRequest{
			SessionID:  r.sessionID,
			MessageKey: messageKey,
			PartKey:    partKey,
			ToolName:   msg.ToolName,
			ToolCallID: msg.ToolCallID,
			Content:    original,
		})
		if err != nil {
			return LedgerReplacement{}, compactionArtifactWarning(r.sessionID, msg.ToolName, msg.ToolCallID, "write", err.Error()), false
		}
		artifactPath = strings.TrimSpace(artifact.Path)
		if artifactPath == "" {
			return LedgerReplacement{}, compactionArtifactWarning(r.sessionID, msg.ToolName, msg.ToolCallID, "write", "artifact writer returned empty path"), false
		}
	}
	replacementText := snipReplacementText(msg, original, artifactPath)
	now := time.Now().UTC()
	return LedgerReplacement{
		MessageKey:      messageKey,
		PartKey:         partKey,
		Role:            string(msg.Role),
		ToolName:        strings.TrimSpace(msg.ToolName),
		Tier:            tierSnip,
		OriginalHash:    ContentHash(original),
		ReplacementHash: ContentHash(replacementText),
		ReplacementText: replacementText,
		FullArtifact:    artifactPath,
		CreatedAt:       now,
		OriginalText:    original,
	}, "", true
}

func (r *localReducer) createToolPruneReplacement(msg llm.Message, parent LedgerReplacement) LedgerReplacement {
	text := toolPruneReplacementText(msg, parent.FullArtifact)
	if parent.CanonicalArtifact != nil {
		text = canonicalToolPruneReplacementText(msg, *parent.CanonicalArtifact)
	}
	return LedgerReplacement{
		MessageKey:            parent.MessageKey + "/tier:prune",
		PartKey:               parent.PartKey,
		Role:                  string(msg.Role),
		ToolName:              strings.TrimSpace(msg.ToolName),
		Tier:                  tierPrune,
		OriginalHash:          parent.ReplacementHash,
		ReplacementHash:       ContentHash(text),
		ReplacementText:       text,
		CanonicalArtifact:     cloneCanonicalManifestPointerFrom(parent.CanonicalArtifact),
		FullArtifact:          strings.TrimSpace(parent.FullArtifact),
		ParentReplacementHash: parent.ReplacementHash,
		CreatedAt:             time.Now().UTC(),
	}
}

func snipReplacementText(msg llm.Message, original, artifactPath string) string {
	tool := strings.TrimSpace(msg.ToolName)
	if tool == "" {
		tool = "tool"
	}
	id := strings.TrimSpace(msg.ToolCallID)
	if id == "" {
		id = "-"
	}
	lines := countTextLines(original)
	return fmt.Sprintf("[Tool result snipped: %s tool_call_id=%s lines=%d bytes=%d full_output=%s]", tool, id, lines, len(original), strings.TrimSpace(artifactPath))
}

func toolPruneReplacementText(msg llm.Message, artifactPath string) string {
	tool := strings.TrimSpace(msg.ToolName)
	if tool == "" {
		tool = "tool"
	}
	id := strings.TrimSpace(msg.ToolCallID)
	if id == "" {
		id = "-"
	}
	return fmt.Sprintf("[Tool result pruned: %s tool_call_id=%s full_output=%s]", tool, id, strings.TrimSpace(artifactPath))
}

func assistantPruneReplacementText(original, artifactPath string, estimate tokenEstimator) string {
	return fmt.Sprintf("[Assistant text compacted: lines=%d bytes=%d full_output=%s]\n%s", countTextLines(original), len(original), strings.TrimSpace(artifactPath), assistantPreviewWithEstimator(original, estimate))
}

type fencedCodeBlock struct {
	startLine int
	endLine   int
	info      string
	language  string
	hint      string
	lines     []string
}

func userCodeMicrocompactCandidate(text string, estimate func(string) int) bool {
	block, ok := largestFencedCodeBlock(text)
	if !ok {
		return false
	}
	return estimate(strings.Join(block.lines, "\n")) >= defaultUserCodeMicrocompactMinTokens
}

func userCodeMicrocompactReplacementText(original, artifactPath string, estimate func(string) int) (string, bool) {
	block, ok := largestFencedCodeBlock(original)
	if !ok {
		return "", false
	}
	if estimate(strings.Join(block.lines, "\n")) < defaultUserCodeMicrocompactMinTokens {
		return "", false
	}
	allLines := strings.Split(strings.ReplaceAll(original, "\r\n", "\n"), "\n")
	out := make([]string, 0, len(allLines)-len(block.lines)+userCodePreviewHeadLines+userCodePreviewTailLines+4)
	out = append(out, allLines[:block.startLine]...)
	out = append(out, userCodeBlockReplacementLines(block, artifactPath)...)
	if block.endLine+1 < len(allLines) {
		out = append(out, allLines[block.endLine+1:]...)
	}
	return strings.TrimSpace(strings.Join(out, "\n")), true
}

func largestFencedCodeBlock(text string) (fencedCodeBlock, bool) {
	lines := strings.Split(strings.ReplaceAll(text, "\r\n", "\n"), "\n")
	best := fencedCodeBlock{}
	haveBest := false
	open := -1
	info := ""
	for i, line := range lines {
		trimmed := strings.TrimSpace(line)
		if !strings.HasPrefix(trimmed, "```") {
			continue
		}
		if open < 0 {
			open = i
			info = strings.TrimSpace(strings.TrimPrefix(trimmed, "```"))
			continue
		}
		blockLines := append([]string(nil), lines[open+1:i]...)
		candidate := fencedCodeBlock{
			startLine: open,
			endLine:   i,
			info:      info,
			lines:     blockLines,
		}
		candidate.language, candidate.hint = parseFenceInfo(info)
		if !haveBest || len(candidate.lines) > len(best.lines) {
			best = candidate
			haveBest = true
		}
		open = -1
		info = ""
	}
	return best, haveBest
}

func parseFenceInfo(info string) (string, string) {
	fields := strings.Fields(strings.TrimSpace(info))
	if len(fields) == 0 {
		return "", ""
	}
	lang := fields[0]
	hint := ""
	if len(fields) > 1 {
		hint = strings.Join(fields[1:], " ")
	}
	return lang, hint
}

func userCodeBlockReplacementLines(block fencedCodeBlock, artifactPath string) []string {
	lang := strings.TrimSpace(block.language)
	if lang == "" {
		lang = "-"
	}
	hint := strings.TrimSpace(block.hint)
	if hint == "" {
		hint = "-"
	}
	header := fmt.Sprintf("[User code block compacted: language=%s hint=%s lines=%d bytes=%d full_output=%s]", lang, hint, len(block.lines), len(strings.Join(block.lines, "\n")), strings.TrimSpace(artifactPath))
	out := []string{header}
	out = append(out, "Preview:")
	switch {
	case len(block.lines) <= userCodePreviewHeadLines+userCodePreviewTailLines:
		out = append(out, block.lines...)
	default:
		out = append(out, block.lines[:userCodePreviewHeadLines]...)
		out = append(out, fmt.Sprintf("[...%d middle lines omitted; full code: %s]", len(block.lines)-userCodePreviewHeadLines-userCodePreviewTailLines, strings.TrimSpace(artifactPath)))
		out = append(out, block.lines[len(block.lines)-userCodePreviewTailLines:]...)
	}
	return out
}

func assistantPreview(text string) string {
	return assistantPreviewWithEstimator(text, approximateTextTokens)
}

func assistantPreviewWithEstimator(text string, estimate tokenEstimator) string {
	return truncateTextToTokenBudget(text, assistantPreviewTokenBudget, estimate)
}

func countTextLines(text string) int {
	if text == "" {
		return 0
	}
	return strings.Count(text, "\n") + 1
}

func extractTruncationArtifactPath(text string) string {
	m := truncationArtifactRe.FindStringSubmatch(text)
	if len(m) < 2 {
		return ""
	}
	return strings.TrimSpace(strings.TrimRight(m[1], ".,;"))
}

func compactionArtifactWarning(sessionID, toolName, toolCallID, stage, detail string) string {
	return fmt.Sprintf("[WARN] Compaction artifact not saved - session=%s stage=%s tool=%s tool_call_id=%s action=leaving original tool result in context: %s", strings.TrimSpace(sessionID), strings.TrimSpace(stage), strings.TrimSpace(toolName), strings.TrimSpace(toolCallID), strings.TrimSpace(detail))
}

func userCodeArtifactWarning(sessionID, stage, detail string) string {
	return fmt.Sprintf("[WARN] Compaction artifact not saved - session=%s stage=%s role=user action=leaving original user code in context: %s", strings.TrimSpace(sessionID), strings.TrimSpace(stage), strings.TrimSpace(detail))
}

func protectedMessageStart(length int, protected int) int {
	if protected <= 0 {
		return length
	}
	start := length - protected
	if start < 0 {
		return 0
	}
	return start
}

func latestRealUserIndex(messages []llm.Message) int {
	for i := len(messages) - 1; i >= 0; i-- {
		msg := messages[i]
		if messageorigin.IsRealUserMessage(msg) {
			return i
		}
	}
	return -1
}

func (s *Service) protectedZoneStart(messages []llm.Message) int {
	start := protectedMessageStart(len(messages), s.protectedRecentMessages())
	if latestUser := latestRealUserIndex(messages); latestUser >= 0 && latestUser < start {
		start = latestUser
	}
	if openToolBlock := openToolBlockStart(messages); openToolBlock >= 0 && openToolBlock < start {
		start = openToolBlock
	}
	if tokenStart := s.protectedRecentTokenStart(messages); tokenStart >= 0 && tokenStart < start {
		start = tokenStart
	}
	return start
}

func (s *Service) protectedRecentTokenStart(messages []llm.Message) int {
	budget := 0
	if s != nil {
		budget = s.Config.ProtectedRecentTokens
	}
	if budget <= 0 || len(messages) == 0 {
		return len(messages)
	}
	total := 0
	for i := len(messages) - 1; i >= 0; i-- {
		total += s.approximateMessageTokens(messages[i : i+1])
		if total >= budget {
			return i
		}
	}
	return 0
}

func openToolBlockStart(messages []llm.Message) int {
	openStart := -1
	pending := map[string]int{}
	for i, msg := range messages {
		if msg.Role == llm.RoleAssistant && len(msg.ToolCalls) > 0 {
			if openStart < 0 {
				openStart = i
			}
			for callIndex, call := range msg.ToolCalls {
				id := strings.TrimSpace(call.ID)
				if id == "" {
					id = fmt.Sprintf("__missing_tool_call_id_%d_%d", i, callIndex)
				}
				pending[id]++
			}
			continue
		}
		if msg.Role != llm.RoleTool || len(pending) == 0 {
			continue
		}
		id := strings.TrimSpace(msg.ToolCallID)
		if count := pending[id]; count > 1 {
			pending[id] = count - 1
		} else {
			delete(pending, id)
		}
		if len(pending) == 0 {
			openStart = -1
		}
	}
	if len(pending) == 0 {
		return -1
	}
	return openStart
}

func (s *Service) protectedRecentMessages() int {
	if s == nil {
		return DefaultKeepRecentUserMessages
	}
	n := s.Config.ProtectedRecentMessages
	if n <= 0 {
		n = DefaultKeepRecentUserMessages
	}
	return n
}

func ledgerReplacementIndex(ledger *Ledger) map[string]LedgerReplacement {
	out := map[string]LedgerReplacement{}
	if ledger == nil {
		return out
	}
	for _, repl := range ledger.Replacements {
		out[replacementLookupKey(repl.MessageKey, repl.PartKey)] = repl
	}
	return out
}

func replacementLookupKey(messageKey, partKey string) string {
	return strings.TrimSpace(messageKey) + "\x00" + strings.TrimSpace(partKey)
}

func orderedLocalTiers(applied map[string]struct{}) []string {
	out := []string{}
	if _, ok := applied[tierSnip]; ok {
		out = append(out, tierSnip)
	}
	if _, ok := applied[tierPrune]; ok {
		out = append(out, tierPrune)
	}
	if _, ok := applied[tierMicrocompact]; ok {
		out = append(out, tierMicrocompact)
	}
	return out
}

type warningLedgerStore interface {
	LoadCompactionLedger(context.Context, string) (*Ledger, string, error)
}

func (s *Service) loadLedger(ctx context.Context, sessionID string) (*Ledger, []string, error) {
	if s == nil || s.Config.LedgerStore == nil {
		return NewLedger(sessionID), nil, nil
	}
	if store, ok := s.Config.LedgerStore.(warningLedgerStore); ok {
		ledger, warning, err := store.LoadCompactionLedger(ctx, sessionID)
		warnings := []string{}
		if strings.TrimSpace(warning) != "" {
			warnings = append(warnings, strings.TrimSpace(warning))
		}
		if err != nil {
			return nil, warnings, err
		}
		if ledger == nil {
			return NewLedger(sessionID), warnings, nil
		}
		ledger = ledger.Clone()
		if err := ledger.Validate(sessionID); err != nil {
			return nil, warnings, err
		}
		return ledger, warnings, nil
	}
	ledger, err := s.Config.LedgerStore.Load(ctx, sessionID)
	if err != nil {
		return nil, nil, err
	}
	if ledger == nil {
		return NewLedger(sessionID), nil, nil
	}
	ledger = ledger.Clone()
	if err := ledger.Validate(sessionID); err != nil {
		return nil, nil, err
	}
	return ledger, nil, nil
}

func (s *Service) saveLedger(ctx context.Context, sessionID string, ledger *Ledger) error {
	if s == nil || s.Config.LedgerStore == nil {
		return nil
	}
	return s.Config.LedgerStore.Save(ctx, sessionID, ledger)
}
