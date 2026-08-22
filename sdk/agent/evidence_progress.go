package agent

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools/sandbox"
)

const evidenceNoProgressThreshold = 1

type evidenceRange struct {
	start int64
	end   int64
}

type evidenceRequest struct {
	family      string
	target      string
	key         string
	signature   string
	fingerprint string
	rangeUnit   string
	readRange   *evidenceRange
	targetState string
}

type evidenceTargetState struct {
	ranges              []evidenceRange
	signatures          map[string]struct{}
	resultDigests       map[string]struct{}
	noProgress          int
	executed            int
	suppressed          int
	recoverySent        bool
	lastCompactionAllow uint64
	pendingAfterCompact bool
	priorGeneration     uint64
	targetState         string
}

type evidenceProgressLedger struct {
	deps                 *tools.Container
	targets              map[string]*evidenceTargetState
	compactionGeneration uint64
}

type evidenceDecision struct {
	suppress bool
	recovery bool
	content  string
	metadata map[string]any
}

func newEvidenceProgressLedger(deps *tools.Container, compactionGeneration uint64) *evidenceProgressLedger {
	return &evidenceProgressLedger{
		deps:                 deps,
		targets:              map[string]*evidenceTargetState{},
		compactionGeneration: compactionGeneration,
	}
}

func evidenceFamily(toolName string) string {
	switch strings.ToLower(strings.TrimSpace(toolName)) {
	case "read", "read_file":
		return "read"
	case "grep", "grep_files":
		return "search"
	case "list", "ls", "list_dir":
		return "list"
	default:
		return ""
	}
}

func newEvidenceRequest(toolName string, normalized json.RawMessage, raw string, deps *tools.Container) (evidenceRequest, bool) {
	family := evidenceFamily(toolName)
	if family == "" {
		return evidenceRequest{}, false
	}
	args := map[string]any{}
	data := normalized
	if len(data) == 0 {
		data = json.RawMessage(strings.TrimSpace(raw))
	}
	_ = json.Unmarshal(data, &args)
	target := evidenceTargetFromArgs(family, args)
	target = normalizeEvidenceTarget(target, deps)
	canonical := canonicalJSON(data, raw)
	key := family + "|" + target
	var readRange *evidenceRange
	rangeUnit := ""
	if family == "read" {
		mode := strings.ToLower(stringArg(args, "mode"))
		switch {
		case mode == "indentation":
			key += "|indentation|" + shortDigest(canonical)
		case mode == "byte_range" || hasArg(args, "byte_offset") || hasArg(args, "byte_limit"):
			rangeUnit = "byte"
			key += "|" + rangeUnit
			offset, offsetOK := int64Arg(args, "byte_offset")
			limit, limitOK := int64Arg(args, "byte_limit")
			if offsetOK && limitOK && offset >= 0 && limit > 0 && offset <= math.MaxInt64-limit {
				rng := evidenceRange{start: offset, end: offset + limit - 1}
				readRange = &rng
			}
		default:
			rangeUnit = "line"
			key += "|" + rangeUnit
			offset, ok := int64Arg(args, "offset")
			if !ok || offset <= 0 {
				offset = 1
			}
			limit, ok := int64Arg(args, "limit")
			if !ok || limit <= 0 {
				limit = 2000
			}
			rng := evidenceRange{start: offset, end: offset + limit - 1}
			readRange = &rng
		}
	} else {
		key += "|" + shortDigest(canonical)
	}
	req := evidenceRequest{
		family:      family,
		target:      target,
		key:         key,
		signature:   family + "|" + canonical,
		fingerprint: evidenceFingerprint(family + "|" + target + "|" + canonical),
		rangeUnit:   rangeUnit,
		readRange:   readRange,
		targetState: evidenceTargetStateVersion(target),
	}
	return req, true
}

func (l *evidenceProgressLedger) preflight(req evidenceRequest, compactionGeneration uint64) evidenceDecision {
	if l == nil || req.key == "" {
		return evidenceDecision{}
	}
	state := l.targets[req.key]
	if state == nil {
		return evidenceDecision{}
	}
	if req.targetState != "" && state.targetState != "" && req.targetState != state.targetState {
		state.ranges = nil
		state.signatures = map[string]struct{}{}
		state.resultDigests = map[string]struct{}{}
		state.noProgress = 0
		state.recoverySent = false
		state.pendingAfterCompact = false
		state.priorGeneration = 0
		state.targetState = req.targetState
		return evidenceDecision{}
	}
	if compactionGeneration > 0 && state.lastCompactionAllow < compactionGeneration {
		state.priorGeneration = state.lastCompactionAllow
		state.lastCompactionAllow = compactionGeneration
		state.pendingAfterCompact = true
		return evidenceDecision{}
	}
	repeated := false
	if _, ok := state.signatures[req.signature]; ok {
		repeated = true
	}
	if req.readRange != nil && rangeCovered(state.ranges, *req.readRange) {
		repeated = true
	}
	if !repeated || state.noProgress < evidenceNoProgressThreshold {
		return evidenceDecision{}
	}
	state.suppressed++
	state.noProgress++
	recovery := !state.recoverySent
	state.recoverySent = true
	meta := evidenceMetadata(req, state, "recovery_failed", nil)
	return evidenceDecision{
		suppress: true,
		recovery: recovery,
		content: fmt.Sprintf(
			"[already_observed] %s evidence %s was already returned in this query; execution suppressed. Use the existing result, request an uncovered range/target, inspect changed state, or call done.",
			req.family,
			req.fingerprint,
		),
		metadata: meta,
	}
}

func (l *evidenceProgressLedger) observe(req evidenceRequest, result string, isError bool) map[string]any {
	if l == nil || req.key == "" {
		return nil
	}
	state := l.targets[req.key]
	first := state == nil
	if state == nil {
		state = &evidenceTargetState{
			signatures:          map[string]struct{}{},
			resultDigests:       map[string]struct{}{},
			lastCompactionAllow: l.compactionGeneration,
		}
		l.targets[req.key] = state
	}
	if req.targetState != "" {
		state.targetState = req.targetState
	}
	state.executed++
	_, signatureSeen := state.signatures[req.signature]
	state.signatures[req.signature] = struct{}{}
	digest := shortDigest(fmt.Sprintf("error=%t|%s", isError, result))
	_, digestSeen := state.resultDigests[digest]
	state.resultDigests[digest] = struct{}{}
	rangeNovel := false
	if req.readRange != nil {
		rangeNovel = addEvidenceRange(&state.ranges, *req.readRange)
	}
	progress := first || !digestSeen || rangeNovel
	if progress {
		state.noProgress = 0
		state.recoverySent = false
	} else {
		state.noProgress++
	}
	repeated := signatureSeen || digestSeen || (req.readRange != nil && !rangeNovel)
	disposition := "first_seen"
	var priorGeneration *uint64
	if repeated {
		disposition = "repeated_same_generation"
		if state.pendingAfterCompact {
			disposition = "repeated_after_compaction"
			prior := state.priorGeneration
			priorGeneration = &prior
			state.pendingAfterCompact = false
		}
	}
	return evidenceMetadata(req, state, disposition, priorGeneration)
}

func (l *evidenceProgressLedger) invalidateAfter(toolName string, isError bool) {
	if l == nil || isError || evidenceFamily(toolName) != "" {
		return
	}
	l.targets = map[string]*evidenceTargetState{}
}

func evidenceRecoveryMessage(req evidenceRequest) string {
	target := req.target
	if target == "" {
		target = "(default target)"
	}
	return fmt.Sprintf(
		"No-progress recovery: %s evidence for %s (fingerprint %s) has already been observed. Do not repeat covered reads. Change target/range or action, use existing evidence, or call done if the task is complete.",
		req.family,
		target,
		req.fingerprint,
	)
}

func evidenceMetadata(req evidenceRequest, state *evidenceTargetState, disposition string, priorGeneration *uint64) map[string]any {
	repeated := disposition != "first_seen"
	meta := map[string]any{
		"evidence_family":       req.family,
		"evidence_target":       req.target,
		"evidence_fingerprint":  req.fingerprint,
		"evidence_executed":     state.executed,
		"evidence_repeat_count": state.executed,
		"evidence_suppressed":   state.suppressed,
		"evidence_repeated":     repeated,
		"evidence_disposition":  disposition,
	}
	if priorGeneration != nil {
		meta["evidence_prior_generation"] = *priorGeneration
	}
	if disposition == "recovery_failed" {
		meta["no_progress_suppressed"] = true
		meta["recovery_action"] = "change_target_range_or_action"
	}
	return meta
}

func evidenceTargetFromArgs(family string, args map[string]any) string {
	switch family {
	case "read":
		return firstStringArg(args, "filePath", "file_path", "path")
	case "search", "list":
		path := firstStringArg(args, "path", "filePath", "file_path")
		if path == "" {
			path = "."
		}
		return path
	default:
		return ""
	}
}

func normalizeEvidenceTarget(target string, deps *tools.Container) string {
	target = strings.TrimSpace(target)
	if target == "" {
		return ""
	}
	if deps != nil {
		if sb, err := tools.Get(deps, context.Background(), sandbox.Key); err == nil && sb != nil {
			if resolved, err := sb.Resolve(target); err == nil {
				return filepath.Clean(resolved)
			}
			base := strings.TrimSpace(sb.WorkingDir)
			if base == "" {
				base = strings.TrimSpace(sb.RootDir)
			}
			if base != "" && !filepath.IsAbs(target) {
				return filepath.Clean(filepath.Join(base, target))
			}
		}
	}
	return filepath.Clean(target)
}

func evidenceTargetStateVersion(target string) string {
	target = strings.TrimSpace(target)
	if target == "" {
		return ""
	}
	info, err := os.Stat(target)
	if err != nil {
		if os.IsNotExist(err) {
			return "missing"
		}
		return ""
	}
	return fmt.Sprintf("size=%d|mtime=%d|mode=%d", info.Size(), info.ModTime().UnixNano(), info.Mode())
}

func canonicalJSON(normalized json.RawMessage, raw string) string {
	data := normalized
	if len(data) == 0 {
		data = json.RawMessage(strings.TrimSpace(raw))
	}
	var decoded any
	if len(data) > 0 && json.Unmarshal(data, &decoded) == nil {
		if encoded, err := json.Marshal(decoded); err == nil {
			return string(encoded)
		}
	}
	return strings.TrimSpace(raw)
}

func shortDigest(text string) string {
	sum := sha256.Sum256([]byte(text))
	return hex.EncodeToString(sum[:6])
}

func evidenceFingerprint(text string) string {
	sum := sha256.Sum256([]byte(text))
	return "sha256:" + hex.EncodeToString(sum[:])
}

func stringArg(args map[string]any, key string) string {
	if args == nil {
		return ""
	}
	v, _ := args[key].(string)
	return strings.TrimSpace(v)
}

func firstStringArg(args map[string]any, keys ...string) string {
	for _, key := range keys {
		if value := stringArg(args, key); value != "" {
			return value
		}
	}
	return ""
}

func hasArg(args map[string]any, key string) bool {
	if args == nil {
		return false
	}
	_, ok := args[key]
	return ok
}

func int64Arg(args map[string]any, key string) (int64, bool) {
	if args == nil {
		return 0, false
	}
	switch v := args[key].(type) {
	case float64:
		if math.IsNaN(v) || math.IsInf(v, 0) || math.Trunc(v) != v || v < math.MinInt64 || v > math.MaxInt64 {
			return 0, false
		}
		return int64(v), true
	case int:
		return int64(v), true
	case int64:
		return v, true
	case json.Number:
		n, err := v.Int64()
		return n, err == nil
	default:
		return 0, false
	}
}

func rangeCovered(ranges []evidenceRange, candidate evidenceRange) bool {
	for _, current := range ranges {
		if candidate.start >= current.start && candidate.end <= current.end {
			return true
		}
	}
	return false
}

func addEvidenceRange(ranges *[]evidenceRange, candidate evidenceRange) bool {
	if ranges == nil || candidate.start < 0 || candidate.end < candidate.start {
		return false
	}
	covered := rangeCovered(*ranges, candidate)
	all := append(append([]evidenceRange(nil), (*ranges)...), candidate)
	sort.Slice(all, func(i, j int) bool { return all[i].start < all[j].start })
	merged := make([]evidenceRange, 0, len(all))
	for _, current := range all {
		if len(merged) == 0 || current.start > merged[len(merged)-1].end+1 {
			merged = append(merged, current)
			continue
		}
		if current.end > merged[len(merged)-1].end {
			merged[len(merged)-1].end = current.end
		}
	}
	*ranges = merged
	return !covered
}
