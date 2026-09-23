package tools

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"math"
	"strings"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

// Tool is an executable capability exposed to the model.
type Tool struct {
	Name        string
	Description string

	// EphemeralKeep controls how many recent outputs are kept in context.
	// 0 means keep all; 1 means keep last 1, etc.
	EphemeralKeep int

	Schema map[string]any

	// Hidden excludes the tool from model-visible tool definitions.
	Hidden bool

	Handler func(ctx context.Context, args json.RawMessage, deps *Container) (llm.Content, error)

	// typed is the sealed decoder of a Func tool. It never replaces Handler:
	// execution always runs the complete (possibly wrapped) Handler chain.
	typed *typedArgsBinding
}

const (
	toolDiagnosticInvalidArgsAction = "Provide valid JSON arguments that match the tool schema and retry."
	toolDiagnosticDefaultAction     = "Review the diagnostic details and retry."
)

func (t Tool) Definition() llm.ToolDefinition {
	return llm.ToolDefinition{
		Name:        t.Name,
		Description: t.Description,
		Parameters:  t.Schema,
		Strict:      true,
	}
}

// PreparedCall owns first-pass argument normalization and the complete Handler
// captured when prepared. It does not predecode typed arguments, freeze closure
// state, authorize effects, or promise replay safety. The caller owns execution.
type PreparedCall struct {
	tool       Tool
	original   string
	normalized json.RawMessage
	meta       map[string]any
	err        error
	final      *preparedTypedArgs
}

// PrepareCall normalizes once without invoking the Handler or resolving deps.
// The returned observation is independent of the bytes and metadata executed by
// the prepared call. Typed repair and wrapper-specific argument changes remain
// inside the captured Handler; this observation is not final business authority.
func (t Tool) PrepareCall(argsJSON string) (PreparedCall, ToolArgsNormalization) {
	norm := NormalizeToolArgs(t.Name, argsJSON, t.Schema)
	prepared := PreparedCall{
		tool: t, original: argsJSON, normalized: bytes.Clone(norm.Normalized),
		// Normalization metadata consists only of bool/string scalars. Display
		// may be nested, but is owned solely by the returned observation.
		meta: maps.Clone(norm.Meta), err: norm.Err,
	}
	if norm.Err == nil {
		prepared.final = prepareTypedArgs(t.typed, prepared.normalized)
	}
	return prepared, norm
}

// FinalArgs returns an owned JSON view of the typed arguments a Func tool's
// sealed decoder accepted when the call was prepared. Execution consumes that
// same decoded object unless a wrapper changes the bytes it forwards (see
// FinalArgsOutcome). ok is false for tools without a sealed decoder, types
// with custom decoders and arguments that do not decode. The view is
// observation only; it neither authorizes effects nor proves independence.
func (p PreparedCall) FinalArgs() (json.RawMessage, bool) {
	if p.final == nil {
		return nil, false
	}
	return bytes.Clone(p.final.view), true
}

// FinalArgsOutcome reports, after Execute, whether the adapter consumed the
// prepared object, decoded different bytes, or was never reached.
func (p PreparedCall) FinalArgsOutcome() string {
	switch {
	case p.final == nil:
		return FinalArgsUnavailable
	case p.final.state.diverged.Load():
		return FinalArgsDiverged
	case p.final.state.consumed.Load():
		return FinalArgsConsumed
	default:
		return FinalArgsUnused
	}
}

func (t Tool) Execute(ctx context.Context, argsJSON string, deps *Container) (llm.Content, error) {
	// Preserve the direct API's missing-handler error before normalization.
	if t.Handler == nil {
		return llm.Content{}, fmt.Errorf("tool %q missing handler", t.Name)
	}
	prepared, _ := t.PrepareCall(argsJSON)
	return prepared.Execute(ctx, deps)
}

// Execute invokes the complete captured Handler, retaining legacy typed decode,
// wrapper, result, and error behavior. Each invocation owns its handler bytes.
func (p PreparedCall) Execute(ctx context.Context, deps *Container) (llm.Content, error) {
	if p.tool.Handler == nil {
		return llm.Content{}, fmt.Errorf("tool %q missing handler", p.tool.Name)
	}
	if p.err != nil {
		if argsRepaired(p.meta) {
			UpsertToolResultMetadata(ctx, p.meta)
		}
		return llm.TextContent(formatToolErrorDiagnostic("Invalid tool arguments", p.err, toolDiagnosticInvalidArgsAction)), p.err
	}
	if p.normalized == nil {
		parseErr := p.err
		if parseErr == nil {
			parseErr = fmt.Errorf("invalid tool args")
		}
		return llm.TextContent(formatToolErrorDiagnostic("Invalid tool arguments", parseErr, toolDiagnosticInvalidArgsAction)), parseErr
	}

	if argsRepaired(p.meta) {
		UpsertToolResultMetadata(ctx, p.meta)
	}
	// A handler error cannot prove that no side effect happened. Never replay
	// execution based on its error text; typed adapters prepare before decoding.
	if ctx != nil {
		ctx = context.WithValue(ctx, originalToolArgsKey{}, p.original)
		if p.final != nil {
			ctx = context.WithValue(ctx, preparedTypedArgsKey{}, p.final)
		}
	}
	content, err := p.tool.Handler(ctx, bytes.Clone(p.normalized), deps)
	if err != nil && content.IsEmpty() {
		content = llm.TextContent(formatToolErrorDiagnostic("Tool execution failed", err, toolDiagnosticDefaultAction))
	}
	if err == nil && content.IsEmpty() {
		content = llm.TextContent("Warning: tool returned no output.")
		UpsertToolResultMetadata(ctx, map[string]any{"tool_warning": "handler returned empty content"})
	}
	return content, err
}

func formatToolErrorDiagnostic(summary string, err error, action string) string {
	if err != nil {
		if detail := strings.TrimSpace(err.Error()); isSeverityActionDiagnostic(detail) {
			return detail
		} else if detail != "" {
			detail = strings.Join(strings.Fields(detail), " ")
			summary = strings.TrimSpace(summary)
			if summary == "" {
				summary = detail
			} else {
				summary = fmt.Sprintf("%s (%s)", summary, detail)
			}
		}
	}
	summary = strings.TrimSpace(summary)
	if summary == "" {
		summary = "Tool execution failed"
	}
	action = strings.TrimSpace(action)
	if action == "" {
		action = toolDiagnosticDefaultAction
	}
	return fmt.Sprintf("[ERROR] %s - %s", summary, action)
}

func isSeverityActionDiagnostic(text string) bool {
	text = strings.TrimSpace(text)
	if !strings.HasPrefix(text, "[") {
		return false
	}
	end := strings.Index(text, "]")
	if end <= 1 {
		return false
	}
	severity := strings.ToUpper(strings.TrimSpace(text[1:end]))
	switch severity {
	case "INFO", "WARN", "ERROR":
	default:
		return false
	}
	body := strings.TrimSpace(text[end+1:])
	if strings.Contains(body, " - ") {
		return true
	}
	return strings.Contains(body, "stage=") && strings.Contains(body, "action=")
}

type schemaRepairOptions struct {
	StripUnknown bool
	ToolName     string
}

func repairJSONKeysBySchema(schema map[string]any, raw []byte) ([]byte, bool) {
	repaired, ok, err := repairJSONKeysBySchemaWithOptions(schema, raw, schemaRepairOptions{})
	if err != nil {
		return nil, false
	}
	return repaired, ok
}

// Fixed reasons for rejected schema-key repairs. They never include argument
// values, key names or schema text.
const (
	ambiguousArgsReasonTarget  = "ambiguous_target"
	ambiguousArgsReasonSources = "multiple_sources"
)

// errAmbiguousToolArguments marks a schema-key repair that had more than one
// possible interpretation. Such input is rejected instead of guessing.
var errAmbiguousToolArguments = errors.New("ambiguous tool arguments")

type ambiguousToolArgumentsError struct {
	Reason string
}

func (e *ambiguousToolArgumentsError) Error() string {
	return "ambiguous tool arguments (reason=" + e.Reason + "): a non-exact argument name matches more than one schema field or several aliases target the same field; resend using the exact schema field names"
}

func (e *ambiguousToolArgumentsError) Is(target error) bool {
	return target == errAmbiguousToolArguments
}

// repairJSONKeysBySchemaWithOptions returns the repaired payload when a unique
// repair exists. An ambiguous repair returns errAmbiguousToolArguments and no
// payload; the input bytes are never modified.
func repairJSONKeysBySchemaWithOptions(schema map[string]any, raw []byte, opts schemaRepairOptions) ([]byte, bool, error) {
	if len(raw) == 0 || schema == nil {
		return nil, false, nil
	}
	var v any
	if err := json.Unmarshal(raw, &v); err != nil {
		return nil, false, nil
	}
	m, ok := v.(map[string]any)
	if !ok {
		return nil, false, nil
	}

	repaired, changed, err := repairObjectBySchema(m, schema, opts)
	if err != nil {
		return nil, false, err
	}
	if !changed {
		return nil, false, nil
	}
	b, err := json.Marshal(repaired)
	if err != nil {
		return nil, false, nil
	}
	return b, true, nil
}

// objectKeyMatcher maps non-exact argument names to schema properties. Each
// normalized token keeps the set of distinct properties it can reach; a token
// that reaches several properties is ambiguous and never resolved by map
// iteration order, rule order, sorting or name length. Exact property names
// always match themselves, even when other properties collide after
// normalization.
type objectKeyMatcher struct {
	expected        map[string]struct{}
	expectedNoDelim map[string]map[string]struct{}
	aliasByNoDelim  map[string]map[string]struct{}
}

func addMatcherTarget(index map[string]map[string]struct{}, token, target string) {
	if token == "" {
		// Punctuation-only or non-ASCII names share the empty token; they can
		// match exactly but never through normalization.
		return
	}
	targets := index[token]
	if targets == nil {
		targets = map[string]struct{}{}
		index[token] = targets
	}
	targets[target] = struct{}{}
}

func newObjectKeyMatcher(props map[string]any, toolName string) objectKeyMatcher {
	matcher := objectKeyMatcher{
		expected:        map[string]struct{}{},
		expectedNoDelim: map[string]map[string]struct{}{},
		aliasByNoDelim:  map[string]map[string]struct{}{},
	}
	for k := range props {
		kk := strings.TrimSpace(k)
		if kk == "" {
			continue
		}
		matcher.expected[kk] = struct{}{}
		addMatcherTarget(matcher.expectedNoDelim, normalizeKeyNoDelims(kk), kk)
	}
	for k := range props {
		for _, alias := range aliasKeysForExpected(toolName, k) {
			if alias == "" {
				continue
			}
			addMatcherTarget(matcher.aliasByNoDelim, normalizeKeyNoDelims(alias), k)
		}
	}
	if len(props) == 1 {
		for k := range props {
			for _, alias := range singleFieldAliases() {
				if alias == "" {
					continue
				}
				addMatcherTarget(matcher.aliasByNoDelim, normalizeKeyNoDelims(alias), k)
			}
			break
		}
	}
	return matcher
}

// matchedTargets returns every distinct schema property a non-exact key can
// reach through the supported normalization, alias and candidate rules.
func (m objectKeyMatcher) matchedTargets(k string) map[string]struct{} {
	targets := map[string]struct{}{}
	collect := func(token string) {
		if token == "" {
			return
		}
		for target := range m.expectedNoDelim[token] {
			targets[target] = struct{}{}
		}
		for target := range m.aliasByNoDelim[token] {
			targets[target] = struct{}{}
		}
	}
	collect(normalizeKeyNoDelims(k))
	if cand := normalizeCandidateKey(k); cand != "" {
		collect(normalizeKeyNoDelims(cand))
	}
	return targets
}

// keyMatch classifies one input key: exact, unique repair target, unknown or
// ambiguous.
type keyMatch int

const (
	keyMatchUnknown keyMatch = iota
	keyMatchExact
	keyMatchUnique
	keyMatchAmbiguous
)

func (m objectKeyMatcher) classifyKey(k string) (string, keyMatch) {
	if _, ok := m.expected[k]; ok {
		return k, keyMatchExact
	}
	targets := m.matchedTargets(k)
	switch len(targets) {
	case 0:
		return "", keyMatchUnknown
	case 1:
		for target := range targets {
			return target, keyMatchUnique
		}
	}
	return "", keyMatchAmbiguous
}

// canonicalKey reports the unique property a key maps to. Ambiguous keys are
// not canonicalized.
func (m objectKeyMatcher) canonicalKey(k string) (string, bool) {
	target, match := m.classifyKey(k)
	return target, match == keyMatchExact || match == keyMatchUnique
}

func repairBySchemaValue(v any, schema map[string]any, opts schemaRepairOptions) (any, bool, error) {
	if schema == nil {
		return v, false, nil
	}
	switch vv := v.(type) {
	case map[string]any:
		return repairObjectBySchema(vv, schema, opts)
	case []any:
		return repairArrayBySchema(vv, schema, opts)
	default:
		return v, false, nil
	}
}

// repairArrayBySchema never mutates in; a changed array is returned as a copy.
func repairArrayBySchema(in []any, schema map[string]any, opts schemaRepairOptions) ([]any, bool, error) {
	if len(in) == 0 {
		return in, false, nil
	}
	itemSchema, ok := schema["items"].(map[string]any)
	if !ok || itemSchema == nil {
		return in, false, nil
	}
	var out []any
	for i, item := range in {
		repaired, itemChanged, err := repairBySchemaValue(item, itemSchema, opts)
		if err != nil {
			return nil, false, err
		}
		if !itemChanged {
			continue
		}
		if out == nil {
			out = append([]any(nil), in...)
		}
		out[i] = repaired
	}
	if out == nil {
		return in, false, nil
	}
	return out, true, nil
}

// repairObjectBySchema plans every key move against the original input before
// building an owned result, so input order cannot choose between candidates:
//   - an exact schema key keeps its value and redundant aliases of it are dropped;
//   - a single unique alias for an absent property is renamed;
//   - several aliases for one absent property, or one key matching several
//     properties, reject the whole repair (even when values are equal).
//
// in is never mutated; any nested rejection rejects the whole object.
func repairObjectBySchema(in map[string]any, schema map[string]any, opts schemaRepairOptions) (map[string]any, bool, error) {
	props, _ := schema["properties"].(map[string]any)
	matcher := newObjectKeyMatcher(props, opts.ToolName)
	_, additionalSchema := schemaAllowsAdditionalProperties(schema)
	stripUnknown := opts.StripUnknown
	// Keep map-like payloads (no fixed properties + typed additionalProperties).
	if len(props) == 0 && additionalSchema != nil {
		stripUnknown = false
	}

	aliasSources := map[string]int{}
	moves := map[string]string{}
	var unknown []string
	for k := range in {
		target, match := matcher.classifyKey(k)
		switch match {
		case keyMatchExact:
		case keyMatchUnique:
			moves[k] = target
			aliasSources[target]++
		case keyMatchAmbiguous:
			return nil, false, &ambiguousToolArgumentsError{Reason: ambiguousArgsReasonTarget}
		default:
			unknown = append(unknown, k)
		}
	}
	for target, sources := range aliasSources {
		if _, exact := in[target]; exact {
			continue
		}
		if sources > 1 {
			return nil, false, &ambiguousToolArgumentsError{Reason: ambiguousArgsReasonSources}
		}
	}

	out := make(map[string]any, len(in))
	changed := false
	for k, v := range in {
		if target, ok := moves[k]; ok {
			changed = true
			if _, exact := in[target]; exact {
				// The exact key keeps priority; its redundant alias is dropped.
				continue
			}
			out[target] = v
			continue
		}
		out[k] = v
	}
	if stripUnknown {
		for _, k := range unknown {
			delete(out, k)
			changed = true
		}
	}

	for k, v := range out {
		childSchema := map[string]any(nil)
		if propSchema, ok := props[k].(map[string]any); ok {
			childSchema = propSchema
		} else if additionalSchema != nil {
			childSchema = additionalSchema
		}
		if childSchema == nil {
			continue
		}
		repaired, childChanged, err := repairBySchemaValue(v, childSchema, opts)
		if err != nil {
			return nil, false, err
		}
		if !childChanged {
			continue
		}
		out[k] = repaired
		changed = true
	}
	if !changed {
		return in, false, nil
	}
	return out, true, nil
}

func schemaAllowsAdditionalProperties(schema map[string]any) (bool, map[string]any) {
	apAny, ok := schema["additionalProperties"]
	if !ok {
		return true, nil
	}
	switch ap := apAny.(type) {
	case bool:
		return ap, nil
	case map[string]any:
		return true, ap
	default:
		return true, nil
	}
}

func normalizeCandidateKey(k string) string {
	k = strings.TrimSpace(k)
	if k == "" {
		return ""
	}
	low := strings.ToLower(k)
	// Collapse duplicated whitespace-separated tokens: "content content" -> "content".
	parts := strings.Fields(low)
	if len(parts) > 1 {
		same := true
		for i := 1; i < len(parts); i++ {
			if parts[i] != parts[0] {
				same = false
				break
			}
		}
		if same {
			return parts[0]
		}
		// Drop tokens that are substrings of longer tokens (e.g. "file filepath" -> "filepath").
		filtered := make([]string, 0, len(parts))
		for i, p := range parts {
			keep := true
			for j, q := range parts {
				if i == j {
					continue
				}
				if strings.Contains(q, p) && len(q) >= len(p) {
					keep = false
					break
				}
			}
			if keep {
				filtered = append(filtered, p)
			}
		}
		if len(filtered) == 1 {
			return filtered[0]
		}
		if len(filtered) > 1 {
			return strings.Join(filtered, "_")
		}
		return strings.Join(parts, "_")
	}
	// Replace common separators.
	low = strings.ReplaceAll(low, "-", "_")
	low = strings.ReplaceAll(low, " ", "_")
	return low
}

func normalizeKeyNoDelims(k string) string {
	k = strings.ToLower(strings.TrimSpace(k))
	if k == "" {
		return ""
	}
	// Keep only [a-z0-9].
	return strings.Map(func(r rune) rune {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') {
			return r
		}
		return -1
	}, k)
}

func aliasKeysForExpected(toolName, key string) []string {
	switch normalizeKeyNoDelims(key) {
	case "filepath":
		return []string{"path", "file", "filename", "file_path"}
	case "path":
		return []string{"filepath", "file_path", "dir", "directory", "folder"}
	case "command":
		return []string{"cmd", "shell", "bash", "sh"}
	case "content":
		return []string{"contents", "data", "text", "body"}
	case "pattern":
		return []string{"query", "regex", "search", "match"}
	case "url":
		return []string{"uri", "link"}
	case "oldstring":
		return []string{"old", "from", "before"}
	case "newstring":
		return []string{"new", "to", "after", "replacement"}
	case "patch":
		return []string{"diff"}
	case "offset":
		aliases := []string{"start"}
		if supportsLineOffsetAliases(toolName) {
			aliases = append(aliases, "line", "start_line")
		}
		return aliases
	case "limit":
		return []string{"lines", "max_lines", "count"}
	default:
		return nil
	}
}

func supportsLineOffsetAliases(toolName string) bool {
	switch NormalizeToolName(toolName) {
	case "read":
		return true
	default:
		return false
	}
}

func singleFieldAliases() []string {
	return []string{"input", "args", "argument", "value", "text", "data"}
}

// repairLooseJSONObject tries to repair a JSON-object-like string where some scalar
// values are unquoted (for example {"path":/tmp}). It applies conservative heuristics
// and validates the repaired payload shape against the tool schema when available.
func repairLooseJSONObject(raw string, schema map[string]any) ([]byte, bool) {
	raw = strings.TrimSpace(raw)
	if !strings.HasPrefix(raw, "{") || !strings.HasSuffix(raw, "}") {
		return nil, false
	}

	if v, err := decodeJSONValueStrict(raw); err == nil {
		obj, ok := v.(map[string]any)
		if !ok || !looseRepairSchemaCompatible(obj, schema) {
			return nil, false
		}
		return []byte(raw), true
	}

	repaired, changed, ok := quoteLooseObjectScalars(raw)
	if !ok || !changed {
		return nil, false
	}

	var parsed any
	if err := json.Unmarshal(repaired, &parsed); err != nil {
		return nil, false
	}
	obj, ok := parsed.(map[string]any)
	if !ok || !looseRepairSchemaCompatible(obj, schema) {
		return nil, false
	}
	return repaired, true
}

func quoteLooseObjectScalars(raw string) ([]byte, bool, bool) {
	out := make([]byte, 0, len(raw)+16)
	inStr := false
	esc := false
	changed := false

	for i := 0; i < len(raw); {
		c := raw[i]
		if inStr {
			out = append(out, c)
			if esc {
				esc = false
				i++
				continue
			}
			if c == '\\' {
				esc = true
				i++
				continue
			}
			if c == '"' {
				inStr = false
			}
			i++
			continue
		}

		if c == '"' {
			inStr = true
			out = append(out, c)
			i++
			continue
		}

		out = append(out, c)
		i++
		if c != ':' {
			continue
		}

		for i < len(raw) {
			s := raw[i]
			if s == ' ' || s == '\n' || s == '\r' || s == '\t' {
				out = append(out, s)
				i++
				continue
			}
			break
		}
		if i >= len(raw) {
			return nil, false, false
		}
		if !shouldRepairLooseScalar(raw[i:]) {
			continue
		}

		end, token, ok := readLooseScalarToken(raw, i)
		if !ok {
			return nil, false, false
		}
		if !isSafeLooseScalarToken(token) {
			continue
		}
		quoted, err := json.Marshal(token)
		if err != nil {
			return nil, false, false
		}
		out = append(out, quoted...)
		i = end
		changed = true
	}

	if inStr {
		return nil, false, false
	}
	return out, changed, true
}

func shouldRepairLooseScalar(raw string) bool {
	if raw == "" {
		return false
	}
	n := raw[0]
	if n == '"' || n == '{' || n == '[' || n == '-' || (n >= '0' && n <= '9') {
		return false
	}
	if n == ',' || n == '}' || n == ']' {
		return false
	}
	if hasJSONLiteralPrefix(raw) {
		return false
	}
	return true
}

func hasJSONLiteralPrefix(raw string) bool {
	for _, lit := range []string{"true", "false", "null"} {
		if !strings.HasPrefix(raw, lit) {
			continue
		}
		if len(raw) == len(lit) {
			return true
		}
		if isJSONValueDelimiter(raw[len(lit)]) {
			return true
		}
	}
	return false
}

func isJSONValueDelimiter(b byte) bool {
	switch b {
	case ' ', '\n', '\r', '\t', ',', '}', ']':
		return true
	default:
		return false
	}
}

func readLooseScalarToken(raw string, start int) (int, string, bool) {
	i := start
	for i < len(raw) {
		c := raw[i]
		if c == ',' || c == '}' || c == ']' {
			break
		}
		if c == '"' || c == '{' || c == '[' {
			return 0, "", false
		}
		i++
	}
	token := strings.TrimSpace(raw[start:i])
	if token == "" {
		return 0, "", false
	}
	return i, token, true
}

func isSafeLooseScalarToken(token string) bool {
	token = strings.TrimSpace(token)
	if token == "" {
		return false
	}
	if strings.ContainsAny(token, "\"'{}[]") {
		return false
	}
	if token == "-" {
		return false
	}
	return true
}

func looseRepairSchemaCompatible(value any, schema map[string]any) bool {
	if schema == nil || len(schema) == 0 {
		return true
	}
	return schemaAllowsValue(value, schema)
}

func schemaAllowsValue(value any, schema map[string]any) bool {
	if schema == nil || len(schema) == 0 {
		return true
	}
	if !schemaAllowsType(value, schema) {
		return false
	}

	switch v := value.(type) {
	case map[string]any:
		props, _ := schema["properties"].(map[string]any)
		allowUnknown, additionalSchema := schemaAllowsAdditionalProperties(schema)
		for k, child := range v {
			if propSchema, ok := props[k].(map[string]any); ok {
				if !schemaAllowsValue(child, propSchema) {
					return false
				}
				continue
			}
			if additionalSchema != nil {
				if !schemaAllowsValue(child, additionalSchema) {
					return false
				}
				continue
			}
			if !allowUnknown {
				return false
			}
		}
	case []any:
		itemSchema, _ := schema["items"].(map[string]any)
		if itemSchema == nil {
			return true
		}
		for _, child := range v {
			if !schemaAllowsValue(child, itemSchema) {
				return false
			}
		}
	}

	return true
}

func schemaAllowsType(value any, schema map[string]any) bool {
	types := schemaTypeSet(schema)
	if len(types) == 0 {
		return true
	}
	vType := jsonValueType(value)
	if vType == "" {
		return false
	}
	if _, ok := types[vType]; ok {
		return true
	}
	if vType == "integer" {
		_, ok := types["number"]
		return ok
	}
	return false
}

func schemaTypeSet(schema map[string]any) map[string]struct{} {
	out := map[string]struct{}{}
	t, ok := schema["type"]
	if !ok {
		return out
	}
	switch vv := t.(type) {
	case string:
		v := strings.ToLower(strings.TrimSpace(vv))
		if v != "" {
			out[v] = struct{}{}
		}
	case []any:
		for _, item := range vv {
			s, ok := item.(string)
			if !ok {
				continue
			}
			v := strings.ToLower(strings.TrimSpace(s))
			if v != "" {
				out[v] = struct{}{}
			}
		}
	}
	return out
}

func jsonValueType(value any) string {
	switch v := value.(type) {
	case nil:
		return "null"
	case map[string]any:
		return "object"
	case []any:
		return "array"
	case string:
		return "string"
	case bool:
		return "boolean"
	case float64:
		if v == math.Trunc(v) {
			return "integer"
		}
		return "number"
	default:
		return ""
	}
}

// Func creates a tool from an Args struct and a handler.
// Args should be a struct type with json tags.
// Plain unknown-field decode errors may receive schema key repair before business
// execution. Custom decoders and business handlers are never retried.
func Func[Args any](name, description string, fn func(ctx context.Context, args Args, deps *Container) (any, error)) Tool {
	schema := SchemaFor[Args]()
	binding := newTypedArgsBinding[Args](name, schema)
	return Tool{
		Name:          name,
		Description:   description,
		EphemeralKeep: 0,
		Schema:        schema,
		typed:         binding,
		Handler: func(ctx context.Context, raw json.RawMessage, deps *Container) (llm.Content, error) {
			a, err := consumeTypedArgs[Args](ctx, binding, name, schema, raw)
			if err != nil {
				return llm.TextContent(formatToolErrorDiagnostic("Invalid tool arguments", err, toolDiagnosticInvalidArgsAction)), err
			}
			res, err := fn(ctx, a, deps)
			if err != nil {
				return llm.TextContent(formatToolErrorDiagnostic("Tool execution failed", err, toolDiagnosticDefaultAction)), err
			}
			return SerializeResult(res)
		},
	}
}

func (t Tool) WithEphemeralKeep(n int) Tool {
	t.EphemeralKeep = n
	return t
}
