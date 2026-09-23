package tools

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"
)

type issue153ContentArgs struct {
	Content string `json:"content"`
}

// executeIssue153 runs input through the real Tool.Execute → PreparedCall →
// Func adapter chain and reports business calls, error and published metadata.
func executeIssue153[Args any](t *testing.T, name, raw string, business func(Args)) (int, error, map[string]any) {
	t.Helper()
	calls := 0
	tool := Func[Args](name, "fixture", func(_ context.Context, args Args, _ *Container) (any, error) {
		calls++
		if business != nil {
			business(args)
		}
		return "ok", nil
	})
	ctx := WithToolResultMetadata(context.Background())
	_, err := tool.Execute(ctx, raw, NewContainer())
	return calls, err, ToolResultMetadataSnapshot(ctx)
}

func requireNoSchemaKeySuccess(t *testing.T, meta map[string]any) {
	t.Helper()
	if kind, _ := meta["args_repair_kind"].(string); strings.Contains(kind, "schema_key") {
		t.Fatalf("rejected repair published schema_key metadata: %v", meta)
	}
}

// A01/A02: two aliases for one absent property are rejected, even with equal
// values, for every input key order.
func TestIssue153ConflictingAliasesAreRejected(t *testing.T) {
	for _, raw := range []string{
		`{"text":"A","body":"B"}`,
		`{"body":"B","text":"A"}`,
		`{"text":"A","body":"A"}`,
	} {
		t.Run(raw, func(t *testing.T) {
			calls, err, meta := executeIssue153[issue153ContentArgs](t, "fixture", raw, nil)
			if !errors.Is(err, errAmbiguousToolArguments) || calls != 0 {
				t.Fatalf("ambiguous repair executed: calls=%d err=%v", calls, err)
			}
			var ambiguous *ambiguousToolArgumentsError
			if !errors.As(err, &ambiguous) || ambiguous.Reason != ambiguousArgsReasonSources {
				t.Fatalf("err = %#v, want multiple_sources", err)
			}
			requireNoSchemaKeySuccess(t, meta)
		})
	}
}

type issue153CollidingArgs struct {
	Snake string `json:"foo_bar"`
	Camel string `json:"fooBar"`
}

type issue153CollidingArgsReordered struct {
	Camel string `json:"fooBar"`
	Snake string `json:"foo_bar"`
}

// A03: FOO-BAR fails standard decode, then normalizes onto both colliding
// properties; the typed chain must reject it for either schema order.
func TestIssue153CollidingPropertiesRejectNonExactKey(t *testing.T) {
	calls, err, meta := executeIssue153[issue153CollidingArgs](t, "fixture", `{"FOO-BAR":"x"}`, nil)
	if !errors.Is(err, errAmbiguousToolArguments) || calls != 0 {
		t.Fatalf("calls=%d err=%v", calls, err)
	}
	requireNoSchemaKeySuccess(t, meta)
	calls, err, _ = executeIssue153[issue153CollidingArgsReordered](t, "fixture", `{"FOO-BAR":"x"}`, nil)
	if !errors.Is(err, errAmbiguousToolArguments) || calls != 0 {
		t.Fatalf("reordered schema: calls=%d err=%v", calls, err)
	}
	var ambiguous *ambiguousToolArgumentsError
	if !errors.As(err, &ambiguous) || ambiguous.Reason != ambiguousArgsReasonTarget {
		t.Fatalf("err = %#v, want ambiguous_target", err)
	}

	// The matcher/helper level rejects the direct FOOBAR form as well.
	schema := SchemaFor[issue153CollidingArgs]()
	for _, raw := range []string{`{"FOOBAR":"x"}`, `{"FOO-BAR":"x"}`, `{"foo bar":"x"}`} {
		if repaired, ok, err := repairJSONKeysBySchemaWithOptions(schema, []byte(raw), schemaRepairOptions{StripUnknown: true}); !errors.Is(err, errAmbiguousToolArguments) || ok || repaired != nil {
			t.Fatalf("%s: repaired=%s ok=%v err=%v", raw, repaired, ok, err)
		}
	}
}

// A04: a schema with colliding properties still accepts exact input.
func TestIssue153CollidingPropertiesKeepExactInput(t *testing.T) {
	var got issue153CollidingArgs
	calls, err, _ := executeIssue153[issue153CollidingArgs](t, "fixture", `{"foo_bar":"S","fooBar":"C"}`, func(a issue153CollidingArgs) { got = a })
	if err != nil || calls != 1 || got.Snake != "S" || got.Camel != "C" {
		t.Fatalf("calls=%d err=%v args=%+v", calls, err, got)
	}
	// Exact keys also survive the repair path when an unknown key forces it.
	var repaired issue153CollidingArgs
	calls, err, _ = executeIssue153[issue153CollidingArgs](t, "fixture", `{"foo_bar":"S","fooBar":"C","extra":1}`, func(a issue153CollidingArgs) { repaired = a })
	if err != nil || calls != 1 || repaired.Snake != "S" || repaired.Camel != "C" {
		t.Fatalf("repair path: calls=%d err=%v args=%+v", calls, err, repaired)
	}
}

// A05: Go's case-insensitive first decode keeps accepting FOOBAR without
// entering schema repair; this fix does not change standard decoding.
func TestIssue153StandardDecodeFastPathUnchanged(t *testing.T) {
	var got issue153CollidingArgs
	calls, err, meta := executeIssue153[issue153CollidingArgs](t, "fixture", `{"FOOBAR":"x"}`, func(a issue153CollidingArgs) { got = a })
	if err != nil || calls != 1 || got.Camel != "x" || got.Snake != "" {
		t.Fatalf("calls=%d err=%v args=%+v", calls, err, got)
	}
	requireNoSchemaKeySuccess(t, meta)
}

// A06: an exact canonical key keeps priority over its aliases.
func TestIssue153ExactCanonicalKeepsPriority(t *testing.T) {
	for _, raw := range []string{`{"content":"C","text":"A","body":"B"}`, `{"body":"B","text":"A","content":"C"}`} {
		var got issue153ContentArgs
		calls, err, meta := executeIssue153[issue153ContentArgs](t, "fixture", raw, func(a issue153ContentArgs) { got = a })
		if err != nil || calls != 1 || got.Content != "C" {
			t.Fatalf("%s: calls=%d err=%v args=%+v", raw, calls, err, got)
		}
		if kind, _ := meta["args_repair_kind"].(string); !strings.Contains(kind, "schema_key") {
			t.Fatalf("%s: metadata=%v", raw, meta)
		}
	}
}

// A07: a canonical value with the wrong type is never replaced by a
// valid-looking alias.
func TestIssue153InvalidCanonicalIsNotReplacedByAlias(t *testing.T) {
	for _, raw := range []string{`{"content":123,"text":"ok"}`, `{"text":"ok","content":123}`} {
		calls, err, meta := executeIssue153[issue153ContentArgs](t, "fixture", raw, nil)
		if err == nil || calls != 0 {
			t.Fatalf("%s: calls=%d err=%v", raw, calls, err)
		}
		requireNoSchemaKeySuccess(t, meta)
	}
}

// A08: unique legacy aliases keep working with their metadata.
func TestIssue153UniqueAliasCompatibility(t *testing.T) {
	var content issue153ContentArgs
	calls, err, meta := executeIssue153[issue153ContentArgs](t, "fixture", `{"text":"A"}`, func(a issue153ContentArgs) { content = a })
	if err != nil || calls != 1 || content.Content != "A" {
		t.Fatalf("calls=%d err=%v args=%+v", calls, err, content)
	}
	if kind, _ := meta["args_repair_kind"].(string); !strings.Contains(kind, "schema_key") || meta["args_raw"] != `{"text":"A"}` {
		t.Fatalf("metadata=%v", meta)
	}

	type readArgs struct {
		Offset int `json:"offset"`
	}
	var read readArgs
	calls, err, _ = executeIssue153[readArgs](t, "read", `{"line":12}`, func(a readArgs) { read = a })
	if err != nil || calls != 1 || read.Offset != 12 {
		t.Fatalf("read offset alias: calls=%d err=%v args=%+v", calls, err, read)
	}

	type writeArgs struct {
		FilePath string `json:"file_path"`
		Content  string `json:"content"`
	}
	var write writeArgs
	calls, err, meta = executeIssue153[writeArgs](t, "write", "```json\n{\"path\":\"a.txt\",\"contents\":\"x\"}\n```", func(a writeArgs) { write = a })
	if err != nil || calls != 1 || write.FilePath != "a.txt" || write.Content != "x" {
		t.Fatalf("fenced aliases: calls=%d err=%v args=%+v", calls, err, write)
	}
	if kind, _ := meta["args_repair_kind"].(string); !strings.Contains(kind, "schema_key") {
		t.Fatalf("fenced metadata=%v", meta)
	}
}

// A09: conflicts inside nested objects and array items reject the whole call.
func TestIssue153NestedConflictsRejectWholeRepair(t *testing.T) {
	type nestedArgs struct {
		Payload issue153ContentArgs   `json:"payload"`
		Items   []issue153ContentArgs `json:"items"`
	}
	for _, raw := range []string{
		`{"payload":{"text":"A","body":"B"}}`,
		`{"items":[{"content":"ok"},{"text":"A","body":"B"}]}`,
		`{"items":[{"text":"A"}],"payload":{"FOO":"x","text":"A","data":"B"}}`,
	} {
		calls, err, meta := executeIssue153[nestedArgs](t, "fixture", raw, nil)
		if !errors.Is(err, errAmbiguousToolArguments) || calls != 0 {
			t.Fatalf("%s: calls=%d err=%v", raw, calls, err)
		}
		requireNoSchemaKeySuccess(t, meta)
	}
	var got nestedArgs
	calls, err, _ := executeIssue153[nestedArgs](t, "fixture", `{"items":[{"text":"A"},{"body":"B"}],"payload":{"data":"C"}}`, func(a nestedArgs) { got = a })
	if err != nil || calls != 1 || got.Payload.Content != "C" || len(got.Items) != 2 || got.Items[0].Content != "A" || got.Items[1].Content != "B" {
		t.Fatalf("unique nested aliases: calls=%d err=%v args=%+v", calls, err, got)
	}
}

// A10: map-like additionalProperties keep free keys; Unicode and
// punctuation-only names never match through the empty normalized token.
func TestIssue153MapLikeAndEmptyTokenKeys(t *testing.T) {
	type labelArgs struct {
		Content string            `json:"content"`
		Labels  map[string]string `json:"labels"`
	}
	var got labelArgs
	calls, err, _ := executeIssue153[labelArgs](t, "fixture", `{"contents":"x","labels":{"text":"A","body":"B","FOO-BAR":"C"}}`, func(a labelArgs) { got = a })
	if err != nil || calls != 1 || got.Content != "x" || !reflect.DeepEqual(got.Labels, map[string]string{"text": "A", "body": "B", "FOO-BAR": "C"}) {
		t.Fatalf("calls=%d err=%v args=%+v", calls, err, got)
	}

	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"名称": map[string]any{"type": "string"},
			"标题": map[string]any{"type": "string"},
		},
	}
	repaired, ok, err := repairJSONKeysBySchemaWithOptions(schema, []byte(`{"名称":"a","名字":"b","--":"c"}`), schemaRepairOptions{StripUnknown: true})
	if err != nil || !ok {
		t.Fatalf("ok=%v err=%v", ok, err)
	}
	var m map[string]any
	if err := json.Unmarshal(repaired, &m); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(m, map[string]any{"名称": "a"}) {
		t.Fatalf("repaired=%v: empty-token keys must stay unknown", m)
	}
	matcher := newObjectKeyMatcher(schema["properties"].(map[string]any), "")
	for _, key := range []string{"名字", "--", ""} {
		if target, match := matcher.classifyKey(key); match != keyMatchUnknown {
			t.Fatalf("key %q matched %q (%v)", key, target, match)
		}
	}
}

type issue153CountingDecoder struct {
	Value string
}

var issue153DecoderCalls int

func (d *issue153CountingDecoder) UnmarshalJSON(b []byte) error {
	issue153DecoderCalls++
	return json.Unmarshal(b, &d.Value)
}

// A11: custom decoders are decoded at most once and never re-tried with a
// guessed interpretation.
func TestIssue153CustomDecoderIsNotRetried(t *testing.T) {
	type customArgs struct {
		Content issue153CountingDecoder `json:"content"`
	}
	issue153DecoderCalls = 0
	calls, err, _ := executeIssue153[customArgs](t, "fixture", `{"content":"C","text":"A","body":"B"}`, nil)
	if err == nil || calls != 0 {
		t.Fatalf("calls=%d err=%v", calls, err)
	}
	if issue153DecoderCalls > 1 {
		t.Fatalf("custom decoder ran %d times", issue153DecoderCalls)
	}
}

// A12: a business error after a valid unique repair is not re-executed.
func TestIssue153BusinessErrorIsNotReExecuted(t *testing.T) {
	calls := 0
	tool := Func[issue153ContentArgs]("fixture", "fixture", func(context.Context, issue153ContentArgs, *Container) (any, error) {
		calls++
		return nil, errors.New("business failed with unknown field")
	})
	if _, err := tool.Execute(context.Background(), `{"text":"A"}`, NewContainer()); err == nil || calls != 1 {
		t.Fatalf("calls=%d err=%v", calls, err)
	}
}

func cloneIssue153JSON(t *testing.T, v any) any {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	var out any
	if err := json.Unmarshal(b, &out); err != nil {
		t.Fatal(err)
	}
	return out
}

// A13: the recursive helper never mutates the caller's input graph, whether
// it rejects or repairs.
func TestIssue153RepairDoesNotMutateInputGraph(t *testing.T) {
	type nestedArgs struct {
		Payload issue153ContentArgs   `json:"payload"`
		Items   []issue153ContentArgs `json:"items"`
	}
	schema := SchemaFor[nestedArgs]()
	for _, raw := range []string{
		`{"items":[{"text":"A"},{"text":"A","body":"B"}],"payload":{"data":"C"}}`,
		`{"items":[{"text":"A"},{"body":"B"}],"payload":{"data":"C"}}`,
	} {
		var input map[string]any
		if err := json.Unmarshal([]byte(raw), &input); err != nil {
			t.Fatal(err)
		}
		before := cloneIssue153JSON(t, input)
		_, _, _ = repairObjectBySchema(input, schema, schemaRepairOptions{StripUnknown: true})
		if !reflect.DeepEqual(cloneIssue153JSON(t, input), before) {
			t.Fatalf("%s: input mutated to %v", raw, input)
		}
	}
}

// Ambiguity diagnostics use fixed reasons and never echo values or keys.
func TestIssue153AmbiguityErrorCarriesNoInput(t *testing.T) {
	const canary = "CANARY_153_VALUE"
	_, err, _ := executeIssue153[issue153ContentArgs](t, "fixture", `{"text":"`+canary+`","body":"B"}`, nil)
	if err == nil || strings.Contains(err.Error(), canary) || strings.Contains(err.Error(), "text") || strings.Contains(err.Error(), "body") {
		t.Fatalf("err = %v", err)
	}
}
