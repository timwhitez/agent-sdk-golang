package agent

import (
	"context"
	"errors"
	"reflect"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/agent/compaction"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

// #172: the atomic read names the publication its history carries, and a
// conditional publication derived from it is refused when another
// publisher moved the revision in between. No mutation is applied and the
// refusal is typed; a retry from the fresh read replaces exactly that one.
func TestReplaceHistoryCheckedIfRevisionRefusesInterveningPublication(t *testing.T) {
	ag, err := New(Config{LLM: plainAnswerModel{}})
	if err != nil {
		t.Fatal(err)
	}
	first, err := ag.ReplaceHistoryCheckedRevision([]llm.Message{llm.NewSystemMessage("rules v1")})
	if err != nil {
		t.Fatal(err)
	}
	read, revision := ag.MessagesWithHostPublication()
	if revision != first.Revision || len(read) != 1 || read[0].Content.PlainText() != "rules v1" {
		t.Fatalf("read revision=%d messages=%+v, want %d", revision, read, first.Revision)
	}
	derived := append(llm.CloneMessages(read), llm.Message{Role: llm.RoleSystem, Name: "memory", Content: llm.TextContent("m")})

	// Another publisher between the read and the conditional publication.
	intervening, err := ag.ReplaceHistoryCheckedRevision([]llm.Message{llm.NewSystemMessage("rules v2")})
	if err != nil {
		t.Fatal(err)
	}
	publication, err := ag.ReplaceHistoryCheckedIfRevision(revision, derived)
	var conflict *HistoryPublicationConflictError
	if !errors.Is(err, ErrHistoryPublicationConflict) || !errors.As(err, &conflict) || conflict.Expected != first.Revision || conflict.Current != intervening.Revision || publication != (HistoryPublication{}) {
		t.Fatalf("conditional publication=%+v err=%v", publication, err)
	}
	current, currentRevision := ag.MessagesWithHostPublication()
	if currentRevision != intervening.Revision || !reflect.DeepEqual(current, []llm.Message{llm.NewSystemMessage("rules v2")}) {
		t.Fatalf("refused publication mutated history: revision=%d messages=%+v", currentRevision, current)
	}

	// Retry from the fresh read: Replaced is exactly the expected revision.
	retried, err := ag.ReplaceHistoryCheckedIfRevision(currentRevision, append(current, llm.Message{Role: llm.RoleSystem, Name: "memory", Content: llm.TextContent("m")}))
	if err != nil || retried.Replaced != intervening.Revision || retried.Revision <= intervening.Revision {
		t.Fatalf("retry publication=%+v err=%v", retried, err)
	}
	if _, after := ag.MessagesWithHostPublication(); after != retried.Revision {
		t.Fatalf("revision after retry=%d, want %d", after, retried.Revision)
	}
}

// Zero is unknown and never compares equal: neither a zero expectation nor
// an unknown current publication may be treated as a match.
func TestReplaceHistoryCheckedIfRevisionRefusesUnknown(t *testing.T) {
	ag, err := New(Config{LLM: plainAnswerModel{}, InitialMessages: []llm.Message{llm.NewUserMessage("seed")}})
	if err != nil {
		t.Fatal(err)
	}
	if _, revision := ag.MessagesWithHostPublication(); revision != 0 {
		t.Fatalf("revision before any publication=%d", revision)
	}
	if _, err := ag.ReplaceHistoryCheckedIfRevision(0, []llm.Message{llm.NewSystemMessage("x")}); !errors.Is(err, ErrHistoryPublicationConflict) {
		t.Fatalf("zero expectation err=%v", err)
	}
	if got := ag.Messages(); !reflect.DeepEqual(got, []llm.Message{llm.NewUserMessage("seed")}) {
		t.Fatalf("refused publication mutated history: %+v", got)
	}

	// The SDK's configured prompt insertion makes the publication unknown;
	// a conditional publication against the earlier revision is refused.
	configured, err := New(Config{LLM: plainAnswerModel{}, SystemPrompt: "configured"})
	if err != nil {
		t.Fatal(err)
	}
	cleared, err := configured.ReplaceHistoryCheckedRevision(nil)
	if err != nil {
		t.Fatal(err)
	}
	_ = observePublications(t, configured, "run")
	if _, revision := configured.MessagesWithHostPublication(); revision != 0 {
		t.Fatalf("revision after an SDK-inserted prompt=%d", revision)
	}
	_, err = configured.ReplaceHistoryCheckedIfRevision(cleared.Revision, nil)
	var conflict *HistoryPublicationConflictError
	if !errors.As(err, &conflict) || conflict.Current != 0 {
		t.Fatalf("stale expectation after SDK change err=%v", err)
	}
}

// The admission check keeps its precedence: a manual compaction in progress
// is ErrActiveHistoryMutation even when the revision matches.
func TestReplaceHistoryCheckedIfRevisionKeepsAdmission(t *testing.T) {
	ag, err := New(Config{LLM: plainAnswerModel{}})
	if err != nil {
		t.Fatal(err)
	}
	published, err := ag.ReplaceHistoryCheckedRevision([]llm.Message{llm.NewSystemMessage("rules")})
	if err != nil {
		t.Fatal(err)
	}
	release, err := ag.beginManualCompaction(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	_, err = ag.ReplaceHistoryCheckedIfRevision(published.Revision, nil)
	release()
	if !errors.Is(err, ErrActiveHistoryMutation) {
		t.Fatalf("err=%v, want ErrActiveHistoryMutation", err)
	}
}

// A host-computed compaction commit returns the publication it recorded:
// the same revision the Frames built from its history report, replacing
// the publication the expected history carried. A refused commit returns
// no publication.
func TestCommitCompactionHistoryRevisionReturnsPublication(t *testing.T) {
	model := &relationScriptModel{steps: []func() (*llm.Completion, error){relationFinal}}
	ag := newRelationAgent(t, model, false)
	admission, err := ag.ReplaceHistoryCheckedRevision([]llm.Message{llm.NewSystemMessage("rules"), llm.NewUserMessage("a"), llm.NewAssistantMessage("b", nil)})
	if err != nil {
		t.Fatal(err)
	}
	source := ag.Messages()
	candidate := []llm.Message{llm.NewSystemMessage("rules"), llm.NewUserMessage("summary")}

	stale := llm.CloneMessages(source)
	stale[1].Name = "different"
	refused, refusedPublication, err := ag.CommitCompactionHistoryRevision(context.Background(), stale, candidate, compaction.Result{Compacted: true})
	if !errors.Is(err, ErrStaleCompactionHistory) || refused.Compacted || refusedPublication != (HistoryPublication{}) {
		t.Fatalf("stale commit result=%+v publication=%+v err=%v", refused, refusedPublication, err)
	}

	result, publication, err := ag.CommitCompactionHistoryRevision(context.Background(), source, candidate, compaction.Result{Compacted: true})
	if err != nil || !result.Compacted {
		t.Fatalf("commit result=%+v err=%v", result, err)
	}
	if publication.Revision <= admission.Revision || publication.Replaced != admission.Revision {
		t.Fatalf("commit publication=%+v admission=%+v", publication, admission)
	}
	if _, revision := ag.MessagesWithHostPublication(); revision != publication.Revision {
		t.Fatalf("current revision=%d, want the commit's %d", revision, publication.Revision)
	}
	observed := observePublications(t, ag, "go")
	if len(observed) == 0 {
		t.Fatal("no Frame-tagged envelope")
	}
	for _, o := range observed {
		if o.publication != publication.Revision {
			t.Fatalf("Frame %s reported %d, want the commit's %d", o.frame, o.publication, publication.Revision)
		}
	}
}
