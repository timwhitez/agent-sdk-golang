package agent

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/agent/compaction"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

// #180 item 1: the source history had no system message, so the in-flight
// summary carries the injected configured prompt. The host publishes its own
// system prompt before the result is applied; the rebase keeps the host's
// prompt and does not restore the stale injected one next to it.
func TestRebaseDoesNotRestoreInjectedBasePromptOverHostPrompt(t *testing.T) {
	const basePrompt = "configured base prompt"
	const hostPrompt = "host published prompt"
	var high *llm.Usage
	model := newGatedSummaryModel(
		func() (*llm.Completion, error) {
			return &llm.Completion{Content: llm.TextContent("first answer"), Usage: high}, nil
		},
	)
	writer := &countingCheckpointWriter{}
	ag, usage := interleaveAgent(t, model, writer, func(c *Config) {
		c.SystemPrompt = basePrompt
		c.InitialMessages = nonSystemMessages(longHistory())
	})
	high = usage
	if _, errs, finals := drainQuery(ag, context.Background(), "first request", nil); errs != 0 || finals != 1 {
		t.Fatalf("first turn errors=%d finals=%d", errs, finals)
	}
	<-model.summaryStarted
	if got := ag.Messages(); got[0].Role == llm.RoleSystem {
		t.Fatalf("fixture: source history already has a system message: %+v", got[0])
	}
	withHost := append([]llm.Message{llm.NewSystemMessage(hostPrompt)}, ag.Messages()...)
	if err := ag.ReplaceHistoryChecked(withHost); err != nil {
		t.Fatalf("host prompt rejected: %v", err)
	}
	close(model.gate)

	compactions, errs, finals := drainQuery(ag, context.Background(), "second request", nil)
	main, summaries := model.snapshot()
	if len(main) != 2 || summaries != 1 || compactions != 1 || errs != 0 || finals != 1 || writer.writes.Load() != 1 {
		t.Fatalf("main=%d summaries=%d compactions=%d errors=%d finals=%d writes=%d", len(main), summaries, compactions, errs, finals, writer.writes.Load())
	}
	next := main[1].Messages
	if countText(next, "prior work summarized") == 0 {
		t.Fatal("the rebased summary was not published")
	}
	if got := countText(next, basePrompt); got != 0 {
		t.Fatalf("stale injected base prompt appears %d times next to the host prompt", got)
	}
	if next[0].Role != llm.RoleSystem || next[0].Content.PlainText() != hostPrompt || countText(next, hostPrompt) != 1 {
		t.Fatalf("host prompt is not the single leading system message: %+v", next[0])
	}
	assertLegalToolPairs(t, next)
}

// #180 item 2: a usage-driven overflow summary that is computed but not
// published (here the checkpoint write fails, so it is requeued) leaves the
// history over the window. The turn must fail instead of sending the
// over-window request.
func TestUnpublishedUsageOverflowSummaryStopsTheTurn(t *testing.T) {
	var over *llm.Usage
	model := newGatedSummaryModel(
		func() (*llm.Completion, error) {
			return &llm.Completion{Usage: over, ToolCalls: []llm.ToolCall{{ID: "c1", Type: "function", Function: llm.FunctionCall{Name: "noop", Arguments: "{}"}}}}, nil
		},
		func() (*llm.Completion, error) { return &llm.Completion{Content: llm.TextContent("final")}, nil },
	)
	close(model.gate)
	noop := tools.Func[struct{}]("noop", "noop", func(context.Context, struct{}, *tools.Container) (any, error) {
		return "tool output", nil
	})
	writes := 0
	writer := compaction.CompactionCheckpointWriterFunc(func(context.Context, compaction.CompactionCheckpoint) error {
		writes++
		return errors.New("checkpoint store unavailable")
	})
	ag, _ := interleaveAgent(t, model, writer, func(c *Config) { c.Tools = []tools.Tool{noop} })
	limit := ag.compactor.OverflowLimit() + 1
	over = &llm.Usage{PromptTokens: limit, TotalTokens: limit}

	compactions, errs, finals := drainQuery(ag, context.Background(), "current request", nil)
	main, summaries := model.snapshot()
	if summaries != 1 || writes != 1 || compactions != 0 {
		t.Fatalf("fixture: summaries=%d writes=%d compactions=%d, want 1/1/0", summaries, writes, compactions)
	}
	if len(main) != 1 || errs != 1 || finals != 0 {
		t.Fatalf("over-window request was sent after an unpublished overflow summary: main=%d errors=%d finals=%d", len(main), errs, finals)
	}
	if countText(ag.Messages(), "prior work summarized") != 0 {
		t.Fatal("unpublished summary reached history")
	}
}

// #180 item 3: publication compares the live history with the clone it was
// rebased from by the JSON identity used for the source check. A clone that
// normalizes a field without changing that identity (an empty non-nil
// ThoughtSig becomes nil) must not defer publication.
func TestPublicationIgnoresCloneNormalization(t *testing.T) {
	writer := &countingCheckpointWriter{}
	ag, _ := interleaveAgent(t, &countingCompactionModel{}, writer, nil)
	call := llm.ToolCall{ID: "c1", Type: "function", Function: llm.FunctionCall{Name: "noop", Arguments: "{}"}, ThoughtSig: []byte{}}
	ag.mu.Lock()
	ag.messages = append(ag.messages, llm.NewAssistantMessage("", []llm.ToolCall{call}), llm.NewToolMessage("c1", "noop", llm.TextContent("tool output"), false))
	live := ag.messages
	ag.mu.Unlock()
	clone := llm.CloneMessages(live)
	if clone[len(clone)-2].ToolCalls[0].ThoughtSig != nil || !messageJSONEqual(live, clone) {
		t.Fatal("fixture: clone does not normalize the empty ThoughtSig under an equal JSON identity")
	}
	summary := []llm.Message{live[0], llm.NewUserMessage(validCompactionSummary("prior work summarized"))}
	ag.pendingCompactionMu.Lock()
	ag.pendingCompaction = &pendingCompaction{
		messages:    summary,
		snapshotLen: len(live),
		source:      clone,
		result:      compaction.Result{Compacted: true, Trigger: "auto", Watermark: "summarize", TiersApplied: []string{"summarize"}},
	}
	ag.pendingCompactionMu.Unlock()

	if !ag.applyPendingCompaction(nil) {
		t.Fatal("publication was deferred although the live history is unchanged")
	}
	if ag.hasPendingCompaction() || writer.writes.Load() != 1 {
		t.Fatalf("pending=%v writes=%d, want published once", ag.hasPendingCompaction(), writer.writes.Load())
	}
	if got := ag.Messages(); !messageJSONEqual(got, summary) {
		t.Fatalf("published history = %d messages, want the summary", len(got))
	}
}

// #180 review N1/N2: rebasing a candidate that injected the configured base
// prompt (its source had no system message). The injected copy is dropped
// only when the host now has its own base prompt (an unnamed system message
// or the same prompt in the prefix) or the kept tail already carries the same
// prompt; a named host context message alone keeps it, as the unchanged path
// does, and the prompt is never duplicated.
func TestRebaseInjectedBasePromptCases(t *testing.T) {
	const basePrompt = "configured base prompt"
	named := func(name, text string) llm.Message {
		m := llm.NewSystemMessage(text)
		m.Name = name
		return m
	}
	user, answer := llm.NewUserMessage("question"), llm.NewAssistantMessage("answer", nil)
	summary := llm.NewUserMessage("prior work summarized")
	base := llm.NewSystemMessage(basePrompt)
	mem := named("memory_context", "memory v1")
	cases := []struct {
		name string
		live []llm.Message
		want []llm.Message
	}{
		{"unchanged", []llm.Message{user, answer}, []llm.Message{base, summary}},
		{"host base prompt in prefix", []llm.Message{llm.NewSystemMessage("host prompt"), user, answer}, []llm.Message{llm.NewSystemMessage("host prompt"), summary}},
		{"same prompt in prefix", []llm.Message{base, user, answer}, []llm.Message{base, summary}},
		{"C: named context only in prefix", []llm.Message{mem, user, answer}, []llm.Message{mem, base, summary}},
		{"D: same prompt appended after the source", []llm.Message{user, answer, base}, []llm.Message{summary, base}},
		{"D: same prompt appended after a changed prefix", []llm.Message{mem, user, answer, base}, []llm.Message{mem, summary, base}},
		{"named prompt text appended after the source", []llm.Message{user, answer, named("other", basePrompt)}, []llm.Message{base, summary, named("other", basePrompt)}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			pending := &pendingCompaction{
				messages:    []llm.Message{base, summary},
				snapshotLen: 2,
				source:      []llm.Message{user, answer},
			}
			merged, _, ok := rebasePendingCompaction(pending, tc.live, basePrompt)
			if !ok {
				t.Fatal("rebase rejected a system-only change")
			}
			if !messageJSONEqual(merged, tc.want) {
				t.Fatalf("merged:\n%s\nwant:\n%s", describeMessages(merged), describeMessages(tc.want))
			}
		})
	}
}

func describeMessages(messages []llm.Message) string {
	var b strings.Builder
	for _, m := range messages {
		fmt.Fprintf(&b, "  %s[%s] %q\n", m.Role, m.Name, m.Content.PlainText())
	}
	return b.String()
}
