package agent

import (
	"context"
	"errors"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/agent/compaction"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

// gatedSummaryModel answers main requests from a script and blocks every
// summary request until gate is closed; summaryStarted receives one value
// per summary request once it has entered the provider.
type gatedSummaryModel struct {
	mu          sync.Mutex
	script      []func() (*llm.Completion, error)
	main        []llm.InvokeRequest
	summaries   int
	summaryText []string
	// keepUnknown makes summaries preserve an UNKNOWN host checkpoint state,
	// as a faithful summarizer must.
	keepUnknown    bool
	gate           chan struct{}
	summaryStarted chan struct{}
}

func newGatedSummaryModel(script ...func() (*llm.Completion, error)) *gatedSummaryModel {
	return &gatedSummaryModel{script: script, gate: make(chan struct{}), summaryStarted: make(chan struct{}, 8)}
}

func (m *gatedSummaryModel) Provider() string { return "fixture" }
func (m *gatedSummaryModel) Model() string    { return "gated-summary" }
func (m *gatedSummaryModel) Invoke(ctx context.Context, req llm.InvokeRequest) (*llm.Completion, error) {
	owned, err := llm.CloneInvokeRequest(req)
	if err != nil {
		return nil, err
	}
	for _, message := range owned.Messages {
		if message.Role == llm.RoleSystem && strings.Contains(message.Content.PlainText(), "operational checkpoint") {
			var text strings.Builder
			for _, part := range owned.Messages {
				text.WriteString(part.Content.PlainText())
			}
			m.mu.Lock()
			m.summaries++
			m.summaryText = append(m.summaryText, text.String())
			m.mu.Unlock()
			m.summaryStarted <- struct{}{}
			select {
			case <-m.gate:
			case <-ctx.Done():
				return nil, ctx.Err()
			}
			body := "prior work summarized"
			if m.keepUnknown && strings.Contains(text.String(), "Status: UNKNOWN") {
				body += "; host checkpoint state UNKNOWN"
			}
			return &llm.Completion{Content: llm.TextContent(validCompactionSummary(body))}, nil
		}
	}
	m.mu.Lock()
	m.main = append(m.main, owned)
	n := len(m.main)
	step := func() (*llm.Completion, error) { return &llm.Completion{Content: llm.TextContent("ok")}, nil }
	if n <= len(m.script) {
		step = m.script[n-1]
	}
	m.mu.Unlock()
	return step()
}

func (m *gatedSummaryModel) snapshot() ([]llm.InvokeRequest, int) {
	m.mu.Lock()
	defer m.mu.Unlock()
	return append([]llm.InvokeRequest(nil), m.main...), m.summaries
}

type countingCheckpointWriter struct{ writes atomic.Int32 }

func (w *countingCheckpointWriter) SaveCompactionCheckpoint(context.Context, compaction.CompactionCheckpoint) error {
	w.writes.Add(1)
	return nil
}

// interleaveAgent sizes compaction so that a completion reporting usage
// between the summary threshold and the overflow limit launches one
// asynchronous automatic summary. It returns that usage.
func interleaveAgent(t *testing.T, model llm.ChatModel, writer compaction.CompactionCheckpointWriter, mutate func(*Config)) (*Agent, *llm.Usage) {
	t.Helper()
	cfg := Config{
		LLM:                    model,
		InitialMessages:        longHistory(),
		InvokeRetryMaxAttempts: 1,
		Compaction:             &compaction.Config{Enabled: true, ContextWindow: 100000, ThresholdRatio: 0.85, CheckpointWriter: writer},
		Warningf:               func(string, ...any) {},
	}
	if mutate != nil {
		mutate(&cfg)
	}
	ag, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	high := (ag.compactor.ThresholdTokens() + ag.compactor.OverflowLimit()) / 2
	return ag, &llm.Usage{PromptTokens: high, TotalTokens: high}
}

func countText(messages []llm.Message, needle string) int {
	n := 0
	for _, m := range messages {
		n += strings.Count(m.Content.PlainText(), needle)
	}
	return n
}

func assertLegalToolPairs(t *testing.T, messages []llm.Message) {
	t.Helper()
	if _, changed := repairToolCallPairs(messages); changed {
		t.Fatalf("request has unpaired tool calls/results: %+v", messages)
	}
	if start := llm.OpenToolCallBlockStart(messages); start >= 0 {
		t.Fatalf("request ends inside an open tool-call block at %d", start)
	}
}

func drainQuery(ag *Agent, ctx context.Context, input string, steering <-chan SteeringMsg) (compactions, errs, finals int) {
	for ev := range ag.QueryStreamWithSteering(ctx, llm.TextContent(input), steering) {
		switch ev.(type) {
		case CompactionEvent:
			compactions++
		case ErrorEvent:
			errs++
		case FinalResponseEvent:
			finals++
		}
	}
	return
}

// G2: a real steering message that arrives while an automatic summary is in
// flight enters history after the summary is published, exactly once, and
// the next request keeps legal tool pairs. The in-flight summary is not an
// interruptible stage, so a steering interrupt does not cancel it.
func TestSteeringDuringInFlightAutomaticSummaryIsKeptOnce(t *testing.T) {
	var high *llm.Usage
	model := newGatedSummaryModel(
		func() (*llm.Completion, error) {
			return &llm.Completion{Usage: high, ToolCalls: []llm.ToolCall{{ID: "c1", Type: "function", Function: llm.FunctionCall{Name: "noop", Arguments: "{}"}}}}, nil
		},
		func() (*llm.Completion, error) { return &llm.Completion{Content: llm.TextContent("final")}, nil },
	)
	noop := tools.Func[struct{}]("noop", "noop", func(context.Context, struct{}, *tools.Container) (any, error) {
		return "tool output", nil
	})
	writer := &countingCheckpointWriter{}
	ag, usage := interleaveAgent(t, model, writer, func(c *Config) { c.Tools = []tools.Tool{noop} })
	high = usage
	steering := make(chan SteeringMsg, 1)
	go func() {
		<-model.summaryStarted
		steering <- SteeringMsg{Content: "steer during summary"}
		if ag.InterruptActiveStageForSteering() {
			t.Error("an in-flight summary was reported as an interruptible stage")
		}
		close(model.gate)
	}()
	compactions, errs, finals := drainQuery(ag, context.Background(), "current request", steering)
	main, summaries := model.snapshot()
	if len(main) != 2 || summaries != 1 || compactions != 1 || errs != 0 || finals != 1 || writer.writes.Load() != 1 {
		t.Fatalf("main=%d summaries=%d compactions=%d errors=%d finals=%d writes=%d", len(main), summaries, compactions, errs, finals, writer.writes.Load())
	}
	next := main[1].Messages
	if countText(next, "prior work summarized") == 0 {
		t.Fatal("next request does not carry the published summary")
	}
	if got := countText(next, "steer during summary"); got != 1 {
		t.Fatalf("steering appears %d times in the next request, want 1", got)
	}
	last := next[len(next)-1]
	if last.Role != llm.RoleUser || !strings.Contains(last.Content.PlainText(), "steer during summary") {
		t.Fatalf("steering is not the newest input: %+v", last)
	}
	assertLegalToolPairs(t, next)
	if got := countText(ag.Messages(), "steer during summary"); got != 1 {
		t.Fatalf("steering appears %d times in history, want 1", got)
	}
}

// G2: steering that arrives while a typed-overflow recovery summary is in
// flight reaches the recovered request exactly once, next to the original
// input, and the recovery stays a single compaction.
func TestSteeringDuringOverflowRecoverySummaryIsKeptOnce(t *testing.T) {
	model := newGatedSummaryModel(
		func() (*llm.Completion, error) { return nil, typedOverflow() },
	)
	writer := &countingCheckpointWriter{}
	ag, _ := interleaveAgent(t, model, writer, nil)
	steering := make(chan SteeringMsg, 1)
	go func() {
		<-model.summaryStarted
		steering <- SteeringMsg{Content: "steer during recovery"}
		close(model.gate)
	}()
	compactions, errs, finals := drainQuery(ag, context.Background(), "current request", steering)
	main, summaries := model.snapshot()
	if len(main) != 2 || summaries != 1 || compactions != 1 || errs != 0 || finals != 1 || writer.writes.Load() != 1 {
		t.Fatalf("main=%d summaries=%d compactions=%d errors=%d finals=%d writes=%d", len(main), summaries, compactions, errs, finals, writer.writes.Load())
	}
	next := main[1].Messages
	if countText(next, "prior work summarized") == 0 {
		t.Fatal("recovered request does not carry the summary")
	}
	if got := countText(next, "steer during recovery"); got != 1 {
		t.Fatalf("steering appears %d times in the recovered request, want 1", got)
	}
	if got := countText(next, "current request"); got < 1 {
		t.Fatal("the original input was lost by recovery")
	}
	assertLegalToolPairs(t, next)
}

// G2: an automatic summary launched at the end of a turn is still in flight
// when the host replaces the idle history with different conversation
// content (branch switch, an edited earlier turn). Its result was computed
// from the superseded history, so it must not be published over the
// replacement: no checkpoint is written for it and the next request carries
// the replacement verbatim. (A change of system messages only is rebased;
// see TestInFlightAutomaticSummaryRebasesOntoHostMemoryRefresh.)
func TestInFlightAutomaticSummaryDoesNotOverwriteReplacedHistory(t *testing.T) {
	branch := func() []llm.Message {
		msgs := []llm.Message{llm.NewSystemMessage("system")}
		for i := 0; i < 10; i++ {
			msgs = append(msgs, llm.NewUserMessage("branch question "+string(rune('a'+i))), llm.NewAssistantMessage("branch answer "+string(rune('a'+i)), nil))
		}
		return msgs
	}
	editedAnswer := func(current []llm.Message) []llm.Message {
		out := llm.CloneMessages(current)
		out[len(out)-1] = llm.NewAssistantMessage("rewritten first answer", nil)
		return out
	}
	for name, replace := range map[string]func([]llm.Message) []llm.Message{
		"longer branch":         func([]llm.Message) []llm.Message { return branch() },
		"edited earlier answer": editedAnswer,
	} {
		t.Run(name, func(t *testing.T) {
			var high *llm.Usage
			model := newGatedSummaryModel(
				func() (*llm.Completion, error) {
					return &llm.Completion{Content: llm.TextContent("first answer"), Usage: high}, nil
				},
			)
			writer := &countingCheckpointWriter{}
			ag, usage := interleaveAgent(t, model, writer, nil)
			high = usage
			if _, errs, finals := drainQuery(ag, context.Background(), "first request", nil); errs != 0 || finals != 1 {
				t.Fatalf("first turn errors=%d finals=%d", errs, finals)
			}
			<-model.summaryStarted
			replacement := replace(ag.Messages())
			if err := ag.ReplaceHistoryChecked(replacement); err != nil {
				t.Fatalf("idle replacement rejected: %v", err)
			}
			close(model.gate)

			compactions, errs, finals := drainQuery(ag, context.Background(), "second request", nil)
			main, summaries := model.snapshot()
			if len(main) != 2 || summaries != 1 || errs != 0 || finals != 1 {
				t.Fatalf("main=%d summaries=%d errors=%d finals=%d", len(main), summaries, errs, finals)
			}
			if compactions != 0 || writer.writes.Load() != 0 {
				t.Fatalf("stale summary was published: compactions=%d checkpoint writes=%d", compactions, writer.writes.Load())
			}
			next := main[1].Messages
			if countText(next, "prior work summarized") != 0 {
				t.Fatal("next request carries a summary of the superseded history")
			}
			want := append(llm.CloneMessages(replacement), llm.NewUserMessage("second request"))
			if !messageJSONEqual(next, want) {
				t.Fatalf("next request is not the replacement plus the new input:\n got %d messages\nwant %d messages", len(next), len(want))
			}
		})
	}
}

// G2: a candidate requeued because the history shrank below its source (a
// shorter host replacement) stays stale when the history later grows past
// that length again; it is dropped rather than spliced onto the new history.
func TestRequeuedStaleSummaryIsNotPublishedAfterHistoryRegrows(t *testing.T) {
	var high *llm.Usage
	model := newGatedSummaryModel(
		func() (*llm.Completion, error) {
			return &llm.Completion{Content: llm.TextContent("first answer"), Usage: high}, nil
		},
	)
	writer := &countingCheckpointWriter{}
	ag, usage := interleaveAgent(t, model, writer, nil)
	high = usage
	if _, errs, finals := drainQuery(ag, context.Background(), "first request", nil); errs != 0 || finals != 1 {
		t.Fatalf("first turn errors=%d finals=%d", errs, finals)
	}
	<-model.summaryStarted
	if err := ag.ReplaceHistoryChecked([]llm.Message{llm.NewSystemMessage("system"), llm.NewUserMessage("short branch")}); err != nil {
		t.Fatal(err)
	}
	close(model.gate)
	if err := ag.waitForCompactionIdle(context.Background(), nil); err != nil {
		t.Fatal(err)
	}
	ag.applyPendingCompaction(nil) // history shrank: requeued, not published
	regrown := []llm.Message{llm.NewSystemMessage("system")}
	for i := 0; i < 12; i++ {
		regrown = append(regrown, llm.NewUserMessage("regrown question"), llm.NewAssistantMessage("regrown answer", nil))
	}
	if err := ag.ReplaceHistoryChecked(regrown); err != nil {
		t.Fatal(err)
	}
	compactions, errs, finals := drainQuery(ag, context.Background(), "next request", nil)
	main, _ := model.snapshot()
	if compactions != 0 || writer.writes.Load() != 0 || errs != 0 || finals != 1 || len(main) != 2 {
		t.Fatalf("compactions=%d writes=%d errors=%d finals=%d main=%d", compactions, writer.writes.Load(), errs, finals, len(main))
	}
	want := append(llm.CloneMessages(regrown), llm.NewUserMessage("next request"))
	if !messageJSONEqual(main[1].Messages, want) {
		t.Fatalf("next request is not the regrown history plus the new input (%d messages)", len(main[1].Messages))
	}
}

// Live publication failure after an acknowledged checkpoint: a host system
// update lands while the checkpoint writer runs, so the acknowledged result no
// longer matches the live history. As before, the Agent keeps the live
// history, rolls the ledger back and ends the recovery; the acknowledged
// checkpoint is left unreferenced. At the next boundary the requeued result is
// rebased onto the host's system update and written once more (two writes in
// total), so the published history carries the update and never the stale
// system message.
func TestAcknowledgedCheckpointWithConcurrentHostUpdateIsRebasedOnce(t *testing.T) {
	model := newGatedSummaryModel(
		func() (*llm.Completion, error) { return nil, typedOverflow() },
	)
	close(model.gate)
	var ag *Agent
	var writes atomic.Int32
	updateErr := make(chan error, 1)
	writer := compaction.CompactionCheckpointWriterFunc(func(context.Context, compaction.CompactionCheckpoint) error {
		if writes.Add(1) == 1 {
			current := ag.Messages()
			current[0] = llm.NewSystemMessage("system with refreshed host context")
			updateErr <- ag.ReplaceHistoryChecked(current)
		}
		return nil
	})
	ag, _ = interleaveAgent(t, model, writer, nil)
	compactions, errs, finals := drainQuery(ag, context.Background(), "current request", nil)
	if err := <-updateErr; err != nil {
		t.Fatalf("host system update during the turn was rejected: %v", err)
	}
	main, summaries := model.snapshot()
	if len(main) != 1 || summaries != 1 || compactions != 0 || errs != 1 || finals != 0 || writes.Load() != 1 {
		t.Fatalf("main=%d summaries=%d compactions=%d errors=%d finals=%d writes=%d", len(main), summaries, compactions, errs, finals, writes.Load())
	}
	updated := ag.Messages()
	if countText(updated, "prior work summarized") != 0 || updated[0].Content.PlainText() != "system with refreshed host context" {
		t.Fatal("the acknowledged result was published over the host update")
	}

	compactions, errs, finals = drainQuery(ag, context.Background(), "next request", nil)
	main, summaries = model.snapshot()
	if compactions != 1 || errs != 0 || finals != 1 || writes.Load() != 2 || summaries != 1 || len(main) != 2 {
		t.Fatalf("next turn: compactions=%d errors=%d finals=%d writes=%d summaries=%d main=%d", compactions, errs, finals, writes.Load(), summaries, len(main))
	}
	next := main[1].Messages
	if countText(next, "prior work summarized") == 0 || countText(next, "system with refreshed host context") != 1 {
		t.Fatal("next request does not carry the rebased summary with the host update")
	}
	for _, m := range next {
		if m.Role == llm.RoleSystem && m.Content.PlainText() == "system" {
			t.Fatal("next request carries the superseded system message")
		}
	}
	if got := countText(next, "next request"); got != 1 {
		t.Fatalf("new input appears %d times, want 1", got)
	}
	assertLegalToolPairs(t, next)
}

// G2 (Goode memory refresh): while an end-of-turn summary is in flight the
// host refreshes its memory context the way Goode does before each prompt:
// the old memory system message is removed from the middle of the history
// and the new one appended at the end. Only system messages changed, so the
// summary is rebased and published once (one checkpoint write) instead of
// being paid for and discarded; the next request carries the new memory, not
// the old, and legal tool pairs.
func TestInFlightAutomaticSummaryRebasesOntoHostMemoryRefresh(t *testing.T) {
	memory := func(text string) llm.Message {
		m := llm.NewSystemMessage(text)
		m.Name = "memory_context"
		return m
	}
	refresh := func(history []llm.Message, text string) []llm.Message {
		out := make([]llm.Message, 0, len(history)+1)
		for _, m := range history {
			if m.Role == llm.RoleSystem && m.Name == "memory_context" {
				continue
			}
			out = append(out, m)
		}
		return append(out, memory(text))
	}
	var high *llm.Usage
	model := newGatedSummaryModel(
		func() (*llm.Completion, error) {
			return &llm.Completion{ToolCalls: []llm.ToolCall{{ID: "c1", Type: "function", Function: llm.FunctionCall{Name: "noop", Arguments: "{}"}}}}, nil
		},
		func() (*llm.Completion, error) {
			return &llm.Completion{Content: llm.TextContent("first answer"), Usage: high}, nil
		},
	)
	noop := tools.Func[struct{}]("noop", "noop", func(context.Context, struct{}, *tools.Container) (any, error) {
		return "tool output", nil
	})
	writer := &countingCheckpointWriter{}
	initial := longHistory()
	initial = append(initial[:9], append([]llm.Message{memory("memory v1")}, initial[9:]...)...)
	ag, usage := interleaveAgent(t, model, writer, func(c *Config) {
		c.Tools = []tools.Tool{noop}
		c.InitialMessages = initial
	})
	high = usage
	if _, errs, finals := drainQuery(ag, context.Background(), "first request", nil); errs != 0 || finals != 1 {
		t.Fatalf("first turn errors=%d finals=%d", errs, finals)
	}
	<-model.summaryStarted
	if err := ag.ReplaceHistoryChecked(refresh(ag.Messages(), "memory v2")); err != nil {
		t.Fatalf("memory refresh rejected: %v", err)
	}
	close(model.gate)

	compactions, errs, finals := drainQuery(ag, context.Background(), "second request", nil)
	main, summaries := model.snapshot()
	if len(main) != 3 || summaries != 1 || compactions != 1 || errs != 0 || finals != 1 || writer.writes.Load() != 1 {
		t.Fatalf("main=%d summaries=%d compactions=%d errors=%d finals=%d writes=%d", len(main), summaries, compactions, errs, finals, writer.writes.Load())
	}
	next := main[2].Messages
	if countText(next, "prior work summarized") == 0 {
		t.Fatal("the rebased summary was not published")
	}
	if countText(next, "memory v2") != 1 || countText(next, "memory v1") != 0 {
		t.Fatalf("next request memory: v2=%d v1=%d, want 1 and 0", countText(next, "memory v2"), countText(next, "memory v1"))
	}
	if got := countText(next, "second request"); got != 1 {
		t.Fatalf("new input appears %d times, want 1", got)
	}
	if next[0].Role != llm.RoleSystem || next[0].Content.PlainText() != "system" {
		t.Fatalf("base system prompt not first: %+v", next[0])
	}
	assertLegalToolPairs(t, next)
	if got := ag.Messages(); countText(got, "memory v1") != 0 || countText(got, "memory v2") != 1 {
		t.Fatal("history after publication lost the refreshed memory")
	}
}

// N1: the emergency trim at the overflow boundary publishes the trimmed
// history through the same checked publication (one checkpoint write, one
// CompactionEvent), and reports failure to the overflow caller when that
// publication does not happen, instead of claiming the overflow was handled.
func TestEmergencyTrimIsPublishedOrReportedAsFailure(t *testing.T) {
	history := []llm.Message{
		llm.NewSystemMessage("sys"),
		llm.NewUserMessage("real request"),
		llm.NewAssistantMessage(strings.Repeat("a ", 800), nil),
		llm.NewAssistantMessage(strings.Repeat("b ", 800), nil),
		llm.NewAssistantMessage(strings.Repeat("c ", 800), nil),
	}
	for _, failWrite := range []bool{false, true} {
		t.Run(map[bool]string{false: "published", true: "checkpoint failure"}[failWrite], func(t *testing.T) {
			writes := 0
			ag, err := New(Config{
				LLM:             &countingCompactionModel{},
				InitialMessages: history,
				Warningf:        func(string, ...any) {},
				Compaction: &compaction.Config{
					Enabled: true, ContextWindow: 1000, ThresholdRatio: 1.0,
					CheckpointWriter: compaction.CompactionCheckpointWriterFunc(func(context.Context, compaction.CompactionCheckpoint) error {
						writes++
						if failWrite {
							return errors.New("checkpoint store unavailable")
						}
						return nil
					}),
				},
			})
			if err != nil {
				t.Fatal(err)
			}
			// Summary tier suppressed, local tiers cannot reduce assistant text:
			// the overflow path must fall back to the emergency trim.
			ag.compactionFailureStreak.Store(compactionSummaryDisableStreak)
			events := make(chan Event, 64)
			completion := &llm.Completion{Usage: &llm.Usage{TotalTokens: 4000, PromptTokens: 4000}}
			compactErr := ag.compactSyncOverflow(context.Background(), "", completion, completion.Usage, wrapLegacyEventOutput(events))
			close(events)
			trims := 0
			for ev := range events {
				if c, ok := ev.(CompactionEvent); ok && slices.Contains(c.Result.TiersApplied, "emergency_trim") {
					trims++
				}
			}
			got := ag.Messages()
			if writes != 1 {
				t.Fatalf("checkpoint writes = %d, want 1", writes)
			}
			if failWrite {
				if compactErr == nil || trims != 0 || !messageJSONEqual(got, history) {
					t.Fatalf("unpublished trim reported: err=%v trims=%d history changed=%v", compactErr, trims, !messageJSONEqual(got, history))
				}
				return
			}
			if compactErr != nil || trims != 1 || len(got) >= len(history) {
				t.Fatalf("trim not published: err=%v trims=%d messages %d -> %d", compactErr, trims, len(history), len(got))
			}
			if estimate := ag.compactor.EstimateMessages(got); estimate > ag.compactor.ThresholdTokens() {
				t.Fatalf("published trim is over budget: %d > %d", estimate, ag.compactor.ThresholdTokens())
			}
		})
	}
}
