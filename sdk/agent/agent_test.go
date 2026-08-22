package agent

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
	"unicode/utf8"

	"github.com/timwhitez/agent-sdk-golang/sdk/agent/compaction"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

func TestToolCallAccumulatorPreservesWhitespace(t *testing.T) {
	acc := &toolCallAccumulator{}
	acc.apply(llm.StreamToolCallDeltaEvent{Index: 0, NameDelta: "write", ArgumentsDelta: `{"content":"hello`})
	acc.apply(llm.StreamToolCallDeltaEvent{Index: 0, ArgumentsDelta: " "})
	acc.apply(llm.StreamToolCallDeltaEvent{Index: 0, ArgumentsDelta: `world"}`})

	calls := acc.finalize()
	if len(calls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(calls))
	}
	if calls[0].Function.Arguments != `{"content":"hello world"}` {
		t.Fatalf("expected whitespace preserved, got %q", calls[0].Function.Arguments)
	}
}

func TestWithCompactionTelemetryPreservesComparableEstimateCounts(t *testing.T) {
	ag := &Agent{compactor: compaction.NewService(&compaction.Config{Enabled: true})}
	usage := &llm.Usage{
		PromptTokens:       90,
		CompletionTokens:   10,
		TotalTokens:        100,
		PromptTokensValid:  true,
		PromptTokensSource: llm.PromptTokensSourceProvider,
	}
	res := compaction.Result{
		Compacted:        true,
		OriginalTokens:   180,
		NewTokens:        120,
		TokenCountSource: compaction.TokenCountSourceEstimate,
		TiersApplied:     []string{"snip"},
	}

	got := ag.withCompactionTelemetry(res, "usage", "snip", usage)
	if got.OriginalTokens != 180 || got.NewTokens != 120 {
		t.Fatalf("withCompactionTelemetry rewrote comparable estimates: %d -> %d", got.OriginalTokens, got.NewTokens)
	}
	if got.TokenCountSource != compaction.TokenCountSourceEstimate {
		t.Fatalf("token_count_source = %q, want %q", got.TokenCountSource, compaction.TokenCountSourceEstimate)
	}
	if got.Usage == nil || got.Usage.TotalTokens != 100 || got.Usage.PromptTokensSource != llm.PromptTokensSourceProvider {
		t.Fatalf("provider trigger usage was not preserved: %#v", got.Usage)
	}
}

func TestToolCallAccumulatorPreservesMissingArgumentsAndUsesPlaceholderPrefix(t *testing.T) {
	acc := &toolCallAccumulator{}
	acc.apply(llm.StreamToolCallDeltaEvent{Index: 0, NameDelta: "ping"})

	calls := acc.finalize()
	if len(calls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(calls))
	}
	if calls[0].Function.Arguments != "" {
		t.Fatalf("expected missing arguments to stay empty, got %q", calls[0].Function.Arguments)
	}
	if !strings.HasPrefix(calls[0].ID, syntheticToolCallIDPrefix) {
		t.Fatalf("expected placeholder id with prefix %q, got %q", syntheticToolCallIDPrefix, calls[0].ID)
	}
}

type toolContinuationModel struct {
	calls int
}

func (m *toolContinuationModel) Provider() string { return "stub" }
func (m *toolContinuationModel) Model() string    { return "stub" }

func (m *toolContinuationModel) Invoke(_ context.Context, _ llm.InvokeRequest) (*llm.Completion, error) {
	m.calls++
	switch m.calls {
	case 1:
		return &llm.Completion{
			Content: llm.TextContent(""),
			ToolCalls: []llm.ToolCall{
				{
					ID:   "call_1",
					Type: "function",
					Function: llm.FunctionCall{
						Name:      "echo",
						Arguments: `{"text":"hello`,
					},
				},
			},
			StopReason: "max_tokens",
			ResponseID: "resp_tool_cont_1",
		}, nil
	case 2:
		return &llm.Completion{
			Content: llm.TextContent(""),
			ToolCalls: []llm.ToolCall{
				{
					ID:   "call_1",
					Type: "function",
					Function: llm.FunctionCall{
						Name:      "echo",
						Arguments: ` world"}`,
					},
				},
			},
			StopReason: "stop",
			ResponseID: "resp_tool_cont_2",
		}, nil
	default:
		return &llm.Completion{Content: llm.TextContent("done"), ResponseID: "resp_tool_cont_3"}, nil
	}
}

type invalidMergedToolContinuationModel struct {
	calls int
}

func (m *invalidMergedToolContinuationModel) Provider() string { return "stub" }
func (m *invalidMergedToolContinuationModel) Model() string    { return "stub" }

func (m *invalidMergedToolContinuationModel) Invoke(_ context.Context, _ llm.InvokeRequest) (*llm.Completion, error) {
	m.calls++
	switch m.calls {
	case 1:
		return &llm.Completion{
			Content: llm.TextContent(""),
			ToolCalls: []llm.ToolCall{{
				ID:   "call_1",
				Type: "function",
				Function: llm.FunctionCall{
					Name:      "echo",
					Arguments: `{"text":"hello`,
				},
			}},
			StopReason: "max_tokens",
			ResponseID: "resp_invalid_merge_1",
		}, nil
	case 2:
		return &llm.Completion{
			Content: llm.TextContent(""),
			ToolCalls: []llm.ToolCall{{
				ID:   "call_1",
				Type: "function",
				Function: llm.FunctionCall{
					Name:      "echo",
					Arguments: ` world"`,
				},
			}},
			StopReason: "stop",
			ResponseID: "resp_invalid_merge_2",
		}, nil
	case 3:
		return &llm.Completion{
			Content: llm.TextContent(""),
			ToolCalls: []llm.ToolCall{{
				ID:   "call_1",
				Type: "function",
				Function: llm.FunctionCall{
					Name:      "echo",
					Arguments: `}`,
				},
			}},
			StopReason: "stop",
			ResponseID: "resp_invalid_merge_3",
		}, nil
	default:
		return &llm.Completion{Content: llm.TextContent("done"), ResponseID: "resp_invalid_merge_4"}, nil
	}
}

type continuationLimitModel struct {
	calls int
}

func (m *continuationLimitModel) Provider() string { return "stub" }
func (m *continuationLimitModel) Model() string    { return "stub" }

func (m *continuationLimitModel) Invoke(_ context.Context, _ llm.InvokeRequest) (*llm.Completion, error) {
	m.calls++
	switch m.calls {
	case 1:
		return &llm.Completion{
			Content: llm.TextContent(""),
			ToolCalls: []llm.ToolCall{{
				ID:   "call_1",
				Type: "function",
				Function: llm.FunctionCall{
					Name:      "echo",
					Arguments: `{"text":"hello`,
				},
			}},
			StopReason: "max_tokens",
			ResponseID: "resp_limit_1",
		}, nil
	case 2:
		return &llm.Completion{
			Content: llm.TextContent(""),
			ToolCalls: []llm.ToolCall{{
				ID:   "call_1",
				Type: "function",
				Function: llm.FunctionCall{
					Name:      "echo",
					Arguments: ` world`,
				},
			}},
			StopReason: "stop",
			ResponseID: "resp_limit_2",
		}, nil
	case 3:
		return &llm.Completion{
			Content: llm.TextContent(""),
			ToolCalls: []llm.ToolCall{{
				ID:   "call_1",
				Type: "function",
				Function: llm.FunctionCall{
					Name:      "echo",
					Arguments: ` still`,
				},
			}},
			StopReason: "stop",
			ResponseID: "resp_limit_3",
		}, nil
	case 4:
		return &llm.Completion{
			Content: llm.TextContent(""),
			ToolCalls: []llm.ToolCall{{
				ID:   "call_1",
				Type: "function",
				Function: llm.FunctionCall{
					Name:      "echo",
					Arguments: ` invalid`,
				},
			}},
			StopReason: "stop",
			ResponseID: "resp_limit_4",
		}, nil
	default:
		return &llm.Completion{
			ToolCalls: []llm.ToolCall{{
				ID:   "done_1",
				Type: "function",
				Function: llm.FunctionCall{
					Name:      "done",
					Arguments: `{"message":"finished via continuation limit"}`,
				},
			}},
			StopReason: "tool_calls",
			ResponseID: "resp_limit_done",
		}, nil
	}
}

type textAutoContinueCompactionModel struct {
	mu              sync.Mutex
	summaryPrompt   string
	regularCalls    int
	compactionCalls int
	compactionDelay time.Duration
	compactionDone  chan struct{}
}

func (m *textAutoContinueCompactionModel) Provider() string { return "stub" }
func (m *textAutoContinueCompactionModel) Model() string    { return "stub" }

func (m *textAutoContinueCompactionModel) Invoke(_ context.Context, req llm.InvokeRequest) (*llm.Completion, error) {
	m.mu.Lock()
	summaryPrompt := m.summaryPrompt
	delay := m.compactionDelay
	doneCh := m.compactionDone
	m.mu.Unlock()

	if len(req.Messages) > 1 {
		first := req.Messages[0]
		last := req.Messages[len(req.Messages)-1]
		firstText := first.Content.PlainText()
		lastText := last.Content.PlainText()
		if first.Role == llm.RoleSystem && last.Role == llm.RoleUser && strings.Contains(firstText, summaryPrompt) && strings.Contains(firstText, "internal context compaction pipeline") && strings.Contains(lastText, "BEGIN_UNTRUSTED_MATERIAL") {
			m.mu.Lock()
			m.compactionCalls++
			if doneCh != nil {
				m.compactionDone = nil
			}
			m.mu.Unlock()
			if delay > 0 {
				time.Sleep(delay)
			}
			if doneCh != nil {
				close(doneCh)
			}
			return &llm.Completion{Content: llm.TextContent(validCompactionSummary("compressed")), StopReason: "stop", ResponseID: "resp_compaction"}, nil
		}
	}

	m.mu.Lock()
	m.regularCalls++
	regularCalls := m.regularCalls
	m.mu.Unlock()
	if regularCalls == 1 {
		return &llm.Completion{
			Content:    llm.TextContent("part 1"),
			StopReason: "max_tokens",
			ResponseID: "resp_regular_1",
			Usage: &llm.Usage{
				PromptTokens:     90,
				CompletionTokens: 10,
				TotalTokens:      100,
			},
		}, nil
	}

	if regularCalls == 2 {
		return &llm.Completion{
			Content:    llm.TextContent("part 2"),
			StopReason: "stop",
			ResponseID: "resp_regular_2",
			Usage:      llm.NewProviderUsage(10, 2, 12),
		}, nil
	}
	return &llm.Completion{
		Content:    llm.TextContent("follow up"),
		StopReason: "stop",
		ResponseID: "resp_regular_3",
		Usage:      llm.NewProviderUsage(12, 2, 14),
	}, nil
}

func (m *textAutoContinueCompactionModel) Counts() (regular, compaction int) {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.regularCalls, m.compactionCalls
}

func TestAgentAutoContinuesToolCallsOnMaxTokens(t *testing.T) {
	type echoArgs struct {
		Text string `json:"text"`
	}
	var called int
	got := ""
	echoTool := tools.Func[echoArgs]("echo", "echo", func(_ context.Context, args echoArgs, _ *tools.Container) (any, error) {
		called++
		got = args.Text
		return "ok", nil
	})
	model := &toolContinuationModel{}
	ag, err := New(Config{LLM: model, Tools: []tools.Tool{echoTool}})
	if err != nil {
		t.Fatalf("new agent: %v", err)
	}
	events := collectEvents(ag.QueryStream(context.Background(), llm.TextContent("hi")))
	if model.calls != 3 {
		t.Fatalf("expected 3 model calls, got %d", model.calls)
	}
	if called != 1 {
		t.Fatalf("expected tool to be called once, got %d", called)
	}
	if got != "hello world" {
		t.Fatalf("expected merged tool args, got %q", got)
	}
	if tr, ok := findToolResult(events, "echo"); !ok || tr.IsError {
		t.Fatalf("expected successful tool result for echo")
	}
	assertHistoryContainsNamedUserMessage(t, ag.Messages(), "Your response was truncated. Please continue exactly where you left off.", "sdk_internal_tool_call_continuation")
}

func TestAgentValidatesMergedToolCallsBeforeExecution(t *testing.T) {
	type echoArgs struct {
		Text string `json:"text"`
	}
	var called int
	got := ""
	echoTool := tools.Func[echoArgs]("echo", "echo", func(_ context.Context, args echoArgs, _ *tools.Container) (any, error) {
		called++
		got = args.Text
		return "ok", nil
	})
	model := &invalidMergedToolContinuationModel{}
	ag, err := New(Config{LLM: model, Tools: []tools.Tool{echoTool}})
	if err != nil {
		t.Fatalf("new agent: %v", err)
	}
	events := collectEvents(ag.QueryStream(context.Background(), llm.TextContent("hi")))
	if model.calls != 4 {
		t.Fatalf("expected 4 model calls (including invalid-merge continuation), got %d", model.calls)
	}
	if called != 1 {
		t.Fatalf("expected tool to be called once after merged args became valid, got %d", called)
	}
	if got != "hello world" {
		t.Fatalf("expected repaired merged args hello world, got %q", got)
	}
	autoContinues := 0
	for _, ev := range events {
		switch e := ev.(type) {
		case AutoContinueEvent:
			if e.Reason != "max_tokens" {
				t.Fatalf("expected max_tokens reason, got %q", e.Reason)
			}
			if strings.TrimSpace(e.ResponseID) == "" {
				t.Fatalf("expected auto-continue event to carry response id")
			}
			autoContinues++
		case TextDeltaEvent:
			if strings.Contains(e.Delta, "[auto-continue]") {
				t.Fatalf("auto-continue marker must not be emitted as text delta")
			}
		}
	}
	if autoContinues < 2 {
		t.Fatalf("expected at least 2 auto-continue prompts, got %d", autoContinues)
	}
	if tr, ok := findToolResult(events, "echo"); !ok || tr.IsError {
		t.Fatalf("expected successful tool result for echo after validation")
	}
	assertHistoryContainsNamedUserMessage(t, ag.Messages(), "Your response was truncated. Please continue exactly where you left off.", "sdk_internal_tool_call_continuation")
}

func TestAgentContinuationTurnLimitEmitsWarningAndResets(t *testing.T) {
	type echoArgs struct {
		Text string `json:"text"`
	}
	echoTool := tools.Func[echoArgs]("echo", "echo", func(_ context.Context, _ echoArgs, _ *tools.Container) (any, error) {
		return "ok", nil
	})
	doneTool := tools.Func[struct {
		Message string `json:"message"`
	}]("done", "done", func(_ context.Context, args struct {
		Message string `json:"message"`
	}, _ *tools.Container) (any, error) {
		return nil, tools.TaskComplete(args.Message)
	})

	model := &continuationLimitModel{}
	ag, err := New(Config{LLM: model, Tools: []tools.Tool{echoTool, doneTool}})
	if err != nil {
		t.Fatalf("new agent: %v", err)
	}

	events := collectEvents(ag.QueryStream(context.Background(), llm.TextContent("hi")))
	if model.calls != 5 {
		t.Fatalf("expected continuation limit flow to reach fifth invoke, got %d", model.calls)
	}

	continuationWarn := 0
	limitWarn := 0
	autoContinue := 0
	final := ""
	for _, ev := range events {
		switch e := ev.(type) {
		case WarnEvent:
			if e.Kind == "continuation" {
				continuationWarn++
			}
			if e.Kind == "continuation_limit" {
				limitWarn++
			}
		case AutoContinueEvent:
			autoContinue++
		case FinalResponseEvent:
			final = e.Content
		}
	}
	if continuationWarn == 0 {
		t.Fatalf("expected continuation warnings before limit")
	}
	if limitWarn != 1 {
		t.Fatalf("expected one continuation_limit warning, got %d", limitWarn)
	}
	if autoContinue != 3 {
		t.Fatalf("expected 3 auto-continue events before limit, got %d", autoContinue)
	}
	if final != "finished via continuation limit" {
		t.Fatalf("expected final response from done tool, got %q", final)
	}
	assertHistoryContainsNamedUserMessage(t, ag.Messages(), "Tool-call arguments are still invalid after continuation. Split the work into smaller tool calls and continue.", "sdk_internal_tool_call_continuation")
}

func TestAgentAutoContinueWaitsForAsyncCompactionAtNextProviderBoundary(t *testing.T) {
	const summaryPrompt = "summarize for compaction"
	const compactionDelay = 200 * time.Millisecond
	compactionDone := make(chan struct{})
	model := &textAutoContinueCompactionModel{
		summaryPrompt:   summaryPrompt,
		compactionDelay: compactionDelay,
		compactionDone:  compactionDone,
	}

	ag, err := New(Config{
		LLM: model,
		Compaction: &compaction.Config{
			Enabled:                true,
			ContextWindow:          200,
			ThresholdRatio:         0.5,
			SummaryPrompt:          summaryPrompt,
			KeepRecentUserMessages: 1,
		},
	})
	if err != nil {
		t.Fatalf("new agent: %v", err)
	}
	if trigger, watermark := ag.compactionTriggerAndWatermark(&llm.Completion{Usage: &llm.Usage{PromptTokens: 90, CompletionTokens: 10, TotalTokens: 100}}); trigger != "usage" || watermark != "summarize" {
		t.Fatalf("precondition compaction trigger=%q watermark=%q, want usage/summarize", trigger, watermark)
	}

	start := time.Now()
	events := collectEvents(ag.QueryStream(context.Background(), llm.TextContent("hi")))
	elapsed := time.Since(start)
	regularCalls, _ := model.Counts()

	if regularCalls != 2 {
		t.Fatalf("expected 2 regular model calls, got %d", regularCalls)
	}
	if elapsed < compactionDelay {
		t.Fatalf("expected next provider boundary to wait for compaction delay (%s), got %s", compactionDelay, elapsed)
	}

	sawCompaction := false
	var compactionResult compaction.Result
	var compactionTriggerUsage *llm.Usage
	autoContinues := 0
	final := ""
	autoContinueRespID := ""
	finalRespID := ""
	for _, ev := range events {
		switch e := ev.(type) {
		case CompactionEvent:
			sawCompaction = true
			compactionResult = e.Result
			compactionTriggerUsage = e.TriggerUsage
		case AutoContinueEvent:
			autoContinueRespID = e.ResponseID
			autoContinues++
		case FinalResponseEvent:
			final = e.Content
			finalRespID = e.ResponseID
		}
	}
	if !sawCompaction {
		t.Fatalf("expected async compaction to apply before the continuation provider call")
	}
	if autoContinues != 1 {
		t.Fatalf("expected one auto-continue metadata event, got %d", autoContinues)
	}
	if autoContinueRespID != "resp_regular_1" {
		t.Fatalf("expected auto-continue response id resp_regular_1, got %q", autoContinueRespID)
	}
	if final != "part 1part 2" {
		t.Fatalf("expected aggregated final response from continuation, got %q", final)
	}
	if finalRespID != "resp_regular_2" {
		t.Fatalf("expected final response id resp_regular_2, got %q", finalRespID)
	}
	assertHistoryContainsNamedUserMessage(t, ag.Messages(), "Your response was truncated. Please continue exactly where you left off.", "sdk_internal_max_tokens_continuation")
	_, compactionCalls := model.Counts()
	if compactionCalls != 1 {
		t.Fatalf("expected 1 compaction model call, got %d", compactionCalls)
	}
	if ag.hasPendingCompaction() {
		t.Fatal("compaction should be applied at the continuation boundary, not left pending")
	}
	if compactionResult.Trigger != "usage" {
		t.Fatalf("compaction trigger = %q, want usage", compactionResult.Trigger)
	}
	if compactionResult.Watermark != "summarize" {
		t.Fatalf("compaction watermark = %q, want summarize", compactionResult.Watermark)
	}
	if len(compactionResult.TiersApplied) != 1 || compactionResult.TiersApplied[0] != "summarize" {
		t.Fatalf("compaction tiers = %#v, want [summarize]", compactionResult.TiersApplied)
	}
	if compactionResult.Usage == nil || compactionResult.Usage.PromptTokens != 90 || compactionResult.Usage.TotalTokens != 100 {
		t.Fatalf("compaction result usage = %#v, want trigger usage", compactionResult.Usage)
	}
	if compactionTriggerUsage == nil || compactionTriggerUsage.PromptTokens != 90 {
		t.Fatalf("legacy trigger usage = %#v, want prompt tokens 90", compactionTriggerUsage)
	}
	if compactionResult.OriginalTokens <= 0 || compactionResult.TokenCountSource != compaction.TokenCountSourceEstimate {
		t.Fatalf("comparable estimate telemetry = %#v", compactionResult)
	}
	if compactionResult.NewTokens <= 0 {
		t.Fatalf("new tokens should be populated, got %d", compactionResult.NewTokens)
	}

	events2 := collectEvents(ag.QueryStream(context.Background(), llm.TextContent("continue")))
	sawCompaction = false
	final = ""
	for _, ev := range events2 {
		switch e := ev.(type) {
		case CompactionEvent:
			sawCompaction = true
		case FinalResponseEvent:
			final = e.Content
		}
	}
	if sawCompaction {
		t.Fatalf("compaction was already applied in the first turn and must not be replayed")
	}
	if final != "follow up" {
		t.Fatalf("expected follow-up final response, got %q", final)
	}

	msgs := ag.Messages()
	foundPartTwo := false
	for _, m := range msgs {
		if m.Role == llm.RoleAssistant && m.Content.PlainText() == "part 2" {
			foundPartTwo = true
			break
		}
	}
	if !foundPartTwo {
		t.Fatalf("expected messages added during async compaction to be preserved")
	}
}

func TestAgentShouldAttemptCompactionOnOverflowBeforeThreshold(t *testing.T) {
	ag, err := New(Config{
		LLM: &stubModel{},
		Compaction: &compaction.Config{
			Enabled:             true,
			ContextWindow:       100,
			ThresholdRatio:      0.99,
			ReserveOutputTokens: 10,
		},
	})
	if err != nil {
		t.Fatalf("new agent: %v", err)
	}
	if !ag.shouldAttemptCompaction(context.Background(), &llm.Completion{Usage: &llm.Usage{PromptTokens: 95, TotalTokens: 95}}) {
		t.Fatal("expected overflow to trigger compaction even below ratio threshold")
	}
}

func TestAgentTruncatesToolResultContentAndMetadata(t *testing.T) {
	const maxBytes = 256
	const dumpTTL = 2 * time.Minute
	large := strings.Repeat("界", 200) // 600 bytes, intentionally exceeds limit.
	tempDir := t.TempDir()
	dumpPath := filepath.Join(tempDir, "tool-result.txt")
	oldDumpWriter := writeToolResultDump
	writeToolResultDump = func(fullOutput string) (string, error) {
		if err := os.WriteFile(dumpPath, []byte(fullOutput), 0o600); err != nil {
			return "", err
		}
		return dumpPath, nil
	}
	t.Cleanup(func() {
		writeToolResultDump = oldDumpWriter
	})

	tool := tools.Tool{
		Name: "echo",
		Handler: func(_ context.Context, _ json.RawMessage, _ *tools.Container) (llm.Content, error) {
			return llm.TextContent(large), nil
		},
	}
	model := &stubModel{toolName: "echo", toolArgs: `{}`}
	ag, err := New(Config{LLM: model, Tools: []tools.Tool{tool}, MaxToolResultBytes: maxBytes, ToolResultDumpTTL: dumpTTL})
	if err != nil {
		t.Fatalf("new agent: %v", err)
	}

	events := collectEvents(ag.QueryStream(context.Background(), llm.TextContent("hi")))
	tr, ok := findToolResult(events, "echo")
	if !ok {
		t.Fatalf("expected tool result event")
	}
	if len(tr.Result) > maxBytes {
		t.Fatalf("expected truncated result <= %d bytes, got %d", maxBytes, len(tr.Result))
	}
	if !strings.Contains(tr.Result, "[WARN] stage=artifact_sink") || !strings.Contains(tr.Result, "complete=false recoverable=false") {
		t.Fatalf("expected explicit non-recoverable artifact diagnostic, got %q", tr.Result)
	}
	if !utf8.ValidString(tr.Result) {
		t.Fatalf("expected valid UTF-8 truncated result")
	}
	if tr.Metadata == nil || tr.Metadata["result_truncated"] != true {
		t.Fatalf("expected result_truncated metadata, got %#v", tr.Metadata)
	}
	if tr.Metadata["result_bytes"] != len(tr.Result) {
		t.Fatalf("expected result_bytes=%d, got %#v", len(tr.Result), tr.Metadata["result_bytes"])
	}
	if tr.Metadata["result_max_bytes"] != maxBytes {
		t.Fatalf("expected result_max_bytes=%d, got %#v", maxBytes, tr.Metadata["result_max_bytes"])
	}
	if tr.Metadata["result_original_bytes"] != len(large) {
		t.Fatalf("expected result_original_bytes=%d, got %#v", len(large), tr.Metadata["result_original_bytes"])
	}
	if tr.Metadata["truncated"] != true {
		t.Fatalf("expected truncated=true metadata, got %#v", tr.Metadata["truncated"])
	}
	if tr.Metadata["originalSize"] != len(large) {
		t.Fatalf("expected originalSize=%d, got %#v", len(large), tr.Metadata["originalSize"])
	}
	if tr.Metadata["outputPath"] != dumpPath {
		t.Fatalf("expected outputPath=%q, got %#v", dumpPath, tr.Metadata["outputPath"])
	}
	if tr.Metadata["result_output_path"] != dumpPath {
		t.Fatalf("expected result_output_path=%q, got %#v", dumpPath, tr.Metadata["result_output_path"])
	}
	if tr.Metadata["result_output_ttl_ms"] != dumpTTL.Milliseconds() {
		t.Fatalf("expected result_output_ttl_ms=%d, got %#v", dumpTTL.Milliseconds(), tr.Metadata["result_output_ttl_ms"])
	}
	expiresRaw, _ := tr.Metadata["result_output_expires_at"].(string)
	if strings.TrimSpace(expiresRaw) == "" {
		t.Fatalf("expected result_output_expires_at metadata")
	}
	expiresAt, err := time.Parse(time.RFC3339, expiresRaw)
	if err != nil {
		t.Fatalf("parse result_output_expires_at: %v", err)
	}
	if !expiresAt.After(time.Now()) {
		t.Fatalf("expected future result_output_expires_at, got %s", expiresAt.Format(time.RFC3339))
	}
	b, err := os.ReadFile(dumpPath)
	if err != nil {
		t.Fatalf("read dump path: %v", err)
	}
	if string(b) != large {
		t.Fatalf("expected dump to contain full original output")
	}

	messages := ag.Messages()
	toolMessageFound := false
	for _, msg := range messages {
		if msg.Role != llm.RoleTool || msg.ToolName != "echo" {
			continue
		}
		toolMessageFound = true
		plain := msg.Content.PlainText()
		if len(plain) > maxBytes {
			t.Fatalf("expected stored tool message <= %d bytes, got %d", maxBytes, len(plain))
		}
		if !strings.Contains(plain, "[WARN] stage=artifact_sink") || !strings.Contains(plain, "complete=false recoverable=false") {
			t.Fatalf("expected stored tool message artifact diagnostic, got %q", plain)
		}
	}
	if !toolMessageFound {
		t.Fatalf("expected stored tool message")
	}
}

func TestToolResultDumpMetadata_IncludesLifecycleFields(t *testing.T) {
	const maxBytes = 80
	const dumpTTL = 90 * time.Second
	large := strings.Repeat("x", maxBytes*2)
	tempDir := t.TempDir()
	dumpPath := filepath.Join(tempDir, "agent-tool-result-metadata.txt")
	fixedNow := time.Unix(1700000000, 0).UTC()

	oldDumpWriter := writeToolResultDump
	oldDumpDir := toolResultDumpDir
	oldDumpNow := toolResultDumpNow
	writeToolResultDump = func(fullOutput string) (string, error) {
		if err := os.WriteFile(dumpPath, []byte(fullOutput), 0o600); err != nil {
			return "", err
		}
		return dumpPath, nil
	}
	toolResultDumpDir = func() string { return tempDir }
	toolResultDumpNow = func() time.Time { return fixedNow }
	t.Cleanup(func() {
		writeToolResultDump = oldDumpWriter
		toolResultDumpDir = oldDumpDir
		toolResultDumpNow = oldDumpNow
	})

	tool := tools.Tool{
		Name: "echo",
		Handler: func(_ context.Context, _ json.RawMessage, _ *tools.Container) (llm.Content, error) {
			return llm.TextContent(large), nil
		},
	}
	model := &stubModel{toolName: "echo", toolArgs: `{}`}
	ag, err := New(Config{
		LLM:                model,
		Tools:              []tools.Tool{tool},
		MaxToolResultBytes: maxBytes,
		ToolResultDumpTTL:  dumpTTL,
	})
	if err != nil {
		t.Fatalf("new agent: %v", err)
	}

	events := collectEvents(ag.QueryStream(context.Background(), llm.TextContent("hi")))
	tr, ok := findToolResult(events, "echo")
	if !ok {
		t.Fatalf("expected tool result event")
	}
	if tr.Metadata["result_output_created_at"] != fixedNow.Format(time.RFC3339) {
		t.Fatalf("expected result_output_created_at=%q, got %#v", fixedNow.Format(time.RFC3339), tr.Metadata["result_output_created_at"])
	}
	expectedExpiresAt := fixedNow.Add(dumpTTL).Format(time.RFC3339)
	if tr.Metadata["result_output_expires_at"] != expectedExpiresAt {
		t.Fatalf("expected result_output_expires_at=%q, got %#v", expectedExpiresAt, tr.Metadata["result_output_expires_at"])
	}
	if tr.Metadata["result_output_expiry_policy"] != toolResultDumpExpiryPolicy {
		t.Fatalf("expected result_output_expiry_policy=%q, got %#v", toolResultDumpExpiryPolicy, tr.Metadata["result_output_expiry_policy"])
	}
}

func TestTruncateStringWithSuffixKeepsUTF8WhenSuffixExceedsBudget(t *testing.T) {
	t.Parallel()

	got := truncateStringWithSuffix(strings.Repeat("你", 8), 7, "🙂🙂")
	if len(got) > 7 {
		t.Fatalf("expected output <= 7 bytes, got %d", len(got))
	}
	if !utf8.ValidString(got) {
		t.Fatalf("expected valid UTF-8 result, got %q", got)
	}
}

func TestMergeToolArgsMergesJSONObjects(t *testing.T) {
	got := mergeToolArgs(`{"file":"test.go"}`, `{"file":"test.go","content":"hello"}`)
	var parsed map[string]any
	if err := json.Unmarshal([]byte(got), &parsed); err != nil {
		t.Fatalf("expected valid JSON, got %q: %v", got, err)
	}
	if parsed["file"] != "test.go" {
		t.Fatalf("expected file=test.go, got %#v", parsed["file"])
	}
	if parsed["content"] != "hello" {
		t.Fatalf("expected content=hello, got %#v", parsed["content"])
	}
}

func TestMergeToolArgsMergesOverlappingFragments(t *testing.T) {
	got := mergeToolArgs(`{"text":"hel`, `el","lang":"en"}`)
	want := `{"text":"hel","lang":"en"}`
	if got != want {
		t.Fatalf("expected merged overlap %q, got %q", want, got)
	}
	var parsed map[string]any
	if err := json.Unmarshal([]byte(got), &parsed); err != nil {
		t.Fatalf("expected valid JSON, got %q: %v", got, err)
	}
}

func TestMergeToolArgsDeepMergesNestedObjectsAndArrays(t *testing.T) {
	got := mergeToolArgs(
		`{"payload":{"nested":{"a":1},"items":[{"id":1,"name":"A"},{"id":2}]}}`,
		`{"payload":{"nested":{"b":2},"items":[{"id":1,"status":"ok"}]}}`,
	)

	var parsed map[string]any
	if err := json.Unmarshal([]byte(got), &parsed); err != nil {
		t.Fatalf("expected valid JSON, got %q: %v", got, err)
	}
	payload, ok := parsed["payload"].(map[string]any)
	if !ok {
		t.Fatalf("expected payload object, got %#v", parsed["payload"])
	}
	nested, ok := payload["nested"].(map[string]any)
	if !ok {
		t.Fatalf("expected nested object, got %#v", payload["nested"])
	}
	if nested["a"] != float64(1) || nested["b"] != float64(2) {
		t.Fatalf("expected deep-merged nested keys, got %#v", nested)
	}
	items, ok := payload["items"].([]any)
	if !ok {
		t.Fatalf("expected items array, got %#v", payload["items"])
	}
	if len(items) != 2 {
		t.Fatalf("expected merged array to preserve unmatched tail, got %d items", len(items))
	}
	first, ok := items[0].(map[string]any)
	if !ok {
		t.Fatalf("expected first array item object, got %#v", items[0])
	}
	if first["id"] != float64(1) || first["name"] != "A" || first["status"] != "ok" {
		t.Fatalf("expected merged first array item, got %#v", first)
	}
	second, ok := items[1].(map[string]any)
	if !ok {
		t.Fatalf("expected second array item object, got %#v", items[1])
	}
	if second["id"] != float64(2) {
		t.Fatalf("expected second array item to remain, got %#v", second)
	}
}

func TestMergeToolArgsWithDiagnosticsReportsShapeConflict(t *testing.T) {
	result := mergeToolArgsWithDiagnostics(
		`{"payload":{"value":{"nested":1}}}`,
		`{"payload":{"value":"override"}}`,
	)

	if len(result.diagnostics) == 0 {
		t.Fatalf("expected merge diagnostics for shape conflict")
	}
	foundPath := false
	for _, diagnostic := range result.diagnostics {
		if strings.Contains(diagnostic, "$.payload.value") {
			foundPath = true
			break
		}
	}
	if !foundPath {
		t.Fatalf("expected conflict path in diagnostics, got %#v", result.diagnostics)
	}
	var parsed map[string]any
	if err := json.Unmarshal([]byte(result.arguments), &parsed); err != nil {
		t.Fatalf("expected valid JSON result, got %q: %v", result.arguments, err)
	}
	payload, _ := parsed["payload"].(map[string]any)
	if payload["value"] != "override" {
		t.Fatalf("expected new scalar value to win conflict, got %#v", payload["value"])
	}
}

func TestToolCallContinuationMergeUsesStableCallIDOnly(t *testing.T) {
	cont := toolCallContinuation{}
	cont.addPartial(0, []llm.ToolCall{{
		ID:   "call_1",
		Type: "function",
		Function: llm.FunctionCall{
			Name:      "echo",
			Arguments: `{"text":"hello`,
		},
	}})

	current := []llm.ToolCall{
		{
			ID:   "call_2",
			Type: "function",
			Function: llm.FunctionCall{
				Name:      "echo",
				Arguments: `{"text":"should-stay"}`,
			},
		},
		{
			ID:   "call_1",
			Type: "function",
			Function: llm.FunctionCall{
				Name:      "echo",
				Arguments: ` world"}`,
			},
		},
	}

	merged := cont.mergeToolCalls(current)
	if len(merged) != 2 {
		t.Fatalf("expected 2 merged calls, got %d", len(merged))
	}
	if merged[0].Function.Arguments != `{"text":"should-stay"}` {
		t.Fatalf("expected different call ID to remain isolated, got %q", merged[0].Function.Arguments)
	}
	if merged[1].Function.Arguments != `{"text":"hello world"}` {
		t.Fatalf("expected matching call ID to merge fragments, got %q", merged[1].Function.Arguments)
	}
}

func TestToolCallContinuationPreservesMergeDiagnosticsByCallID(t *testing.T) {
	cont := toolCallContinuation{}
	cont.addPartial(0, []llm.ToolCall{{
		ID:   "call_1",
		Type: "function",
		Function: llm.FunctionCall{
			Name:      "echo",
			Arguments: `{"payload":{"value":{"nested":1}}}`,
		},
	}})

	merged := cont.mergeToolCalls([]llm.ToolCall{{
		ID:   "call_1",
		Type: "function",
		Function: llm.FunctionCall{
			Name:      "echo",
			Arguments: `{"payload":{"value":"override"}}`,
		},
	}})

	diagnostics := cont.mergeDiagnosticsForCalls(merged)
	if len(diagnostics) == 0 {
		t.Fatalf("expected merge diagnostics for call")
	}
	if len(diagnostics["call_1"]) == 0 {
		t.Fatalf("expected diagnostics for call_1, got %#v", diagnostics)
	}
}

func TestToolCallContinuationClearsPartialAfterIndexShift(t *testing.T) {
	partial := []llm.ToolCall{{
		ID:   "call_1",
		Type: "function",
		Function: llm.FunctionCall{
			Name:      "echo",
			Arguments: `{"text":"hello`,
		},
	}}
	merged := []llm.ToolCall{{
		ID:   "call_1",
		Type: "function",
		Function: llm.FunctionCall{
			Name:      "echo",
			Arguments: `{"text":"hello world"}`,
		},
	}}

	cont := toolCallContinuation{}
	cont.addPartial(9, partial) // stale index after compaction

	messages := []llm.Message{
		{Role: llm.RoleSystem, Content: llm.TextContent("sys")},
		{Role: llm.RoleAssistant, Content: llm.TextContent(""), ToolCalls: cloneToolCalls(partial)},
		{Role: llm.RoleAssistant, Content: llm.TextContent(""), ToolCalls: cloneToolCalls(merged)},
	}

	cont.clearPartialToolCalls(messages, 2)

	if len(messages[1].ToolCalls) != 0 {
		t.Fatalf("expected stale partial assistant message to be cleared")
	}
	if !sameToolCalls(messages[2].ToolCalls, merged) {
		t.Fatalf("expected merged assistant tool calls to stay intact")
	}
}

func TestToolResult_tempCleanupRemovesExpiredArtifacts(t *testing.T) {
	artifactDir := t.TempDir()
	artifactPath := filepath.Join(artifactDir, "agent-tool-result-1.txt")
	if err := os.WriteFile(artifactPath, []byte("tool-result"), 0o600); err != nil {
		t.Fatalf("write artifact: %v", err)
	}

	ag, err := New(Config{LLM: &stubModel{}, ToolResultDumpTTL: 50 * time.Millisecond})
	if err != nil {
		t.Fatalf("new agent: %v", err)
	}
	now := time.Unix(1700000000, 0)
	_ = ag.registerToolResultDump(artifactPath, now)

	ag.cleanupToolResultDumps(now.Add(40*time.Millisecond), false)
	if _, err := os.Stat(artifactPath); err != nil {
		t.Fatalf("expected artifact to remain before ttl expiry: %v", err)
	}

	ag.cleanupToolResultDumps(now.Add(60*time.Millisecond), false)
	if _, err := os.Stat(artifactPath); !os.IsNotExist(err) {
		t.Fatalf("expected artifact removed after ttl expiry, got err=%v", err)
	}
	ag.toolResultDumpsMu.Lock()
	_, tracked := ag.toolResultDumps[artifactPath]
	ag.toolResultDumpsMu.Unlock()
	if tracked {
		t.Fatalf("expected artifact tracker entry to be cleaned")
	}
}

func TestToolResultDumpCleanup_ReclaimsOrphansFromPreviousProcess(t *testing.T) {
	const dumpTTL = time.Minute
	fixedNow := time.Unix(1700000000, 0).UTC()
	artifactDir := t.TempDir()
	expiredOrphan := filepath.Join(artifactDir, "agent-tool-result-expired-orphan.txt")
	freshOrphan := filepath.Join(artifactDir, "agent-tool-result-fresh-orphan.txt")
	expiredIndexed := filepath.Join(artifactDir, "agent-tool-result-expired-indexed.txt")

	writeArtifact := func(path, body string) {
		if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
			t.Fatalf("write artifact %q: %v", path, err)
		}
	}
	writeArtifact(expiredOrphan, "expired-orphan")
	writeArtifact(freshOrphan, "fresh-orphan")
	writeArtifact(expiredIndexed, "expired-indexed")

	expiredMTime := fixedNow.Add(-3 * dumpTTL)
	freshMTime := fixedNow.Add(-dumpTTL / 2)
	if err := os.Chtimes(expiredOrphan, expiredMTime, expiredMTime); err != nil {
		t.Fatalf("chtimes expired orphan: %v", err)
	}
	if err := os.Chtimes(freshOrphan, freshMTime, freshMTime); err != nil {
		t.Fatalf("chtimes fresh orphan: %v", err)
	}
	if err := os.Chtimes(expiredIndexed, expiredMTime, expiredMTime); err != nil {
		t.Fatalf("chtimes expired indexed: %v", err)
	}

	staleIndexPath := filepath.Join(artifactDir, "agent-tool-result-index-stale.json")
	staleIndex := toolResultDumpIndexFile{
		Version:   toolResultDumpIndexVersion,
		SessionID: "stale-session",
		Dumps: []toolResultDumpIndexEntry{{
			Path:      expiredIndexed,
			CreatedAt: expiredMTime.UTC().Format(time.RFC3339),
			ExpiresAt: fixedNow.Add(-time.Second).UTC().Format(time.RFC3339),
		}},
	}
	idxBytes, err := json.Marshal(staleIndex)
	if err != nil {
		t.Fatalf("marshal stale index: %v", err)
	}
	if err := os.WriteFile(staleIndexPath, idxBytes, 0o600); err != nil {
		t.Fatalf("write stale index: %v", err)
	}

	oldDumpDir := toolResultDumpDir
	oldDumpNow := toolResultDumpNow
	toolResultDumpDir = func() string { return artifactDir }
	toolResultDumpNow = func() time.Time { return fixedNow }
	t.Cleanup(func() {
		toolResultDumpDir = oldDumpDir
		toolResultDumpNow = oldDumpNow
	})

	if _, err := New(Config{LLM: &stubModel{}, ToolResultDumpTTL: dumpTTL}); err != nil {
		t.Fatalf("new agent: %v", err)
	}

	if _, err := os.Stat(expiredOrphan); !os.IsNotExist(err) {
		t.Fatalf("expected expired orphan dump to be reclaimed, got err=%v", err)
	}
	if _, err := os.Stat(expiredIndexed); !os.IsNotExist(err) {
		t.Fatalf("expected expired indexed dump to be reclaimed, got err=%v", err)
	}
	if _, err := os.Stat(staleIndexPath); !os.IsNotExist(err) {
		t.Fatalf("expected stale index file to be reclaimed, got err=%v", err)
	}
	if _, err := os.Stat(freshOrphan); err != nil {
		t.Fatalf("expected fresh orphan dump to remain before TTL expiry, got err=%v", err)
	}
}

func TestToolResult_cleanupAllOnClearHistory(t *testing.T) {
	artifactDir := t.TempDir()
	artifactPath := filepath.Join(artifactDir, "agent-tool-result-2.txt")
	if err := os.WriteFile(artifactPath, []byte("tool-result"), 0o600); err != nil {
		t.Fatalf("write artifact: %v", err)
	}

	ag, err := New(Config{LLM: &stubModel{}, ToolResultDumpTTL: time.Hour})
	if err != nil {
		t.Fatalf("new agent: %v", err)
	}
	_ = ag.registerToolResultDump(artifactPath, time.Now())
	ag.ReplaceHistory([]llm.Message{llm.NewUserMessage("hi")})

	if _, err := os.Stat(artifactPath); !os.IsNotExist(err) {
		t.Fatalf("expected ReplaceHistory to cleanup tracked artifact, got err=%v", err)
	}
	ag.toolResultDumpsMu.Lock()
	remaining := len(ag.toolResultDumps)
	ag.toolResultDumpsMu.Unlock()
	if remaining != 0 {
		t.Fatalf("expected artifact tracker to be empty, got %d entries", remaining)
	}
}

func TestDestroyEphemeralMessagesScansBeyondLastPromptCount(t *testing.T) {
	ag := &Agent{
		toolMap: map[string]tools.Tool{
			"search": {Name: "search", EphemeralKeep: 1},
		},
		messages: []llm.Message{
			llm.NewUserMessage("u1"),
			{Role: llm.RoleTool, ToolName: "search", Content: llm.TextContent("tool-1"), Ephemeral: true},
			llm.NewAssistantMessage("a1", nil),
			{Role: llm.RoleTool, ToolName: "search", Content: llm.TextContent("tool-2"), Ephemeral: true},
			{Role: llm.RoleTool, ToolName: "search", Content: llm.TextContent("tool-3"), Ephemeral: true},
		},
		lastPromptCount: 2,
	}

	ag.destroyEphemeralMessages()

	if !ag.messages[1].Destroyed || ag.messages[1].Content.PlainText() != ephemeralReleasedPlaceholder {
		t.Fatalf("expected first ephemeral message to be destroyed, got %#v", ag.messages[1])
	}
	if !ag.messages[3].Destroyed || ag.messages[3].Content.PlainText() != ephemeralReleasedPlaceholder {
		t.Fatalf("expected post-cutoff ephemeral message to be destroyed, got %#v", ag.messages[3])
	}
	if ag.messages[4].Destroyed || ag.messages[4].Content.PlainText() != "tool-3" {
		t.Fatalf("expected newest ephemeral message to be preserved, got %#v", ag.messages[4])
	}
}

func TestDestroyEphemeralMessagesResetsTrackingAfterReplaceHistory(t *testing.T) {
	ag := &Agent{
		toolMap: map[string]tools.Tool{
			"search": {Name: "search", EphemeralKeep: 1},
		},
		messages: []llm.Message{
			llm.NewUserMessage("u1"),
			{Role: llm.RoleTool, ToolName: "search", Content: llm.TextContent("old"), Ephemeral: true},
		},
		ephemeralByKey: make(map[string][]int),
	}

	// Prime tracking state so ReplaceHistory must clear stale indices/cursor.
	ag.destroyEphemeralMessages()

	ag.ReplaceHistory([]llm.Message{
		{Role: llm.RoleTool, ToolName: "search", Content: llm.TextContent("new-1"), Ephemeral: true},
		{Role: llm.RoleTool, ToolName: "search", Content: llm.TextContent("new-2"), Ephemeral: true},
	})
	ag.destroyEphemeralMessages()

	if !ag.messages[0].Destroyed || ag.messages[0].Content.PlainText() != ephemeralReleasedPlaceholder {
		t.Fatalf("expected oldest replaced-history ephemeral message to be destroyed, got %#v", ag.messages[0])
	}
	if ag.messages[1].Destroyed || ag.messages[1].Content.PlainText() != "new-2" {
		t.Fatalf("expected latest replaced-history ephemeral message to be preserved, got %#v", ag.messages[1])
	}
}

// TestDestroyEphemeralMessagesKeepsDistinctTargets verifies the structural fix
// for the self-bootstrap read loop: reading several *different* targets (each a
// distinct path/offset/limit signature) must not evict one another, so a later
// re-read never comes back as a placeholder just because other files were read
// in between. Only redundant re-reads of the *same* signature collapse.
func TestDestroyEphemeralMessagesKeepsDistinctTargets(t *testing.T) {
	callA1 := llm.ToolCall{ID: "a1", Function: llm.FunctionCall{Name: "read", Arguments: `{"filePath":"a.go"}`}}
	callB := llm.ToolCall{ID: "b", Function: llm.FunctionCall{Name: "read", Arguments: `{"filePath":"b.go"}`}}
	callA2 := llm.ToolCall{ID: "a2", Function: llm.FunctionCall{Name: "read", Arguments: `{"filePath":"a.go"}`}}

	ag := &Agent{
		toolMap: map[string]tools.Tool{
			"read": {Name: "read", EphemeralKeep: 1},
		},
		messages: []llm.Message{
			llm.NewUserMessage("u1"),
			llm.NewAssistantMessage("read a", []llm.ToolCall{callA1}),
			{Role: llm.RoleTool, ToolCallID: "a1", ToolName: "read", Content: llm.TextContent("A-first"), Ephemeral: true},
			llm.NewAssistantMessage("read b", []llm.ToolCall{callB}),
			{Role: llm.RoleTool, ToolCallID: "b", ToolName: "read", Content: llm.TextContent("B-content"), Ephemeral: true},
			llm.NewAssistantMessage("re-read a", []llm.ToolCall{callA2}),
			{Role: llm.RoleTool, ToolCallID: "a2", ToolName: "read", Content: llm.TextContent("A-second"), Ephemeral: true},
		},
	}

	ag.destroyEphemeralMessages()

	// b.go was read once and never re-read: it must survive intact even though
	// a.go was read after it (the old tool-name grouping would have evicted it).
	if ag.messages[4].Destroyed || ag.messages[4].Content.PlainText() != "B-content" {
		t.Fatalf("expected distinct-target read to be preserved, got %#v", ag.messages[4])
	}
	// The newest a.go re-read must be readable (real content, not placeholder).
	if ag.messages[6].Destroyed || ag.messages[6].Content.PlainText() != "A-second" {
		t.Fatalf("expected newest same-signature read to be preserved, got %#v", ag.messages[6])
	}
	// The stale earlier a.go read (same signature) is the only one recycled.
	if !ag.messages[2].Destroyed || ag.messages[2].Content.PlainText() != ephemeralReleasedPlaceholder {
		t.Fatalf("expected stale same-signature read to be recycled, got %#v", ag.messages[2])
	}
}

func TestWithPreservedSystemDeduplicatesSystemMessages(t *testing.T) {
	ag := &Agent{}
	orig := []llm.Message{
		llm.NewSystemMessage("system rules"),
		llm.NewSystemMessage("  system rules  "),
		llm.NewUserMessage("user-1"),
	}
	compacted := []llm.Message{llm.NewUserMessage("summary")}

	out := ag.withPreservedSystem(orig, compacted)
	if len(out) != 2 {
		t.Fatalf("expected 2 messages after dedupe, got %d", len(out))
	}
	if out[0].Role != llm.RoleSystem || out[0].Content.PlainText() != "system rules" {
		t.Fatalf("expected single preserved system message, got %#v", out[0])
	}
	if out[1].Role != llm.RoleUser || out[1].Content.PlainText() != "summary" {
		t.Fatalf("expected compacted message to remain, got %#v", out[1])
	}
}

func TestWithPreservedSystemDeduplicatesCompactedSystemMessages(t *testing.T) {
	ag := &Agent{}
	orig := []llm.Message{
		llm.NewSystemMessage("system rules"),
		llm.NewUserMessage("user-1"),
	}
	compacted := []llm.Message{
		llm.NewSystemMessage("  system rules  "),
		llm.NewSystemMessage("system extension"),
		llm.NewUserMessage("summary"),
	}

	out := ag.withPreservedSystem(orig, compacted)
	if len(out) != 3 {
		t.Fatalf("expected deduped systems + summary, got %d messages", len(out))
	}
	if out[0].Role != llm.RoleSystem || out[0].Content.PlainText() != "system rules" {
		t.Fatalf("expected preserved original system first, got %#v", out[0])
	}
	if out[1].Role != llm.RoleSystem || out[1].Content.PlainText() != "system extension" {
		t.Fatalf("expected unique compacted system retained once, got %#v", out[1])
	}
	if out[2].Role != llm.RoleUser || out[2].Content.PlainText() != "summary" {
		t.Fatalf("expected non-system compacted message retained, got %#v", out[2])
	}
}
