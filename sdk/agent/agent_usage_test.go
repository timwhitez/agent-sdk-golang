package agent

import (
	"context"
	"sync"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

type zeroPromptUsageModel struct {
	mu    sync.Mutex
	calls int
}

func (m *zeroPromptUsageModel) Provider() string { return "fixture" }
func (m *zeroPromptUsageModel) Model() string    { return "zero-prompt" }

func (m *zeroPromptUsageModel) Invoke(context.Context, llm.InvokeRequest) (*llm.Completion, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.calls++
	if m.calls == 1 {
		return &llm.Completion{
			ToolCalls: []llm.ToolCall{{
				ID:   "read-1",
				Type: "function",
				Function: llm.FunctionCall{
					Name:      "read",
					Arguments: `{"filePath":"a.go"}`,
				},
			}},
			Usage: llm.NewProviderUsage(0, 5, 5),
		}, nil
	}
	return &llm.Completion{
		ToolCalls: []llm.ToolCall{{
			ID:   "done-1",
			Type: "function",
			Function: llm.FunctionCall{
				Name:      "done",
				Arguments: `{"message":"complete"}`,
			},
		}},
		Usage: llm.NewProviderUsage(0, 7, 7),
	}, nil
}

type usageReadArgs struct {
	FilePath string `json:"filePath"`
}

type usageDoneArgs struct {
	Message string `json:"message"`
}

func TestAgentZeroPromptUsageFallsBackOncePerQuery(t *testing.T) {
	readTool := tools.Func[usageReadArgs]("read", "read", func(context.Context, usageReadArgs, *tools.Container) (any, error) {
		return "package fixture", nil
	})
	doneTool := tools.Func[usageDoneArgs]("done", "done", func(_ context.Context, args usageDoneArgs, _ *tools.Container) (any, error) {
		return nil, &tools.TaskCompleteError{Message: args.Message}
	})
	ag, err := New(Config{
		LLM:             &zeroPromptUsageModel{},
		Tools:           []tools.Tool{readTool, doneTool},
		SystemPrompt:    "system prompt",
		MaxIterations:   -1,
		RequireDoneTool: true,
	})
	if err != nil {
		t.Fatal(err)
	}

	var usages []UsageEvent
	warnings := 0
	var warningMetadata map[string]any
	for ev := range ag.QueryStream(context.Background(), llm.TextContent("inspect")) {
		switch e := ev.(type) {
		case UsageEvent:
			usages = append(usages, e)
		case WarnEvent:
			if e.Kind == "provider_usage_prompt_tokens_missing" {
				warnings++
				warningMetadata = e.Metadata
			}
		}
	}
	if len(usages) != 2 {
		t.Fatalf("usage events = %d, want 2", len(usages))
	}
	if warnings != 1 {
		t.Fatalf("missing-usage warnings = %d, want 1", warnings)
	}
	if warningMetadata["prompt_tokens_source"] != llm.PromptTokensSourceEstimate {
		t.Fatalf("missing-usage warning metadata = %#v", warningMetadata)
	}
	for i, event := range usages {
		u := event.Usage
		if u.PromptTokens <= 0 || u.TotalTokens != u.PromptTokens+u.CompletionTokens {
			t.Fatalf("usage[%d] effective totals = %#v", i, u)
		}
		if u.PromptTokensSource != llm.PromptTokensSourceEstimate || u.PromptTokensValid {
			t.Fatalf("usage[%d] quality = %#v", i, u)
		}
		if u.PromptTokensSemantics != llm.PromptTokensSemanticsTotalInputV1 {
			t.Fatalf("usage[%d] semantics = %q", i, u.PromptTokensSemantics)
		}
		if u.ProviderPromptTokens == nil || *u.ProviderPromptTokens != 0 {
			t.Fatalf("usage[%d] provider prompt = %#v", i, u.ProviderPromptTokens)
		}
	}
}

type noUsageModel struct{}

func (*noUsageModel) Provider() string { return "fixture" }
func (*noUsageModel) Model() string    { return "no-usage" }
func (*noUsageModel) Invoke(context.Context, llm.InvokeRequest) (*llm.Completion, error) {
	return &llm.Completion{Content: llm.TextContent("complete")}, nil
}

func TestAgentDoesNotInventUsageWhenProviderOmitsUsageObject(t *testing.T) {
	ag, err := New(Config{LLM: &noUsageModel{}, MaxIterations: 1})
	if err != nil {
		t.Fatal(err)
	}
	usageEvents := 0
	missingWarnings := 0
	for ev := range ag.QueryStream(context.Background(), llm.TextContent("hello")) {
		switch e := ev.(type) {
		case UsageEvent:
			usageEvents++
		case WarnEvent:
			if e.Kind == "provider_usage_prompt_tokens_missing" {
				missingWarnings++
			}
		}
	}
	if usageEvents != 0 || missingWarnings != 0 {
		t.Fatalf("usage events=%d missing warnings=%d, want zero", usageEvents, missingWarnings)
	}
}
