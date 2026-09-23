package agent

import (
	"context"
	"strings"
	"sync"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/agent/compaction"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

func typedOverflow() error {
	return &llm.ProviderError{Provider: "fixture", StatusCode: 400, Message: "input too long", Reason: llm.ProviderErrorReasonContextOverflow}
}

// overflowScriptModel answers summary requests with a valid checkpoint and
// main requests from a script; each step is a completion or an error.
type overflowScriptModel struct {
	mu       sync.Mutex
	script   []func() (*llm.Completion, error)
	main     []llm.InvokeRequest
	summary  int
	onInvoke func(n int)
}

func (m *overflowScriptModel) Provider() string { return "fixture" }
func (m *overflowScriptModel) Model() string    { return "overflow-script" }
func (m *overflowScriptModel) Invoke(_ context.Context, req llm.InvokeRequest) (*llm.Completion, error) {
	owned, err := llm.CloneInvokeRequest(req)
	if err != nil {
		return nil, err
	}
	m.mu.Lock()
	for _, message := range owned.Messages {
		if message.Role == llm.RoleSystem && strings.Contains(message.Content.PlainText(), "operational checkpoint") {
			m.summary++
			m.mu.Unlock()
			return &llm.Completion{Content: llm.TextContent(validCompactionSummary("prior work summarized"))}, nil
		}
	}
	m.main = append(m.main, owned)
	n := len(m.main)
	step := func() (*llm.Completion, error) { return &llm.Completion{Content: llm.TextContent("ok")}, nil }
	if n <= len(m.script) {
		step = m.script[n-1]
	}
	hook := m.onInvoke
	m.mu.Unlock()
	if hook != nil {
		hook(n)
	}
	return step()
}

func (m *overflowScriptModel) snapshot() ([]llm.InvokeRequest, int) {
	m.mu.Lock()
	defer m.mu.Unlock()
	return append([]llm.InvokeRequest(nil), m.main...), m.summary
}

func longHistory() []llm.Message {
	msgs := []llm.Message{llm.NewSystemMessage("system")}
	for i := 0; i < 8; i++ {
		msgs = append(msgs, llm.NewUserMessage(strings.Repeat("earlier request ", 60)), llm.NewAssistantMessage(strings.Repeat("earlier answer ", 60), nil))
	}
	return msgs
}

func overflowAgent(t *testing.T, model llm.ChatModel, mutate func(*Config)) *Agent {
	t.Helper()
	cfg := Config{
		LLM:                    model,
		InitialMessages:        longHistory(),
		InvokeRetryMaxAttempts: 1,
		Compaction:             &compaction.Config{Enabled: true, ContextWindow: 100000, ThresholdRatio: 0.85},
		Warningf:               func(string, ...any) {},
	}
	if mutate != nil {
		mutate(&cfg)
	}
	ag, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	return ag
}

type overflowRun struct {
	errors, compactions, recoveries, finals  int
	lastError                                ErrorEvent
	warnFrame, finalFrame                    string
	finalHistoryRelation, finalHistorySource string
}

func runOverflowQuery(ag *Agent, steering <-chan SteeringMsg) overflowRun {
	var r overflowRun
	for env := range ag.QueryStreamEnvelopedWithSteering(context.Background(), llm.TextContent("current request"), steering) {
		switch e := env.Event.(type) {
		case ErrorEvent:
			r.errors++
			r.lastError = e
		case CompactionEvent:
			r.compactions++
		case WarnEvent:
			if e.Kind == "context_overflow_recovery" {
				r.recoveries++
				r.warnFrame = env.FrameID
			}
		case FinalResponseEvent:
			r.finals++
			r.finalFrame = env.FrameID
			r.finalHistoryRelation, r.finalHistorySource = env.RequestHistoryRelation, env.RequestHistorySourceFrameID
		}
	}
	return r
}

// A typed overflow is recovered once: history is compacted and a new, smaller
// request under a new Frame succeeds.
func TestTypedContextOverflowRecoversOnceWithNewFrame(t *testing.T) {
	model := &overflowScriptModel{script: []func() (*llm.Completion, error){
		func() (*llm.Completion, error) { return nil, typedOverflow() },
	}}
	ag := overflowAgent(t, model, nil)
	r := runOverflowQuery(ag, nil)
	main, summaries := model.snapshot()
	if len(main) != 2 || r.finals != 1 || r.errors != 0 || r.compactions != 1 || r.recoveries != 1 {
		t.Fatalf("main=%d run=%+v summaries=%d", len(main), r, summaries)
	}
	if llm.EstimateMessagesTokens(main[1].Messages) >= llm.EstimateMessagesTokens(main[0].Messages) {
		t.Fatal("retried request is not smaller than the rejected one")
	}
	if r.warnFrame == "" || r.finalFrame == "" || r.warnFrame == r.finalFrame {
		t.Fatalf("recovery reused the rejected Frame: warn=%q final=%q", r.warnFrame, r.finalFrame)
	}
	// Lineage: the retried request records the applied compaction and names
	// the rejected Frame as its source (the producer of the change).
	if r.finalHistoryRelation != RequestHistoryCompactionApplied || r.finalHistorySource != r.warnFrame {
		t.Fatalf("retried request lineage=%q source=%q, want compaction from %q", r.finalHistoryRelation, r.finalHistorySource, r.warnFrame)
	}
	if !strings.Contains(main[1].Messages[len(main[1].Messages)-1].Content.PlainText(), "current request") {
		t.Fatal("the user's request was lost by recovery")
	}
}

// A second typed overflow in the same user epoch ends the turn with the
// provider error; there is no second compaction or third request.
func TestTypedContextOverflowSecondRejectionIsTerminal(t *testing.T) {
	model := &overflowScriptModel{script: []func() (*llm.Completion, error){
		func() (*llm.Completion, error) { return nil, typedOverflow() },
		func() (*llm.Completion, error) { return nil, typedOverflow() },
		func() (*llm.Completion, error) { return nil, typedOverflow() },
	}}
	ag := overflowAgent(t, model, nil)
	r := runOverflowQuery(ag, nil)
	main, _ := model.snapshot()
	if len(main) != 2 || r.compactions != 1 || r.recoveries != 1 || r.errors != 1 || r.finals != 0 {
		t.Fatalf("main=%d run=%+v", len(main), r)
	}
	if r.lastError.StatusCode != 400 || r.lastError.Kind != "invalid_request" {
		t.Fatalf("terminal error=%+v", r.lastError)
	}
}

// Without typed evidence, when disabled, or when nothing can be compacted,
// the rejection stays terminal after exactly one request.
func TestContextOverflowRecoveryRequiresTypedEvidenceAndChange(t *testing.T) {
	untyped := func() (*llm.Completion, error) {
		return nil, &llm.ProviderError{Provider: "fixture", StatusCode: 400, Message: "This model's maximum context length is 100 tokens"}
	}
	cases := map[string]struct {
		step   func() (*llm.Completion, error)
		mutate func(*Config)
	}{
		"untyped 400 message":        {untyped, nil},
		"typed but disabled":         {func() (*llm.Completion, error) { return nil, typedOverflow() }, func(c *Config) { c.DisableContextOverflowRecovery = true }},
		"typed, nothing compactable": {func() (*llm.Completion, error) { return nil, typedOverflow() }, func(c *Config) { c.Compaction = &compaction.Config{Enabled: false} }},
		"rate limit": {func() (*llm.Completion, error) {
			return nil, &llm.RateLimitError{Provider: "fixture", Message: "context length"}
		}, nil},
		"typed after partial": {func() (*llm.Completion, error) {
			return &llm.Completion{Content: llm.TextContent("partial")}, typedOverflow()
		}, nil},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			model := &overflowScriptModel{script: []func() (*llm.Completion, error){tc.step, tc.step}}
			ag := overflowAgent(t, model, tc.mutate)
			r := runOverflowQuery(ag, nil)
			main, summaries := model.snapshot()
			if len(main) != 1 || r.compactions != 0 || r.recoveries != 0 || r.errors != 1 || summaries != 0 {
				t.Fatalf("main=%d run=%+v summaries=%d", len(main), r, summaries)
			}
		})
	}
}

// A real user steering message starts a new epoch, so a later overflow may be
// recovered again; internal continuation of the same epoch may not.
func TestContextOverflowRecoveryBudgetFollowsUserEpoch(t *testing.T) {
	for _, steer := range []bool{false, true} {
		t.Run(map[bool]string{false: "same epoch", true: "after steering"}[steer], func(t *testing.T) {
			steering := make(chan SteeringMsg, 1)
			noop := tools.Func[struct{}]("noop", "noop", func(context.Context, struct{}, *tools.Container) (any, error) {
				if steer {
					steering <- SteeringMsg{Content: "new user input"}
				}
				return "done", nil
			})
			model := &overflowScriptModel{script: []func() (*llm.Completion, error){
				func() (*llm.Completion, error) { return nil, typedOverflow() },
				func() (*llm.Completion, error) {
					return &llm.Completion{ToolCalls: []llm.ToolCall{{ID: "c1", Type: "function", Function: llm.FunctionCall{Name: "noop", Arguments: "{}"}}}}, nil
				},
				func() (*llm.Completion, error) { return nil, typedOverflow() },
			}}
			ag := overflowAgent(t, model, func(c *Config) { c.Tools = []tools.Tool{noop} })
			r := runOverflowQuery(ag, steering)
			main, _ := model.snapshot()
			if steer {
				if len(main) != 4 || r.recoveries != 2 || r.finals != 1 || r.errors != 0 {
					t.Fatalf("main=%d run=%+v", len(main), r)
				}
				return
			}
			if len(main) != 3 || r.recoveries != 1 || r.errors != 1 || r.finals != 0 {
				t.Fatalf("main=%d run=%+v", len(main), r)
			}
		})
	}
}
