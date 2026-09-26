package agent

import (
	"context"
	"encoding/json"
	"regexp"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

// continuationObservation keeps, per Frame, the continuation sets its
// envelopes reported (joined) and the invoke attempts seen, plus the other
// request relations the same Frame carried.
type continuationObservation struct {
	sets     map[string]map[string]bool
	attempts map[string]map[uint64]bool
	steering map[string]string
	recovery map[string]string
	order    []string
}

func observeContinuation(t *testing.T, events <-chan EventEnvelope) continuationObservation {
	t.Helper()
	observed := continuationObservation{sets: map[string]map[string]bool{}, attempts: map[string]map[uint64]bool{}, steering: map[string]string{}, recovery: map[string]string{}}
	var previous []string
	for env := range events {
		if e, ok := env.Event.(ErrorEvent); ok {
			t.Errorf("unexpected error event: %+v", e)
		}
		related := env.RequestContinuationRelation != ""
		if related != (len(env.RequestContinuationSourceFrameIDs) > 0) {
			t.Errorf("half continuation relation: %q %v", env.RequestContinuationRelation, env.RequestContinuationSourceFrameIDs)
		}
		if related && env.RequestContinuationRelation != RequestContinuationToolResults {
			t.Errorf("relation=%q", env.RequestContinuationRelation)
		}
		if env.FrameID == "" {
			if related {
				t.Errorf("uncorrelated envelope carries a continuation: %+v", env)
			}
			continue
		}
		// Every envelope owns its copy: a consumer mutating one never
		// changes another.
		if len(previous) > 0 {
			previous[0] = "MUTATED"
		}
		for _, id := range env.RequestContinuationSourceFrameIDs {
			if id == "MUTATED" {
				t.Errorf("frame %s shares its continuation set with an earlier envelope", env.FrameID)
			}
		}
		previous = env.RequestContinuationSourceFrameIDs
		if observed.sets[env.FrameID] == nil {
			observed.sets[env.FrameID] = map[string]bool{}
			observed.attempts[env.FrameID] = map[uint64]bool{}
			observed.order = append(observed.order, env.FrameID)
		}
		observed.sets[env.FrameID][strings.Join(env.RequestContinuationSourceFrameIDs, ",")] = true
		observed.attempts[env.FrameID][env.InvokeAttempt] = true
		if env.RequestSteeringRelation != "" {
			observed.steering[env.FrameID] = env.RequestSteeringSourceFrameID
		}
		if env.RequestRecoveryRelation != "" {
			observed.recovery[env.FrameID] = env.RequestRecoverySourceFrameID
		}
	}
	return observed
}

// wantContinuation asserts that every envelope of each listed Frame reports
// exactly the listed set ("" = none) and that each listed Frame was observed.
func wantContinuation(t *testing.T, observed continuationObservation, want map[string]string) {
	t.Helper()
	for frame, set := range want {
		sets, seen := observed.sets[frame]
		if !seen {
			t.Errorf("frame %s emitted no events (frames %v)", frame, observed.order)
			continue
		}
		if len(sets) != 1 || !sets[set] {
			t.Errorf("frame %s reported continuation sets %v, want only %q", frame, sets, set)
		}
	}
}

func lineageCall(id, name, args, stop string) func() (*llm.Completion, error) {
	return func() (*llm.Completion, error) {
		return &llm.Completion{StopReason: stop, Usage: &llm.Usage{PromptTokens: 10, CompletionTokens: 1, TotalTokens: 11},
			ToolCalls: []llm.ToolCall{{ID: id, Type: "function", Function: llm.FunctionCall{Name: name, Arguments: args}}}}, nil
	}
}

func requestAnswersCall(req llm.InvokeRequest, callID string) bool {
	for _, message := range req.Messages {
		if message.Role == llm.RoleTool && message.ToolCallID == callID {
			return true
		}
	}
	return false
}

var lineageFrameIDPattern = regexp.MustCompile(`^lineage-query/frame/[1-9][0-9]*$`)

// #182: the Frame after a tool block carries the block's results and names
// the Frame whose response produced the answered calls; its SDK retry reuses
// that set; the Frame after an accepted response to them names only its own
// predecessor block; a new Query starts empty. Only Frame IDs are reported.
func TestToolContinuationNamesProducingFrameAndRetryReusesIt(t *testing.T) {
	model := &steeringRelationModel{steps: []func() (*llm.Completion, error){
		lineageCall("SECRET_CALL_1", "work", `{"q":"SECRET_ARG"}`, "tool_calls"), relationTransient,
		lineageCall("SECRET_CALL_2", "work", `{"q":"SECRET_ARG"}`, "tool_calls"), steeringFinal,
	}}
	work := tools.Func[struct {
		Q string `json:"q"`
	}]("work", "fixture", func(context.Context, struct {
		Q string `json:"q"`
	}, *tools.Container) (any, error) {
		return "SECRET_RESULT", nil
	})
	ag, err := New(Config{LLM: model, Tools: []tools.Tool{work}, InvokeRetryMaxAttempts: 2, InvokeRetryBackoff: time.Nanosecond,
		Warningf: func(string, ...any) {}, QueryIDGenerator: func() string { return "lineage-query" }})
	if err != nil {
		t.Fatal(err)
	}
	// Privacy: the relation holds Frame IDs only, never CallIDs, arguments
	// or results. Checked before the observer touches the envelope.
	events := make(chan EventEnvelope)
	go func() {
		defer close(events)
		for env := range ag.QueryStreamEnveloped(context.Background(), llm.TextContent("start")) {
			fields, err := json.Marshal(struct {
				R string
				S []string
			}{env.RequestContinuationRelation, env.RequestContinuationSourceFrameIDs})
			if err != nil || strings.Contains(string(fields), "SECRET") {
				t.Errorf("continuation fields leak content: %s %v", fields, err)
			}
			for _, id := range env.RequestContinuationSourceFrameIDs {
				if !lineageFrameIDPattern.MatchString(id) {
					t.Errorf("source %q is not a Frame ID", id)
				}
			}
			events <- env
		}
	}()
	observed := observeContinuation(t, events)
	wantContinuation(t, observed, map[string]string{
		"lineage-query/frame/1": "",
		"lineage-query/frame/2": "lineage-query/frame/1",
		"lineage-query/frame/3": "lineage-query/frame/2",
	})
	// The failed first attempt emits nothing; its retry is attempt 2.
	if !observed.attempts["lineage-query/frame/2"][2] {
		t.Fatalf("retried Frame attempts=%v", observed.attempts["lineage-query/frame/2"])
	}
	// Independent oracle: the provider saw call 1's result first in both
	// attempts of Frame 2 and call 2's result first in Frame 3.
	requests := model.recorded()
	if len(requests) != 4 {
		t.Fatalf("requests=%d", len(requests))
	}
	for i, want := range [][2]bool{{false, false}, {true, false}, {true, false}, {true, true}} {
		if requestAnswersCall(requests[i], "SECRET_CALL_1") != want[0] || requestAnswersCall(requests[i], "SECRET_CALL_2") != want[1] {
			t.Fatalf("request[%d] answered calls differ from %v", i, want)
		}
	}
	model.mu.Lock()
	model.steps = []func() (*llm.Completion, error){steeringFinal}
	model.mu.Unlock()
	fresh := observeContinuation(t, ag.QueryStreamEnveloped(context.Background(), llm.TextContent("again")))
	wantContinuation(t, fresh, map[string]string{"lineage-query/frame/1": ""})
}

// A tool call whose truncated arguments were merged across Frames names
// every Frame whose fragment was merged into it, not merely the nearest
// Frame; a fragment whose ID the provider rotated was never merged into the
// answered call, so its Frame is not named.
func TestMergedToolContinuationNamesEveryFragmentFrame(t *testing.T) {
	type echoArgs struct {
		Text string `json:"text"`
	}
	cases := map[string]struct {
		secondID string
		wantText string
		want     string
	}{
		"stable ID merges both fragments": {"echo-1", "help", "lineage-query/frame/1,lineage-query/frame/2"},
		"rotated ID merges nothing":       {"echo-rotated", "", "lineage-query/frame/2"},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			second := `p"}`
			if tc.wantText == "" {
				second = `{"text":""}`
			}
			model := &steeringRelationModel{steps: []func() (*llm.Completion, error){
				lineageCall("echo-1", "echo", `{"text":"hel`, "max_tokens"),
				lineageCall(tc.secondID, "echo", second, "tool_calls"),
				steeringFinal,
			}}
			var got []string
			var mu sync.Mutex
			echo := tools.Func[echoArgs]("echo", "fixture", func(_ context.Context, args echoArgs, _ *tools.Container) (any, error) {
				mu.Lock()
				got = append(got, args.Text)
				mu.Unlock()
				return "ok", nil
			})
			ag, err := New(Config{LLM: model, Tools: []tools.Tool{echo}, Warningf: func(string, ...any) {}, QueryIDGenerator: func() string { return "lineage-query" }})
			if err != nil {
				t.Fatal(err)
			}
			observed := observeContinuation(t, ag.QueryStreamEnveloped(context.Background(), llm.TextContent("start")))
			// Frame 2 continues a truncated response, not a tool block.
			wantContinuation(t, observed, map[string]string{
				"lineage-query/frame/1": "",
				"lineage-query/frame/2": "",
				"lineage-query/frame/3": tc.want,
			})
			// Independent oracle: the handler ran on the merged (or unmerged)
			// arguments.
			mu.Lock()
			defer mu.Unlock()
			if len(got) != 1 || got[0] != tc.wantText {
				t.Fatalf("handler args=%q, want %q", got, tc.wantText)
			}
		})
	}
}

// Steering accepted with a tool block, or interrupting the Frame that
// answers one, keeps both relations on the next Frame: the steering names
// the interrupted or extended Frame, the continuation the block's producer.
func TestToolContinuationAfterSteeringKeepsBothRelations(t *testing.T) {
	t.Run("steering in the tool block", func(t *testing.T) {
		model := &steeringRelationModel{steps: []func() (*llm.Completion, error){steeringWorkCall("work-1"), steeringFinal}}
		steering := make(chan SteeringMsg, 1)
		work := tools.Func[struct{}]("work", "fixture", func(context.Context, struct{}, *tools.Container) (any, error) {
			steering <- SteeringMsg{Content: "change course"}
			return "work result", nil
		})
		ag, err := New(Config{LLM: model, Tools: []tools.Tool{work}, Warningf: func(string, ...any) {}, QueryIDGenerator: func() string { return "lineage-query" }})
		if err != nil {
			t.Fatal(err)
		}
		observed := observeContinuation(t, ag.QueryStreamEnvelopedWithSteering(context.Background(), llm.TextContent("start"), steering))
		wantContinuation(t, observed, map[string]string{"lineage-query/frame/1": "", "lineage-query/frame/2": "lineage-query/frame/1"})
		if observed.steering["lineage-query/frame/2"] != "lineage-query/frame/1" || len(observed.steering) != 1 {
			t.Fatalf("steering=%v", observed.steering)
		}
	})
	t.Run("steering interrupts the answering Frame", func(t *testing.T) {
		model := &lineageStreamModel{started: make(chan struct{}), steps: []lineageStreamStep{
			lineageStreamToolCall("work-1"), lineageStreamBlockUntilCanceled, lineageStreamAnswer,
		}}
		steering := make(chan SteeringMsg, 1)
		work := tools.Func[struct{}]("work", "fixture", func(context.Context, struct{}, *tools.Container) (any, error) { return "work result", nil })
		ag, err := New(Config{LLM: model, Tools: []tools.Tool{work}, Warningf: func(string, ...any) {}, QueryIDGenerator: func() string { return "lineage-query" }})
		if err != nil {
			t.Fatal(err)
		}
		events := ag.QueryStreamEnvelopedWithSteering(context.Background(), llm.TextContent("start"), steering)
		go func() {
			<-model.started
			steering <- SteeringMsg{Content: "interrupting steer"}
		}()
		observed := observeContinuation(t, events)
		wantContinuation(t, observed, map[string]string{"lineage-query/frame/3": "lineage-query/frame/1"})
		if observed.steering["lineage-query/frame/3"] != "lineage-query/frame/2" || len(observed.steering) != 1 {
			t.Fatalf("steering=%v", observed.steering)
		}
		// Independent oracle: the third request still answers call 1.
		if requests := model.recorded(); len(requests) != 3 || !requestAnswersCall(requests[2], "work-1") {
			t.Fatalf("requests=%d", len(requests))
		}
	})
}

// A stream-idle recovery of the Frame answering a tool block keeps both
// relations on the recovery Frame.
func TestToolContinuationAfterStreamIdleRecoveryKeepsBothRelations(t *testing.T) {
	origTimeout, origRecoveries := agentStreamIdleTimeout, agentStreamIdleMaxRecoveries
	agentStreamIdleTimeout, agentStreamIdleMaxRecoveries = 20*time.Millisecond, 2
	t.Cleanup(func() { agentStreamIdleTimeout, agentStreamIdleMaxRecoveries = origTimeout, origRecoveries })
	model := &lineageStreamModel{started: make(chan struct{}), steps: []lineageStreamStep{
		lineageStreamToolCall("work-1"), lineageStreamBlockUntilCanceled, lineageStreamAnswer,
	}}
	work := tools.Func[struct{}]("work", "fixture", func(context.Context, struct{}, *tools.Container) (any, error) { return "work result", nil })
	ag, err := New(Config{LLM: model, Tools: []tools.Tool{work}, StreamIdleMaxRecoveries: -1, Warningf: func(string, ...any) {}, QueryIDGenerator: func() string { return "lineage-query" }})
	if err != nil {
		t.Fatal(err)
	}
	observed := observeContinuation(t, ag.QueryStreamEnveloped(context.Background(), llm.TextContent("start")))
	wantContinuation(t, observed, map[string]string{"lineage-query/frame/1": "", "lineage-query/frame/3": "lineage-query/frame/1"})
	if observed.recovery["lineage-query/frame/3"] != "lineage-query/frame/2" || len(observed.recovery) != 1 {
		t.Fatalf("recovery=%v", observed.recovery)
	}
	if requests := model.recorded(); len(requests) != 3 || !requestAnswersCall(requests[2], "work-1") {
		t.Fatalf("requests=%d", len(requests))
	}
}

// Without producer evidence the relation stays empty: no tool block, a
// response that only continues truncated text, a compaction that rewrote
// history after the block, and a released ephemeral result.
func TestToolContinuationWithoutEvidenceStaysEmpty(t *testing.T) {
	t.Run("no tool block", func(t *testing.T) {
		model := &steeringRelationModel{steps: []func() (*llm.Completion, error){
			func() (*llm.Completion, error) {
				return &llm.Completion{StopReason: "max_tokens", Content: llm.TextContent("part")}, nil
			},
			steeringFinal,
		}}
		ag, err := New(Config{LLM: model, Warningf: func(string, ...any) {}, QueryIDGenerator: func() string { return "lineage-query" }})
		if err != nil {
			t.Fatal(err)
		}
		observed := observeContinuation(t, ag.QueryStreamEnveloped(context.Background(), llm.TextContent("start")))
		wantContinuation(t, observed, map[string]string{"lineage-query/frame/1": "", "lineage-query/frame/2": ""})
	})
	t.Run("response to the block accepted", func(t *testing.T) {
		model := &steeringRelationModel{steps: []func() (*llm.Completion, error){
			steeringWorkCall("work-1"),
			func() (*llm.Completion, error) {
				return &llm.Completion{StopReason: "max_tokens", Content: llm.TextContent("part")}, nil
			},
			steeringFinal,
		}}
		work := tools.Func[struct{}]("work", "fixture", func(context.Context, struct{}, *tools.Container) (any, error) { return "work result", nil })
		ag, err := New(Config{LLM: model, Tools: []tools.Tool{work}, Warningf: func(string, ...any) {}, QueryIDGenerator: func() string { return "lineage-query" }})
		if err != nil {
			t.Fatal(err)
		}
		observed := observeContinuation(t, ag.QueryStreamEnveloped(context.Background(), llm.TextContent("start")))
		// Frame 3 still carries the results, but Frame 2 already answered
		// them: it continues Frame 2's text, not the block.
		wantContinuation(t, observed, map[string]string{"lineage-query/frame/2": "lineage-query/frame/1", "lineage-query/frame/3": ""})
		if requests := model.recorded(); len(requests) != 3 || !requestAnswersCall(requests[2], "work-1") {
			t.Fatalf("requests=%d", len(requests))
		}
	})
	t.Run("compaction after the block", func(t *testing.T) {
		model := &overflowScriptModel{script: []func() (*llm.Completion, error){
			steeringWorkCall("work-1"),
			func() (*llm.Completion, error) { return nil, typedOverflow() },
		}}
		work := tools.Func[struct{}]("work", "fixture", func(context.Context, struct{}, *tools.Container) (any, error) { return "work result", nil })
		ag := overflowAgent(t, model, func(c *Config) {
			c.Tools = []tools.Tool{work}
			c.QueryIDGenerator = func() string { return "lineage-query" }
		})
		observed := observeContinuation(t, ag.QueryStreamEnveloped(context.Background(), llm.TextContent("start")))
		// Frame 2 carried the results and was rejected; the compacted
		// history of Frame 3 is not described.
		wantContinuation(t, observed, map[string]string{"lineage-query/frame/2": "lineage-query/frame/1", "lineage-query/frame/3": ""})
		if main, summaries := model.snapshot(); len(main) != 3 || summaries != 1 {
			t.Fatalf("main=%d summaries=%d", len(main), summaries)
		}
	})
	t.Run("released ephemeral result", func(t *testing.T) {
		model := &steeringRelationModel{steps: []func() (*llm.Completion, error){
			func() (*llm.Completion, error) {
				return &llm.Completion{StopReason: "tool_calls", ToolCalls: []llm.ToolCall{
					{ID: "read-1", Type: "function", Function: llm.FunctionCall{Name: "read", Arguments: `{}`}},
					{ID: "read-2", Type: "function", Function: llm.FunctionCall{Name: "read", Arguments: `{}`}},
				}}, nil
			},
			steeringFinal,
		}}
		read := tools.Func[struct{}]("read", "fixture", func(context.Context, struct{}, *tools.Container) (any, error) { return "contents", nil }).WithEphemeralKeep(1)
		ag, err := New(Config{LLM: model, Tools: []tools.Tool{read}, Warningf: func(string, ...any) {}, QueryIDGenerator: func() string { return "lineage-query" }})
		if err != nil {
			t.Fatal(err)
		}
		observed := observeContinuation(t, ag.QueryStreamEnveloped(context.Background(), llm.TextContent("start")))
		wantContinuation(t, observed, map[string]string{"lineage-query/frame/2": ""})
		// Independent oracle: the second request carried a released result.
		released := false
		for _, message := range model.recorded()[1].Messages {
			released = released || message.Destroyed
		}
		if !released {
			t.Fatal("no result was released")
		}
	})
}

// lineageStreamModel streams scripted steps and records each request.
type lineageStreamStep func(ctx context.Context, ch chan<- llm.StreamEvent, started chan struct{})

type lineageStreamModel struct {
	calls    atomic.Int32
	started  chan struct{}
	mu       sync.Mutex
	steps    []lineageStreamStep
	requests []llm.InvokeRequest
}

func (*lineageStreamModel) Provider() string { return "fixture" }
func (*lineageStreamModel) Model() string    { return "lineage-stream" }
func (*lineageStreamModel) Invoke(context.Context, llm.InvokeRequest) (*llm.Completion, error) {
	panic("invoke should not be called")
}
func (m *lineageStreamModel) InvokeStream(ctx context.Context, req llm.InvokeRequest) (<-chan llm.StreamEvent, error) {
	owned, err := llm.CloneInvokeRequest(req)
	if err != nil {
		return nil, err
	}
	m.mu.Lock()
	m.requests = append(m.requests, owned)
	step := m.steps[min(len(m.requests), len(m.steps))-1]
	m.mu.Unlock()
	m.calls.Add(1)
	ch := make(chan llm.StreamEvent, 3)
	go func() {
		defer close(ch)
		step(ctx, ch, m.started)
	}()
	return ch, nil
}

func (m *lineageStreamModel) recorded() []llm.InvokeRequest {
	m.mu.Lock()
	defer m.mu.Unlock()
	return append([]llm.InvokeRequest(nil), m.requests...)
}

func lineageStreamToolCall(id string) lineageStreamStep {
	return func(_ context.Context, ch chan<- llm.StreamEvent, _ chan struct{}) {
		ch <- llm.StreamToolCallDeltaEvent{Index: 0, ID: id, NameDelta: "work", ArgumentsDelta: `{}`}
		ch <- llm.StreamDoneEvent{StopReason: "tool_calls"}
	}
}

func lineageStreamBlockUntilCanceled(ctx context.Context, _ chan<- llm.StreamEvent, started chan struct{}) {
	close(started)
	<-ctx.Done()
}

func lineageStreamAnswer(_ context.Context, ch chan<- llm.StreamEvent, _ chan struct{}) {
	ch <- llm.StreamTextDeltaEvent{Delta: "answer"}
	ch <- llm.StreamDoneEvent{StopReason: "stop"}
}

// A set beyond the bound is left unreported, never truncated; within it,
// the IDs keep the given Frame order.
func TestContinuationFrameIDsBound(t *testing.T) {
	within := make([]int, MaxRequestContinuationSources)
	for i := range within {
		within[i] = i + 1
	}
	ids := continuationFrameIDs("q", within)
	if len(ids) != MaxRequestContinuationSources || ids[0] != "q/frame/1" || ids[len(ids)-1] != executionFrameID("q", MaxRequestContinuationSources) {
		t.Fatalf("ids=%v", ids)
	}
	if ids := continuationFrameIDs("q", append(within, MaxRequestContinuationSources+1)); ids != nil {
		t.Fatalf("oversized set reported %d sources", len(ids))
	}
	if got := unionFrameSources([][]int{{1, 3}, {2, 3}, {3}}); len(got) != 3 || got[0] != 1 || got[1] != 2 || got[2] != 3 {
		t.Fatalf("union=%v", got)
	}
}
