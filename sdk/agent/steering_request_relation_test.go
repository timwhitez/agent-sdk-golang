package agent

import (
	"context"
	"errors"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

// steeringRelationModel returns scripted completions/errors and records a
// clone of every request it receives.
type steeringRelationModel struct {
	mu       sync.Mutex
	steps    []func() (*llm.Completion, error)
	requests []llm.InvokeRequest
}

func (*steeringRelationModel) Provider() string { return "fixture" }
func (*steeringRelationModel) Model() string    { return "steering-relation" }
func (m *steeringRelationModel) Invoke(_ context.Context, req llm.InvokeRequest) (*llm.Completion, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	owned, err := llm.CloneInvokeRequest(req)
	if err != nil {
		return nil, err
	}
	m.requests = append(m.requests, owned)
	step := m.steps[0]
	if len(m.steps) > 1 {
		m.steps = m.steps[1:]
	}
	return step()
}

func (m *steeringRelationModel) recorded() []llm.InvokeRequest {
	m.mu.Lock()
	defer m.mu.Unlock()
	return append([]llm.InvokeRequest(nil), m.requests...)
}

func steeringWorkCall(id string) func() (*llm.Completion, error) {
	return func() (*llm.Completion, error) {
		return &llm.Completion{StopReason: "tool_calls", Usage: &llm.Usage{PromptTokens: 10, CompletionTokens: 1, TotalTokens: 11},
			ToolCalls: []llm.ToolCall{{ID: id, Type: "function", Function: llm.FunctionCall{Name: "work", Arguments: `{}`}}}}, nil
	}
}

func steeringFinal() (*llm.Completion, error) {
	return &llm.Completion{StopReason: "stop", Content: llm.TextContent("finished"), Usage: &llm.Usage{PromptTokens: 10, CompletionTokens: 1, TotalTokens: 11}}, nil
}

func requestHasUserText(req llm.InvokeRequest, text string) bool {
	for _, message := range req.Messages {
		if message.Role == llm.RoleUser && message.Content.PlainText() == text {
			return true
		}
	}
	return false
}

// steeringObservation keeps each Frame's observed steering relations and the
// invoke attempts seen for it.
type steeringObservation struct {
	sources   map[string]map[string]bool // frame → steering sources
	attempts  map[string]map[uint64]bool // frame → attempts
	recovery  map[string]string          // frame → recovery source
	steerings int
}

func observeSteeringRelations(t *testing.T, events <-chan EventEnvelope) steeringObservation {
	t.Helper()
	observed := steeringObservation{sources: map[string]map[string]bool{}, attempts: map[string]map[uint64]bool{}, recovery: map[string]string{}}
	for env := range events {
		if _, ok := env.Event.(SteeringReceivedEvent); ok {
			observed.steerings++
			if env.FrameID != "" || env.RequestSteeringRelation != "" || env.RequestSteeringSourceFrameID != "" {
				t.Errorf("steering acknowledgement carries Frame metadata: %+v", env)
			}
		}
		if e, ok := env.Event.(ErrorEvent); ok {
			t.Errorf("unexpected error event: %+v", e)
		}
		if (env.RequestSteeringRelation != "") != (env.RequestSteeringSourceFrameID != "") {
			t.Errorf("half relation: %+v", env)
		}
		if env.RequestRecoveryRelation != "" {
			observed.recovery[env.FrameID] = env.RequestRecoverySourceFrameID
		}
		if env.FrameID == "" {
			continue
		}
		if observed.attempts[env.FrameID] == nil {
			observed.attempts[env.FrameID] = map[uint64]bool{}
			observed.sources[env.FrameID] = map[string]bool{}
		}
		observed.attempts[env.FrameID][env.InvokeAttempt] = true
		if env.RequestSteeringRelation != "" {
			if env.RequestSteeringRelation != RequestSteeringAccepted {
				t.Errorf("relation=%q", env.RequestSteeringRelation)
			}
			observed.sources[env.FrameID][env.RequestSteeringSourceFrameID] = true
		}
	}
	return observed
}

// wantSteeringSources asserts that exactly the listed Frames carry a steering
// relation and that each carries only its listed source.
func wantSteeringSources(t *testing.T, observed steeringObservation, want map[string]string) {
	t.Helper()
	for frame, sources := range observed.sources {
		source, related := want[frame]
		if !related && len(sources) != 0 {
			t.Errorf("frame %s carries steering sources %v, want none", frame, sources)
		}
		if related && (len(sources) != 1 || !sources[source]) {
			t.Errorf("frame %s carries steering sources %v, want only %s", frame, sources, source)
		}
	}
	for frame := range want {
		if _, seen := observed.sources[frame]; !seen {
			t.Errorf("frame %s emitted no events", frame)
		}
	}
}

// #83: user steering accepted in the middle of a tool block enters history
// when the block closes and changes the next logical request. That Frame —
// including its SDK retry — names the Frame whose block the steering
// extended; the Frame after it and a new Query carry no relation.
func TestSteeringAcceptedInToolBlockRecordsRequestRelation(t *testing.T) {
	model := &steeringRelationModel{steps: []func() (*llm.Completion, error){steeringWorkCall("work-1"), relationTransient, steeringWorkCall("work-2"), steeringFinal}}
	steering := make(chan SteeringMsg, 1)
	var workCalls atomic.Int32
	work := tools.Func[struct{}]("work", "fixture", func(context.Context, struct{}, *tools.Container) (any, error) {
		if workCalls.Add(1) == 1 {
			// Received at the block boundary right after this call.
			steering <- SteeringMsg{Content: "change course"}
		}
		return "work result", nil
	})
	ag, err := New(Config{LLM: model, Tools: []tools.Tool{work}, InvokeRetryMaxAttempts: 2, InvokeRetryBackoff: time.Nanosecond,
		Warningf: func(string, ...any) {}, QueryIDGenerator: func() string { return "steer-query" }})
	if err != nil {
		t.Fatal(err)
	}
	observed := observeSteeringRelations(t, ag.QueryStreamEnvelopedWithSteering(context.Background(), llm.TextContent("start"), steering))
	if observed.steerings != 1 {
		t.Fatalf("steering acknowledgements=%d", observed.steerings)
	}
	wantSteeringSources(t, observed, map[string]string{"steer-query/frame/2": "steer-query/frame/1"})
	if !observed.attempts["steer-query/frame/2"][2] {
		t.Fatalf("retried Frame attempt not observed: %v", observed.attempts)
	}
	if _, ok := observed.sources["steer-query/frame/3"]; !ok {
		t.Fatalf("following Frame not observed: %v", observed.attempts)
	}

	// Independent oracle: the steering is absent from Frame 1's request and
	// present in both attempts of Frame 2.
	requests := model.recorded()
	if len(requests) != 4 {
		t.Fatalf("requests=%d", len(requests))
	}
	for i, want := range []bool{false, true, true, true} {
		if got := requestHasUserText(requests[i], "change course"); got != want {
			t.Fatalf("request[%d] has steering=%v", i, got)
		}
	}

	// A new Query on the same Agent starts clean.
	model.mu.Lock()
	model.steps = []func() (*llm.Completion, error){steeringFinal}
	model.mu.Unlock()
	fresh := observeSteeringRelations(t, ag.QueryStreamEnvelopedWithSteering(context.Background(), llm.TextContent("again"), steering))
	wantSteeringSources(t, fresh, nil)
}

// steeringInterruptStreamModel streams part of its first response and then
// blocks until the stage is canceled; later calls answer.
type steeringInterruptStreamModel struct {
	calls   atomic.Int32
	started chan struct{}
	mu      sync.Mutex
	seen    []bool // per request: contains the steering text
}

func (*steeringInterruptStreamModel) Provider() string { return "fixture" }
func (*steeringInterruptStreamModel) Model() string    { return "steering-interrupt" }
func (*steeringInterruptStreamModel) Invoke(context.Context, llm.InvokeRequest) (*llm.Completion, error) {
	return nil, errors.New("invoke should not be called")
}
func (m *steeringInterruptStreamModel) InvokeStream(ctx context.Context, req llm.InvokeRequest) (<-chan llm.StreamEvent, error) {
	m.mu.Lock()
	m.seen = append(m.seen, requestHasUserText(req, "interrupting steer"))
	m.mu.Unlock()
	ch := make(chan llm.StreamEvent, 2)
	n := m.calls.Add(1)
	go func() {
		defer close(ch)
		if n == 1 {
			ch <- llm.StreamTextDeltaEvent{Delta: "partial "}
			close(m.started)
			<-ctx.Done()
			return
		}
		ch <- llm.StreamTextDeltaEvent{Delta: "answer"}
		ch <- llm.StreamDoneEvent{StopReason: "stop"}
	}()
	return ch, nil
}

// Steering that interrupts a streaming Frame is recorded on the next Frame
// with the interrupted Frame as the source.
func TestSteeringInterruptRecordsInterruptedFrame(t *testing.T) {
	model := &steeringInterruptStreamModel{started: make(chan struct{})}
	ag, err := New(Config{LLM: model, Warningf: func(string, ...any) {}, QueryIDGenerator: func() string { return "interrupt-query" }})
	if err != nil {
		t.Fatal(err)
	}
	steering := make(chan SteeringMsg, 1)
	events := ag.QueryStreamEnvelopedWithSteering(context.Background(), llm.TextContent("start"), steering)
	go func() {
		<-model.started
		steering <- SteeringMsg{Content: "interrupting steer"}
	}()
	observed := observeSteeringRelations(t, events)
	if observed.steerings != 1 || model.calls.Load() != 2 {
		t.Fatalf("steerings=%d calls=%d", observed.steerings, model.calls.Load())
	}
	wantSteeringSources(t, observed, map[string]string{"interrupt-query/frame/2": "interrupt-query/frame/1"})
	if model.seen[0] || !model.seen[1] {
		t.Fatalf("steering in requests=%v", model.seen)
	}
}

// Steering that never enters history creates no relation and no extra model
// request: an empty message is dropped, a stage canceled for steering with no
// queued message applies nothing, and steering accepted before the Query's
// first Frame has no producer Frame.
func TestSteeringWithoutAcceptedHistoryChangeHasNoRelation(t *testing.T) {
	t.Run("empty message in tool block", func(t *testing.T) {
		model := &steeringRelationModel{steps: []func() (*llm.Completion, error){steeringWorkCall("work-1"), steeringFinal}}
		steering := make(chan SteeringMsg, 1)
		work := tools.Func[struct{}]("work", "fixture", func(context.Context, struct{}, *tools.Container) (any, error) {
			steering <- SteeringMsg{Content: "   "}
			return "work result", nil
		})
		ag, err := New(Config{LLM: model, Tools: []tools.Tool{work}, Warningf: func(string, ...any) {}})
		if err != nil {
			t.Fatal(err)
		}
		observed := observeSteeringRelations(t, ag.QueryStreamEnvelopedWithSteering(context.Background(), llm.TextContent("start"), steering))
		wantSteeringSources(t, observed, nil)
		if observed.steerings != 0 || len(model.recorded()) != 2 {
			t.Fatalf("steerings=%d requests=%d", observed.steerings, len(model.recorded()))
		}
	})
	t.Run("tool stage canceled without message", func(t *testing.T) {
		model := &steeringRelationModel{steps: []func() (*llm.Completion, error){steeringWorkCall("work-1"), steeringFinal}}
		steering := make(chan SteeringMsg, 1)
		var ag *Agent
		work := tools.Func[struct{}]("work", "fixture", func(ctx context.Context, _ struct{}, _ *tools.Container) (any, error) {
			if !ag.InterruptActiveStageForSteering() {
				t.Error("tool stage not interruptible")
			}
			<-ctx.Done()
			return nil, ctx.Err()
		})
		var err error
		ag, err = New(Config{LLM: model, Tools: []tools.Tool{work}, Warningf: func(string, ...any) {}})
		if err != nil {
			t.Fatal(err)
		}
		observed := observeSteeringRelations(t, ag.QueryStreamEnvelopedWithSteering(context.Background(), llm.TextContent("start"), steering))
		wantSteeringSources(t, observed, nil)
		if observed.steerings != 0 || len(model.recorded()) != 2 {
			t.Fatalf("steerings=%d requests=%d", observed.steerings, len(model.recorded()))
		}
	})
	t.Run("provider stage canceled after pre-query steering", func(t *testing.T) {
		model := &steeringAlreadyAppliedStreamModel{firstStageStarted: make(chan struct{})}
		ag, err := New(Config{LLM: model, Warningf: func(string, ...any) {}})
		if err != nil {
			t.Fatal(err)
		}
		steering := make(chan SteeringMsg, 1)
		steering <- SteeringMsg{Content: "already applied steering"}
		events := ag.QueryStreamEnvelopedWithSteering(context.Background(), llm.TextContent("start"), steering)
		go func() {
			<-model.firstStageStarted
			if !ag.InterruptActiveStageForSteering() {
				t.Error("provider stage not interruptible")
			}
		}()
		observed := observeSteeringRelations(t, events)
		wantSteeringSources(t, observed, nil)
		// The second request re-sends the interrupted stage; the steering
		// itself added none.
		if observed.steerings != 1 || model.calls.Load() != 2 {
			t.Fatalf("steerings=%d calls=%d", observed.steerings, model.calls.Load())
		}
	})
}

// Steering and stream-idle recovery keep separate relation fields: a Frame
// built after both carries both, and the steering relation of a later Frame
// does not carry the consumed recovery relation.
func TestSteeringRelationComposesWithRecoveryRelation(t *testing.T) {
	origTimeout, origRecoveries := agentStreamIdleTimeout, agentStreamIdleMaxRecoveries
	agentStreamIdleTimeout, agentStreamIdleMaxRecoveries = 20*time.Millisecond, 2
	t.Cleanup(func() { agentStreamIdleTimeout, agentStreamIdleMaxRecoveries = origTimeout, origRecoveries })
	steering := make(chan SteeringMsg, 2)
	var noopCalls atomic.Int32
	noop := tools.Func[struct{}]("noop", "noop", func(context.Context, struct{}, *tools.Container) (any, error) {
		if noopCalls.Add(1) == 1 {
			steering <- SteeringMsg{Content: "steer during tool"}
		}
		return "ok", nil
	})
	var recovered atomic.Bool
	ag, err := New(Config{LLM: &stallThenToolModel{}, Tools: []tools.Tool{noop}, StreamIdleMaxRecoveries: -1,
		QueryIDGenerator: func() string { return "compose-query" },
		// Called synchronously after the recovery reminder entered history;
		// the steering is then drained at the next loop boundary.
		Warningf: func(format string, _ ...any) {
			if strings.Contains(format, "auto-recovering") && recovered.CompareAndSwap(false, true) {
				steering <- SteeringMsg{Content: "steer after stall"}
			}
		}})
	if err != nil {
		t.Fatal(err)
	}
	observed := observeSteeringRelations(t, ag.QueryStreamEnvelopedWithSteering(context.Background(), llm.TextContent("start"), steering))
	if observed.steerings != 2 {
		t.Fatalf("steerings=%d", observed.steerings)
	}
	wantSteeringSources(t, observed, map[string]string{
		"compose-query/frame/2": "compose-query/frame/1",
		"compose-query/frame/3": "compose-query/frame/2",
	})
	if len(observed.recovery) != 1 || observed.recovery["compose-query/frame/2"] != "compose-query/frame/1" {
		t.Fatalf("recovery relations=%v", observed.recovery)
	}
}

// A max-token continuation merges tool-call arguments from two requests and
// executes them under the finalizing Frame. No relation names the earlier
// fragment's Frame as a request source, and steering accepted in that block
// names the finalizing Frame as the execution it extended, not as the sole
// content source.
func TestContinuationClaimsNoSingleContentSource(t *testing.T) {
	model := &steeringRelationModel{steps: []func() (*llm.Completion, error){
		func() (*llm.Completion, error) {
			return &llm.Completion{StopReason: "max_tokens", ToolCalls: []llm.ToolCall{{ID: "work-1", Function: llm.FunctionCall{Name: "work", Arguments: `{"text":`}}}}, nil
		},
		func() (*llm.Completion, error) {
			return &llm.Completion{StopReason: "tool_calls", ToolCalls: []llm.ToolCall{{ID: "work-1", Function: llm.FunctionCall{Name: "work", Arguments: `"ok"}`}}}}, nil
		},
		steeringFinal,
	}}
	steering := make(chan SteeringMsg, 1)
	work := tools.Func[struct {
		Text string `json:"text"`
	}]("work", "fixture", func(_ context.Context, args struct {
		Text string `json:"text"`
	}, _ *tools.Container) (any, error) {
		if args.Text != "ok" {
			t.Errorf("merged args=%q", args.Text)
		}
		steering <- SteeringMsg{Content: "after merged call"}
		return "work result", nil
	})
	ag, err := New(Config{LLM: model, Tools: []tools.Tool{work}, Warningf: func(string, ...any) {}, QueryIDGenerator: func() string { return "cont-query" }})
	if err != nil {
		t.Fatal(err)
	}
	toolFrames := map[string]bool{}
	var otherRelations []EventEnvelope
	var all []EventEnvelope
	for env := range ag.QueryStreamEnvelopedWithSteering(context.Background(), llm.TextContent("start"), steering) {
		all = append(all, env)
	}
	replay := make(chan EventEnvelope, len(all))
	for _, env := range all {
		replay <- env
	}
	close(replay)
	observed := observeSteeringRelations(t, replay)
	for _, env := range all {
		if _, ok := env.Event.(ToolCallEvent); ok {
			toolFrames[env.FrameID] = true
		}
		if env.RequestControlRelation != "" || env.RequestHistoryRelation != "" || env.RequestRecoveryRelation != "" {
			otherRelations = append(otherRelations, env)
		}
		if env.RequestSteeringSourceFrameID == "cont-query/frame/1" {
			t.Errorf("fragment Frame named as a request source: %+v", env)
		}
	}
	if len(toolFrames) != 1 || !toolFrames["cont-query/frame/2"] {
		t.Fatalf("tool call frames=%v, want only the finalizing Frame", toolFrames)
	}
	if len(otherRelations) != 0 {
		t.Fatalf("continuation reported relations: %+v", otherRelations)
	}
	wantSteeringSources(t, observed, map[string]string{"cont-query/frame/3": "cont-query/frame/2"})
}
