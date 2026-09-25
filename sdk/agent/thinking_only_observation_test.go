package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

// thinkingOnlyCanary is hidden reasoning text; it must never reach a
// thinking-only observation.
const thinkingOnlyCanary = "CANARY_HIDDEN_REASONING_7f3a"

const thinkingOnlyOpaqueState = `{"id":"rs_1","type":"reasoning","encrypted_content":"CANARY_OPAQUE_CIPHERTEXT"}`

// thinkingStep scripts one provider response.
type thinkingStep struct {
	thinking  string // Thinking text (streamed as a legacy thinking delta)
	block     string // "", "thinking" or "redacted_thinking": a structured block
	text      string
	toolCall  bool
	state     bool // attach opaque provider state
	stop      string
	noDone    bool // stream closes without StreamDoneEvent
	streamErr bool // stream ends with a StreamErrorEvent / buffered error
	stall     bool // stream stops sending until the attempt is canceled
	cancel    bool // the provider cancels the Query before returning
}

type thinkingScript struct {
	mu       sync.Mutex
	steps    []thinkingStep
	calls    int
	requests [][]byte
	cancel   context.CancelFunc
}

func (s *thinkingScript) next(req llm.InvokeRequest) (thinkingStep, int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	encoded, err := json.Marshal(req)
	if err != nil {
		return thinkingStep{}, 0, err
	}
	s.requests = append(s.requests, encoded)
	s.calls++
	if s.calls > len(s.steps) {
		return thinkingStep{}, s.calls, fmt.Errorf("unexpected call %d", s.calls)
	}
	return s.steps[s.calls-1], s.calls, nil
}

func (s *thinkingScript) snapshot() (int, [][]byte) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.calls, append([][]byte(nil), s.requests...)
}

func thinkingOnlyProviderState() []llm.ProviderState {
	return []llm.ProviderState{{Provider: "test-responses", Kind: "response.output_item.v1", Data: json.RawMessage(thinkingOnlyOpaqueState)}}
}

// thinkingBufferedModel implements only Invoke.
type thinkingBufferedModel struct{ thinkingScript }

func (*thinkingBufferedModel) Provider() string { return "fixture" }
func (*thinkingBufferedModel) Model() string    { return "thinking-buffered" }
func (m *thinkingBufferedModel) Invoke(_ context.Context, req llm.InvokeRequest) (*llm.Completion, error) {
	step, call, err := m.next(req)
	if err != nil {
		return nil, err
	}
	content := llm.Content{Text: step.text}
	switch step.block {
	case "thinking":
		content = llm.Content{Blocks: []llm.ContentBlock{{Type: "thinking", Thinking: step.thinking, Signature: "sig"}}}
		if step.text != "" {
			content.Blocks = append(content.Blocks, llm.ContentBlock{Type: "text", Text: step.text})
		}
	case "redacted_thinking":
		content = llm.Content{Blocks: []llm.ContentBlock{{Type: "redacted_thinking", Data: "REDACTED_BLOB"}}}
	}
	if step.state {
		if content, err = llm.WithProviderState(content, thinkingOnlyProviderState()); err != nil {
			return nil, err
		}
	}
	comp := &llm.Completion{Content: content, Thinking: step.thinking, StopReason: step.stop, Usage: &llm.Usage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15}}
	if step.toolCall {
		comp.ToolCalls = []llm.ToolCall{{ID: "probe-" + strconv.Itoa(call), Type: "function", Function: llm.FunctionCall{Name: "probe", Arguments: `{}`}}}
	}
	if step.cancel {
		m.cancel()
	}
	if step.streamErr {
		return comp, &llm.ProviderError{Provider: "fixture", StatusCode: 400, Message: "bad request"}
	}
	return comp, nil
}

// thinkingStreamModel implements InvokeStream; Invoke is not expected.
type thinkingStreamModel struct{ thinkingScript }

func (*thinkingStreamModel) Provider() string { return "fixture" }
func (*thinkingStreamModel) Model() string    { return "thinking-stream" }
func (*thinkingStreamModel) Invoke(context.Context, llm.InvokeRequest) (*llm.Completion, error) {
	return nil, errors.New("buffered invoke is not expected")
}
func (m *thinkingStreamModel) InvokeStream(ctx context.Context, req llm.InvokeRequest) (<-chan llm.StreamEvent, error) {
	step, call, err := m.next(req)
	if err != nil {
		return nil, err
	}
	ch := make(chan llm.StreamEvent, 8)
	go func() {
		defer close(ch)
		switch step.block {
		case "thinking":
			ch <- llm.StreamThinkingDeltaEvent{Delta: step.thinking, BlockType: "thinking", SignatureDelta: "sig"}
		case "redacted_thinking":
			ch <- llm.StreamThinkingDeltaEvent{BlockType: "redacted_thinking", Data: "REDACTED_BLOB"}
		default:
			if step.thinking != "" {
				ch <- llm.StreamThinkingDeltaEvent{Delta: step.thinking}
			}
		}
		if step.text != "" {
			ch <- llm.StreamTextDeltaEvent{Delta: step.text}
		}
		if step.toolCall {
			ch <- llm.StreamToolCallDeltaEvent{ID: "probe-" + strconv.Itoa(call), NameDelta: "probe", ArgumentsDelta: `{}`}
		}
		if step.state {
			ch <- llm.StreamProviderStateEvent{State: thinkingOnlyProviderState()}
		}
		ch <- llm.StreamUsageEvent{Usage: llm.Usage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15}}
		switch {
		case step.stall:
			<-ctx.Done()
			return
		case step.streamErr:
			ch <- llm.StreamErrorEvent{Provider: "fixture", StatusCode: 400, Message: "bad request"}
			return
		case step.noDone:
			return
		}
		ch <- llm.StreamDoneEvent{StopReason: step.stop}
		if step.cancel {
			// Cancel before the channel closes: the agent observes the
			// cancellation no later than the completed stream.
			m.cancel()
		}
	}()
	return ch, nil
}

type thinkingModel interface {
	llm.ChatModel
	script() *thinkingScript
}

func (m *thinkingBufferedModel) script() *thinkingScript { return &m.thinkingScript }
func (m *thinkingStreamModel) script() *thinkingScript   { return &m.thinkingScript }

func newThinkingModels(steps ...thinkingStep) map[string]func() thinkingModel {
	return map[string]func() thinkingModel{
		"buffered":  func() thinkingModel { return &thinkingBufferedModel{thinkingScript{steps: steps}} },
		"streaming": func() thinkingModel { return &thinkingStreamModel{thinkingScript{steps: steps}} },
	}
}

// runThinkingQueries runs each prompt as one Query on one Agent and returns
// every received envelope.
func runThinkingQueries(t *testing.T, model thinkingModel, observe bool, mutate func(*Config), prompts ...string) (*Agent, []EventEnvelope) {
	t.Helper()
	probe := tools.Func[struct{}]("probe", "probe", func(context.Context, struct{}, *tools.Container) (any, error) { return "probe-ok", nil })
	cfg := Config{LLM: model, Tools: []tools.Tool{probe}, ObserveThinkingOnlyResponses: observe, StreamIdleTimeout: 20 * time.Millisecond, StreamIdleMaxRecoveries: 0}
	if mutate != nil {
		mutate(&cfg)
	}
	ag, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	var envs []EventEnvelope
	for _, prompt := range prompts {
		ctx, cancel := context.WithCancel(context.Background())
		model.script().mu.Lock()
		model.script().cancel = cancel
		model.script().mu.Unlock()
		for env := range ag.QueryStreamEnveloped(ctx, llm.TextContent(prompt)) {
			envs = append(envs, env)
		}
		cancel()
	}
	return ag, envs
}

func thinkingOnlyEnvelopes(envs []EventEnvelope) []EventEnvelope {
	var out []EventEnvelope
	for _, env := range envs {
		w, isWarn := env.Event.(WarnEvent)
		if env.Intervention == InterventionThinkingOnly || env.InterventionResult == InterventionResultObservedOnly ||
			env.InterventionStage == InterventionStageDetected || isWarn && w.Kind == thinkingOnlyObservedKind {
			out = append(out, env)
		}
	}
	return out
}

// assertThinkingOnlyObservation checks one observe-only report: fixed labels,
// detected (never applied), no strike, Frame correlation, fixed content.
func assertThinkingOnlyObservation(t *testing.T, env EventEnvelope, frame int) {
	t.Helper()
	w, ok := env.Event.(WarnEvent)
	if !ok || w.Kind != thinkingOnlyObservedKind || w.Message != thinkingOnlyObservedMessage || w.Metadata != nil {
		t.Fatalf("event=%#v", env.Event)
	}
	if env.Intervention != InterventionThinkingOnly || env.InterventionStage != InterventionStageDetected ||
		env.InterventionStage == InterventionStageApplied || env.InterventionResult != InterventionResultObservedOnly || env.InterventionStrike != 0 {
		t.Fatalf("labels=%q/%q/%q/%d", env.Intervention, env.InterventionStage, env.InterventionResult, env.InterventionStrike)
	}
	if env.Kind != EventKindWarning || env.Origin != EventOriginSDKDriver || !strings.HasSuffix(env.FrameID, "/frame/"+strconv.Itoa(frame)) || env.InvokeAttempt != 1 {
		t.Fatalf("envelope=%+v", env)
	}
	encoded, err := json.Marshal(env)
	if err != nil {
		t.Fatal(err)
	}
	// The message is the fixed constant (no text or length); the whole
	// envelope carries no reasoning, opaque state or redacted data.
	for _, secret := range []string{thinkingOnlyCanary, "CANARY_OPAQUE_CIPHERTEXT", "REDACTED_BLOB", "PRIVATE_"} {
		if strings.Contains(string(encoded), secret) {
			t.Fatalf("observation leaked %q: %s", secret, encoded)
		}
	}
}

// A complete, normally terminated response with reasoning activity and no
// visible output or tool calls is observed once, in both invocation modes
// and for every accepted reasoning evidence.
func TestThinkingOnlyObservationDetectsCompleteResponse(t *testing.T) {
	cases := map[string]thinkingStep{
		"thinking text":              {thinking: thinkingOnlyCanary, stop: "end_turn"},
		"signed thinking block":      {thinking: thinkingOnlyCanary, block: "thinking", stop: "end_turn"},
		"redacted block only":        {block: "redacted_thinking", stop: "end_turn"},
		"thinking with opaque state": {thinking: thinkingOnlyCanary, state: true, stop: "stop"},
		"whitespace text":            {thinking: thinkingOnlyCanary, text: " \n", stop: "stop_sequence"},
	}
	for name, step := range cases {
		for mode, build := range newThinkingModels(step) {
			t.Run(name+"/"+mode, func(t *testing.T) {
				model := build()
				_, envs := runThinkingQueries(t, model, true, nil, "PRIVATE_PROMPT")
				labeled := thinkingOnlyEnvelopes(envs)
				if len(labeled) != 1 {
					t.Fatalf("observations=%d: %+v", len(labeled), labeled)
				}
				assertThinkingOnlyObservation(t, labeled[0], 1)
				if calls, _ := model.script().snapshot(); calls != 1 {
					t.Fatalf("model calls=%d", calls)
				}
				// Observation does not end or alter the turn: the ordinary
				// terminal follows it.
				last := envs[len(envs)-1]
				if final, ok := last.Event.(FinalResponseEvent); !ok || final.Status != "complete" || last.Sequence <= labeled[0].Sequence {
					t.Fatalf("terminal=%+v", last)
				}
			})
		}
	}
}

// Responses that are not complete, legal thinking-only responses are never
// observed, and the switch off reports nothing.
func TestThinkingOnlyObservationExclusions(t *testing.T) {
	thinking := thinkingStep{thinking: thinkingOnlyCanary, state: true, stop: "end_turn"}
	text := thinkingStep{text: "answer", stop: "end_turn"}
	with := func(step thinkingStep, change func(*thinkingStep)) thinkingStep {
		change(&step)
		return step
	}
	cases := []struct {
		name      string
		steps     []thinkingStep
		streaming bool // stream-only failure mode
		recover   int
		observe   bool
		wantErr   bool // the Query ends with an ErrorEvent
	}{
		{name: "switch off", steps: []thinkingStep{thinking}},
		{name: "visible text", steps: []thinkingStep{with(thinking, func(s *thinkingStep) { s.text = "visible" })}, observe: true},
		{name: "visible text beside thinking block", steps: []thinkingStep{with(thinking, func(s *thinkingStep) { s.block, s.text = "thinking", "visible" })}, observe: true},
		{name: "tool call", steps: []thinkingStep{with(thinking, func(s *thinkingStep) { s.toolCall, s.stop = true, "tool_use" }), text}, observe: true},
		{name: "tool call with normal stop", steps: []thinkingStep{with(thinking, func(s *thinkingStep) { s.toolCall = true }), text}, observe: true},
		{name: "max_tokens", steps: []thinkingStep{with(thinking, func(s *thinkingStep) { s.stop = "max_tokens" }), text}, observe: true},
		{name: "raw length", steps: []thinkingStep{with(thinking, func(s *thinkingStep) { s.stop = "length" })}, observe: true},
		{name: "unknown stop", steps: []thinkingStep{with(thinking, func(s *thinkingStep) { s.stop = "" })}, observe: true},
		{name: "content filter", steps: []thinkingStep{with(thinking, func(s *thinkingStep) { s.stop = "content_filter" })}, observe: true},
		{name: "continuation completion", steps: []thinkingStep{{text: "part", stop: "max_tokens"}, thinking}, observe: true},
		// Extended thinking can use up max_tokens before any text: the reply
		// to that continuation is still part of the truncated response.
		{name: "continuation of a thinking-only truncation", steps: []thinkingStep{with(thinking, func(s *thinkingStep) { s.stop = "max_tokens" }), thinking}, observe: true},
		{name: "continuation of a truncated tool call", steps: []thinkingStep{with(thinking, func(s *thinkingStep) { s.toolCall, s.stop = true, "max_tokens" }), thinking}, observe: true},
		{name: "opaque state without reasoning", steps: []thinkingStep{{state: true, stop: "end_turn"}}, observe: true},
		{name: "no reasoning", steps: []thinkingStep{{stop: "end_turn"}}, observe: true},
		{name: "cancellation", steps: []thinkingStep{with(thinking, func(s *thinkingStep) { s.cancel = true })}, observe: true, wantErr: true},
		{name: "provider error", steps: []thinkingStep{with(thinking, func(s *thinkingStep) { s.streamErr = true })}, observe: true, wantErr: true},
		{name: "incomplete stream", steps: []thinkingStep{with(thinking, func(s *thinkingStep) { s.noDone = true })}, streaming: true, observe: true, wantErr: true},
		{name: "stream idle stall", steps: []thinkingStep{with(thinking, func(s *thinkingStep) { s.stall = true })}, streaming: true, observe: true, wantErr: true},
		{name: "stream idle stall recovered", steps: []thinkingStep{with(thinking, func(s *thinkingStep) { s.stall = true }), text}, streaming: true, recover: 1, observe: true},
	}
	for _, tc := range cases {
		for mode, build := range newThinkingModels(tc.steps...) {
			if tc.streaming && mode != "streaming" {
				continue
			}
			t.Run(tc.name+"/"+mode, func(t *testing.T) {
				model := build()
				_, envs := runThinkingQueries(t, model, tc.observe, func(cfg *Config) { cfg.StreamIdleMaxRecoveries = tc.recover }, "PRIVATE_PROMPT")
				if labeled := thinkingOnlyEnvelopes(envs); len(labeled) != 0 {
					t.Fatalf("unexpected observation: %+v", labeled)
				}
				if calls, _ := model.script().snapshot(); calls != len(tc.steps) {
					t.Fatalf("model calls=%d want %d", calls, len(tc.steps))
				}
				_, isErr := envs[len(envs)-1].Event.(ErrorEvent)
				if isErr != tc.wantErr {
					t.Fatalf("terminal=%#v wantErr=%v", envs[len(envs)-1].Event, tc.wantErr)
				}
			})
		}
	}
}

// With the switch on and off, the same script sends byte-identical requests
// (including the replayed opaque state in the next Query), makes the same
// number of model calls, and leaves identical history; only the observation
// event differs.
func TestThinkingOnlyObservationDoesNotChangeRequestsOrHistory(t *testing.T) {
	steps := []thinkingStep{
		{thinking: thinkingOnlyCanary, block: "thinking", state: true, stop: "end_turn"},
		{text: "second answer", stop: "end_turn"},
	}
	for mode, build := range newThinkingModels(steps...) {
		t.Run(mode, func(t *testing.T) {
			run := func(observe bool) (int, [][]byte, []llm.Message, []EventEnvelope) {
				model := build()
				ag, envs := runThinkingQueries(t, model, observe, nil, "PRIVATE_FIRST", "PRIVATE_SECOND")
				calls, requests := model.script().snapshot()
				return calls, requests, ag.Messages(), envs
			}
			offCalls, offRequests, offHistory, offEnvs := run(false)
			onCalls, onRequests, onHistory, onEnvs := run(true)
			if offCalls != 2 || onCalls != 2 || len(offRequests) != 2 || len(onRequests) != 2 {
				t.Fatalf("calls off=%d on=%d", offCalls, onCalls)
			}
			for i := range offRequests {
				if string(offRequests[i]) != string(onRequests[i]) {
					t.Fatalf("request %d differs:\noff=%s\non=%s", i+1, offRequests[i], onRequests[i])
				}
			}
			if !strings.Contains(string(onRequests[1]), "CANARY_OPAQUE_CIPHERTEXT") {
				t.Fatalf("next request lost the opaque provider state: %s", onRequests[1])
			}
			if !reflect.DeepEqual(offHistory, onHistory) {
				t.Fatalf("history differs:\noff=%+v\non=%+v", offHistory, onHistory)
			}
			state, err := llm.ProviderStateFromContent(onHistory[1].Content)
			if err != nil || !reflect.DeepEqual(state, thinkingOnlyProviderState()) {
				t.Fatalf("stored provider state=%+v err=%v", state, err)
			}
			if labeled := thinkingOnlyEnvelopes(offEnvs); len(labeled) != 0 {
				t.Fatalf("switch off observed: %+v", labeled)
			}
			labeled := thinkingOnlyEnvelopes(onEnvs)
			if len(labeled) != 1 {
				t.Fatalf("observations=%d", len(labeled))
			}
			assertThinkingOnlyObservation(t, labeled[0], 1)
			// Apart from the one observation, both runs emitted the same
			// event kinds in the same order.
			kinds := func(envs []EventEnvelope, skip bool) []string {
				var out []string
				for _, env := range envs {
					if skip && env.Intervention == InterventionThinkingOnly {
						continue
					}
					out = append(out, fmt.Sprintf("%s/%T", env.Kind, env.Event))
				}
				return out
			}
			if !reflect.DeepEqual(kinds(offEnvs, false), kinds(onEnvs, true)) {
				t.Fatalf("event kinds differ:\noff=%v\non=%v", kinds(offEnvs, false), kinds(onEnvs, true))
			}
		})
	}
}

// Within one Query the observation does not interfere with the existing
// early-stop reminder: the thinking-only stop after a tool is still reminded
// and the next request is unchanged by the switch.
func TestThinkingOnlyObservationLeavesEarlyStopUnchanged(t *testing.T) {
	steps := []thinkingStep{
		{toolCall: true, stop: "tool_use"},
		{thinking: thinkingOnlyCanary, state: true, stop: "end_turn"},
		{text: "final", stop: "end_turn"},
	}
	done := tools.Func[struct{}]("done", "done", func(context.Context, struct{}, *tools.Container) (any, error) { return "done", nil })
	for mode, build := range newThinkingModels(steps...) {
		t.Run(mode, func(t *testing.T) {
			run := func(observe bool) ([][]byte, []EventEnvelope) {
				model := build()
				_, envs := runThinkingQueries(t, model, observe, func(cfg *Config) { cfg.Tools = append(cfg.Tools, done) }, "PRIVATE_PROMPT")
				calls, requests := model.script().snapshot()
				if calls != 3 {
					t.Fatalf("calls=%d", calls)
				}
				return requests, envs
			}
			offRequests, _ := run(false)
			onRequests, onEnvs := run(true)
			if !reflect.DeepEqual(offRequests, onRequests) {
				t.Fatal("requests differ with the switch on")
			}
			labeled := thinkingOnlyEnvelopes(onEnvs)
			if len(labeled) != 1 {
				t.Fatalf("observations=%d", len(labeled))
			}
			assertThinkingOnlyObservation(t, labeled[0], 2)
		})
	}
}

func TestCompletionIsThinkingOnlyEvidence(t *testing.T) {
	state, err := llm.WithProviderState(llm.Content{}, thinkingOnlyProviderState())
	if err != nil {
		t.Fatal(err)
	}
	for name, tc := range map[string]struct {
		comp *llm.Completion
		want bool
	}{
		"nil":            {nil, false},
		"thinking":       {&llm.Completion{Thinking: "x", StopReason: "END_TURN"}, true},
		"blank thinking": {&llm.Completion{Thinking: "  ", StopReason: "end_turn"}, false},
		"state only":     {&llm.Completion{Content: state, StopReason: "end_turn"}, false},
		"image":          {&llm.Completion{Thinking: "x", Content: llm.Content{Blocks: []llm.ContentBlock{{Type: "image_url", ImageURL: &llm.ImageURL{URL: "u"}}}}, StopReason: "end_turn"}, false},
		"unknown block":  {&llm.Completion{Thinking: "x", Content: llm.Content{Blocks: []llm.ContentBlock{{Type: "audio"}}}, StopReason: "end_turn"}, false},
		"pause_turn":     {&llm.Completion{Thinking: "x", StopReason: "pause_turn"}, false},
		"refusal":        {&llm.Completion{Thinking: "x", StopReason: "refusal"}, false},
	} {
		if got := completionIsThinkingOnly(tc.comp); got != tc.want {
			t.Errorf("%s: got %v want %v", name, got, tc.want)
		}
	}
}

// A text continuation that finished does not suppress later judgements: with
// RequireDone, thinking-only replies after "part"(max_tokens)+"rest" are
// observed (the continuation flag does not go stale on the reminder path).
func TestThinkingOnlyObservationAfterFinishedContinuation(t *testing.T) {
	thinking := thinkingStep{thinking: thinkingOnlyCanary, state: true, stop: "end_turn"}
	steps := []thinkingStep{{toolCall: true, stop: "tool_use"}, {text: "part", stop: "max_tokens"}, {text: "rest", stop: "end_turn"}, thinking, thinking}
	for mode, build := range newThinkingModels(steps...) {
		t.Run(mode, func(t *testing.T) {
			model := build()
			_, envs := runThinkingQueries(t, model, true, func(cfg *Config) { cfg.RequireDoneTool = true }, "PRIVATE_PROMPT")
			if labeled := thinkingOnlyEnvelopes(envs); len(labeled) == 0 {
				t.Fatal("thinking-only replies after a finished continuation were not observed")
			}
		})
	}
}
