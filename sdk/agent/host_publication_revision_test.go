package agent

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/agent/compaction"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

// publicationObservation is one Frame-correlated envelope's publication.
type publicationObservation struct {
	frame       string
	attempt     uint64
	publication uint64
}

// observePublications returns the Frame-tagged envelopes' publications and
// fails on an untagged envelope that reports one.
func observePublications(t *testing.T, ag *Agent, prompt string) []publicationObservation {
	t.Helper()
	var observed []publicationObservation
	for envelope := range ag.QueryStreamEnveloped(context.Background(), llm.TextContent(prompt)) {
		if envelope.FrameID == "" {
			if envelope.HostPublicationRevision != 0 {
				t.Fatalf("envelope without a Frame reported publication %d: %+v", envelope.HostPublicationRevision, envelope)
			}
			continue
		}
		observed = append(observed, publicationObservation{frame: envelope.FrameID, attempt: envelope.InvokeAttempt, publication: envelope.HostPublicationRevision})
	}
	return observed
}

// frameSuffix returns the Frame ordinal part of a default Frame ID.
func frameSuffix(frameID string) string {
	for i := len(frameID) - 1; i >= 0; i-- {
		if frameID[i] == '/' {
			return frameID[i+1:]
		}
	}
	return frameID
}

// #172/#83: a checked host publication returns its revision atomically, and
// every Frame built from that history reports exactly that revision. A later
// publication supersedes it and names the one it replaced.
func TestHostPublicationRevisionFirstFrameCarriesPublication(t *testing.T) {
	ag, err := New(Config{LLM: plainAnswerModel{}})
	if err != nil {
		t.Fatal(err)
	}
	// No host publication yet: unknown, even though the history is valid.
	for _, o := range observePublications(t, ag, "before") {
		if o.publication != 0 {
			t.Fatalf("Frame before any publication reported %d", o.publication)
		}
	}
	first, err := ag.ReplaceHistoryCheckedRevision([]llm.Message{llm.NewSystemMessage("rules v1")})
	if err != nil {
		t.Fatal(err)
	}
	second, err := ag.ReplaceHistoryCheckedRevision([]llm.Message{llm.NewSystemMessage("rules v2")})
	if err != nil {
		t.Fatal(err)
	}
	if first.Revision == 0 || first.Replaced != 0 || second.Revision <= first.Revision || second.Replaced != first.Revision {
		t.Fatalf("publications first=%+v second=%+v", first, second)
	}
	observed := observePublications(t, ag, "run")
	if len(observed) == 0 {
		t.Fatal("no Frame-tagged envelope")
	}
	for _, o := range observed {
		if o.publication != second.Revision {
			t.Fatalf("Frame %s reported %d, want the latest publication %d", o.frame, o.publication, second.Revision)
		}
	}
	// Revisions are unique across Agents: another Agent's publication can
	// never equal this Agent's.
	other, err := New(Config{LLM: plainAnswerModel{}})
	if err != nil {
		t.Fatal(err)
	}
	third, err := other.ReplaceHistoryCheckedRevision([]llm.Message{llm.NewSystemMessage("rules v1")})
	if err != nil {
		t.Fatal(err)
	}
	if third.Revision <= second.Revision || third.Replaced != 0 {
		t.Fatalf("second Agent publication=%+v after %+v", third, second)
	}
}

// publishingRetryModel publishes a system-only update during Frame 1's first
// attempt and fails it transiently; the retry calls a tool, and the next
// Frame answers. It records each attempt's system text as an independent
// oracle of what the request carried.
type publishingRetryModel struct {
	mu      sync.Mutex
	ag      *Agent
	systems []string
	publish func(*Agent) (HistoryPublication, error)
	result  HistoryPublication
	pubErr  error
	calls   int
}

func (*publishingRetryModel) Provider() string { return "fixture" }
func (*publishingRetryModel) Model() string    { return "publishing-retry" }
func (m *publishingRetryModel) Invoke(_ context.Context, req llm.InvokeRequest) (*llm.Completion, error) {
	m.mu.Lock()
	m.calls++
	call := m.calls
	system := ""
	for _, message := range req.Messages {
		if message.Role == llm.RoleSystem {
			system += message.Content.PlainText()
		}
	}
	m.systems = append(m.systems, system)
	m.mu.Unlock()
	switch call {
	case 1:
		result, err := m.publish(m.ag)
		m.mu.Lock()
		m.result, m.pubErr = result, err
		m.mu.Unlock()
		return nil, &llm.ProviderError{Provider: "fixture", StatusCode: 503, Message: "overloaded"}
	case 2:
		return &llm.Completion{StopReason: "tool_calls", ToolCalls: []llm.ToolCall{{ID: "echo-1", Type: "function", Function: llm.FunctionCall{Name: "echo", Arguments: `{}`}}}}, nil
	default:
		return &llm.Completion{StopReason: "stop", Content: llm.TextContent("finished")}, nil
	}
}

// A publication made while Frame 1 is in flight applies to the next Frame;
// Frame 1's retry still reports (and carries) the publication it was built
// from. A rejected publication changes nothing.
func TestHostPublicationRevisionMidQueryMovesNextFrameRetryKeepsOld(t *testing.T) {
	echo := tools.Func[struct{}]("echo", "echo", func(context.Context, struct{}, *tools.Container) (any, error) { return "ok", nil })
	model := &publishingRetryModel{publish: func(ag *Agent) (HistoryPublication, error) {
		current := ag.Messages()
		// A non-system change during the Query is rejected and records nothing.
		if _, err := ag.ReplaceHistoryCheckedRevision(append(llm.CloneMessages(current), llm.NewUserMessage("smuggled"))); !errors.Is(err, ErrActiveHistoryMutation) {
			return HistoryPublication{}, errors.New("non-system change was not rejected")
		}
		updated := llm.CloneMessages(current)
		updated[0] = llm.NewSystemMessage("rules plan")
		return ag.ReplaceHistoryCheckedRevision(updated)
	}}
	ag, err := New(Config{LLM: model, Tools: []tools.Tool{echo}, InvokeRetryMaxAttempts: 2, InvokeRetryBackoff: time.Millisecond, Warningf: func(string, ...any) {}})
	if err != nil {
		t.Fatal(err)
	}
	model.ag = ag
	admission, err := ag.ReplaceHistoryCheckedRevision([]llm.Message{llm.NewSystemMessage("rules base")})
	if err != nil {
		t.Fatal(err)
	}
	observed := observePublications(t, ag, "run")
	model.mu.Lock()
	mid, pubErr, systems := model.result, model.pubErr, append([]string(nil), model.systems...)
	model.mu.Unlock()
	if pubErr != nil {
		t.Fatal(pubErr)
	}
	if mid.Revision <= admission.Revision || mid.Replaced != admission.Revision {
		t.Fatalf("mid-query publication=%+v after %+v", mid, admission)
	}
	// Independent oracle: the provider saw the old prompt on both attempts of
	// Frame 1 and the new prompt on Frame 2.
	if len(systems) != 3 || systems[0] != "rules base" || systems[1] != "rules base" || systems[2] != "rules plan" {
		t.Fatalf("provider system prompts=%q", systems)
	}
	sawRetry, sawSecond := false, false
	for _, o := range observed {
		switch frameSuffix(o.frame) {
		case "1":
			sawRetry = sawRetry || o.attempt == 2
			if o.publication != admission.Revision {
				t.Fatalf("Frame 1 attempt %d reported %d, want its own publication %d", o.attempt, o.publication, admission.Revision)
			}
		case "2":
			sawSecond = true
			if o.publication != mid.Revision {
				t.Fatalf("Frame 2 reported %d, want the mid-query publication %d", o.publication, mid.Revision)
			}
		default:
			t.Fatalf("unexpected Frame %s", o.frame)
		}
	}
	if !sawRetry || !sawSecond {
		t.Fatalf("retry observed=%v second Frame observed=%v: %+v", sawRetry, sawSecond, observed)
	}
}

// An SDK compaction rewrites the history's system messages itself, so the
// Frames after it are unknown until the host publishes again; the next
// publication reports that it replaced an unknown one.
func TestHostPublicationRevisionUnknownAfterSDKCompaction(t *testing.T) {
	model := &relationScriptModel{steps: []func() (*llm.Completion, error){relationToolCall(75), relationToolCall(10), relationFinal}}
	ag := newRelationAgent(t, model, true)
	// The fixture's summarizer is the scripted model, so the published history
	// keeps the fixture's shape (it has no system message; that is still a
	// host publication of the system messages it carries: none).
	admission, err := ag.ReplaceHistoryCheckedRevision(ag.Messages())
	if err != nil {
		t.Fatal(err)
	}
	frames := map[string]uint64{}
	compacted := false
	for envelope := range ag.QueryStreamEnveloped(context.Background(), llm.TextContent("go")) {
		if event, ok := envelope.Event.(CompactionEvent); ok && event.Result.Compacted {
			compacted = true
		}
		if envelope.FrameID == "" {
			continue
		}
		if previous, seen := frames[envelope.FrameID]; seen && previous != envelope.HostPublicationRevision {
			t.Fatalf("Frame %s reported %d and %d", envelope.FrameID, previous, envelope.HostPublicationRevision)
		}
		frames[envelope.FrameID] = envelope.HostPublicationRevision
		// Independent oracle: the Frame built after the compaction names it.
		if envelope.RequestHistoryRelation == RequestHistoryCompactionApplied && frameSuffix(envelope.FrameID) != "2" {
			t.Fatalf("compaction applied before Frame %s, want 2", envelope.FrameID)
		}
	}
	if !compacted || len(frames) != 3 {
		t.Fatalf("compacted=%v frames=%v", compacted, frames)
	}
	// The SDK's own compaction resets the current publication to unknown but
	// is not a host publication: the latest host publication stays.
	if got := ag.LastHostPublicationRevision(); got != admission.Revision {
		t.Fatalf("latest host publication after an SDK compaction=%d, want %d", got, admission.Revision)
	}
	for frame, publication := range frames {
		want := uint64(0)
		if frameSuffix(frame) == "1" {
			want = admission.Revision
		}
		if publication != want {
			t.Fatalf("Frame %s reported %d, want %d (frames=%v)", frame, publication, want, frames)
		}
	}
	republished, err := ag.ReplaceHistoryCheckedRevision(ag.Messages())
	if err != nil {
		t.Fatal(err)
	}
	if republished.Replaced != 0 || republished.Revision <= admission.Revision {
		t.Fatalf("publication after the compaction=%+v", republished)
	}
	if got := ag.LastHostPublicationRevision(); got != republished.Revision {
		t.Fatalf("latest host publication=%d, want %d", got, republished.Revision)
	}
	model.mu.Lock()
	model.steps = []func() (*llm.Completion, error){relationFinal}
	model.mu.Unlock()
	for _, o := range observePublications(t, ag, "again") {
		if o.publication != republished.Revision {
			t.Fatalf("Frame %s after republication reported %d, want %d", o.frame, o.publication, republished.Revision)
		}
	}
}

// The Agent's configured SystemPrompt, inserted into an empty history by the
// SDK, is not a host publication: those Frames are unknown.
func TestHostPublicationRevisionUnknownForSDKInsertedSystemPrompt(t *testing.T) {
	ag, err := New(Config{LLM: plainAnswerModel{}, SystemPrompt: "configured"})
	if err != nil {
		t.Fatal(err)
	}
	cleared, err := ag.ReplaceHistoryCheckedRevision(nil)
	if err != nil || cleared.Revision == 0 {
		t.Fatalf("clear publication=%+v err=%v", cleared, err)
	}
	observed := observePublications(t, ag, "run")
	if len(observed) == 0 {
		t.Fatal("no Frame-tagged envelope")
	}
	for _, o := range observed {
		if o.publication != 0 {
			t.Fatalf("Frame %s reported %d for an SDK-inserted prompt", o.frame, o.publication)
		}
	}
	if messages := ag.Messages(); len(messages) == 0 || messages[0].Role != llm.RoleSystem || messages[0].Content.PlainText() != "configured" {
		t.Fatalf("configured prompt not inserted: %+v", messages)
	}
}

// A host-computed compaction commit is a host publication of its candidate.
func TestHostPublicationRevisionCommitCompactionHistoryPublishes(t *testing.T) {
	model := &relationScriptModel{steps: []func() (*llm.Completion, error){relationFinal}}
	ag := newRelationAgent(t, model, false)
	admission, err := ag.ReplaceHistoryCheckedRevision([]llm.Message{llm.NewSystemMessage("rules"), llm.NewUserMessage("a"), llm.NewAssistantMessage("b", nil)})
	if err != nil {
		t.Fatal(err)
	}
	source := ag.Messages()
	candidate := []llm.Message{llm.NewSystemMessage("rules"), llm.NewUserMessage("summary")}
	result, err := ag.CommitCompactionHistory(context.Background(), source, candidate, compaction.Result{Compacted: true})
	if err != nil || !result.Compacted {
		t.Fatalf("commit result=%+v err=%v", result, err)
	}
	after, err := ag.ReplaceHistoryCheckedRevision(ag.Messages())
	if err != nil {
		t.Fatal(err)
	}
	if after.Replaced == 0 || after.Replaced <= admission.Revision || after.Revision <= after.Replaced {
		t.Fatalf("commit did not publish: admission=%+v next=%+v", admission, after)
	}
}

// A host-requested but SDK-computed compaction (CompactPipelineNow) installs
// system messages the host did not publish: later Frames are unknown.
func TestHostPublicationRevisionUnknownAfterCompactPipelineNow(t *testing.T) {
	model := &relationScriptModel{steps: []func() (*llm.Completion, error){relationFinal}}
	ag := newRelationAgent(t, model, true)
	// The old tool result must precede a protected recent message to be snipped.
	if _, err := ag.ReplaceHistoryCheckedRevision(append(ag.Messages(), llm.NewUserMessage("latest"))); err != nil {
		t.Fatal(err)
	}
	result, err := ag.CompactLocalNow(context.Background(), 90)
	if err != nil || !result.Compacted {
		t.Fatalf("local compaction result=%+v err=%v", result, err)
	}
	observed := observePublications(t, ag, "go")
	if len(observed) == 0 {
		t.Fatal("no Frame-tagged envelope")
	}
	for _, o := range observed {
		if o.publication != 0 {
			t.Fatalf("Frame %s after an SDK-computed compaction reported %d", o.frame, o.publication)
		}
	}
}

// LastHostPublicationRevision is zero before any host publication and names
// the latest one, including a rejected CAS leaving it unchanged.
func TestLastHostPublicationRevisionTracksHostPublications(t *testing.T) {
	ag, err := New(Config{LLM: plainAnswerModel{}})
	if err != nil {
		t.Fatal(err)
	}
	if got := ag.LastHostPublicationRevision(); got != 0 {
		t.Fatalf("fresh Agent reported %d", got)
	}
	first, err := ag.ReplaceHistoryCheckedRevision([]llm.Message{llm.NewSystemMessage("one")})
	if err != nil {
		t.Fatal(err)
	}
	if got := ag.LastHostPublicationRevision(); got != first.Revision {
		t.Fatalf("after the first publication=%d, want %d", got, first.Revision)
	}
	if _, err := ag.ReplaceHistoryCheckedIfRevision(first.Revision+1000, []llm.Message{llm.NewSystemMessage("stale")}); err == nil {
		t.Fatal("stale CAS publication accepted")
	}
	if got := ag.LastHostPublicationRevision(); got != first.Revision {
		t.Fatalf("a rejected publication moved the revision to %d", got)
	}
	if err := ag.ReplaceHistoryChecked([]llm.Message{llm.NewSystemMessage("two")}); err != nil {
		t.Fatal(err)
	}
	if got := ag.LastHostPublicationRevision(); got <= first.Revision {
		t.Fatalf("after the second publication=%d, want > %d", got, first.Revision)
	}
}
