package agent

import (
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"strings"
	"sync/atomic"
	"time"
)

const EventEnvelopeSchemaVersion = 1

// RequestControlRequireDoneDisableThinking identifies only the SDK recovery
// control that disabled thinking, not full request/content parentage.
const RequestControlRequireDoneDisableThinking = "require_done_disable_thinking"

// RequestHistoryCompactionApplied reports that an automatic compaction,
// triggered by the usage of RequestHistorySourceFrameID, was published into
// history before this Frame's request was built. It names that one producer,
// not the sole source of the request's content.
const RequestHistoryCompactionApplied = "compaction_applied"

// RequestRecoveryStreamIdle reports that the stream of
// RequestRecoverySourceFrameID stalled and the driver appended a recovery
// reminder before this Frame's request was built. It names that one
// producer, not the sole source of the request's content.
const RequestRecoveryStreamIdle = "stream_idle_recovery"

// RequestSteeringAccepted reports that a non-empty user steering message
// entered history while RequestSteeringSourceFrameID was executing (its
// stream or tool block was interrupted, or its iteration ended) and before
// this Frame's request was built. It names that one producer, not the
// sole source of the request's content.
const RequestSteeringAccepted = "steering_accepted"

// RequestContinuationToolResults reports that this Frame's request carries
// the tool results of a closed tool block to which no model response had
// been accepted when it was built. RequestContinuationSourceFrameIDs names
// the Frames whose responses produced the answered tool calls: the Frame
// that finalized the block and every earlier Frame whose truncated
// tool-call fragments the SDK merged into those calls. It names those
// producers only, not every source of the request's content.
const RequestContinuationToolResults = "tool_results_carried"

// MaxRequestContinuationSources bounds RequestContinuationSourceFrameIDs. A
// set the SDK cannot report within it is left unreported, never truncated.
const MaxRequestContinuationSources = 16

type EventKind string

const (
	EventKindText              EventKind = "text"
	EventKindTextDelta         EventKind = "text_delta"
	EventKindThinking          EventKind = "thinking"
	EventKindThinkingDelta     EventKind = "thinking_delta"
	EventKindError             EventKind = "error"
	EventKindWarning           EventKind = "warning"
	EventKindHiddenUserMessage EventKind = "hidden_user_message"
	EventKindStepStart         EventKind = "step_start"
	EventKindStepComplete      EventKind = "step_complete"
	EventKindToolCall          EventKind = "tool_call"
	EventKindToolResult        EventKind = "tool_result"
	EventKindFinalResponse     EventKind = "final_response"
	EventKindUsage             EventKind = "usage"
	EventKindCompaction        EventKind = "compaction"
	EventKindAccounting        EventKind = "accounting"
	EventKindSteeringReceived  EventKind = "steering_received"
	EventKindAutoContinue      EventKind = "auto_continue"
)

type EventOrigin string

const (
	EventOriginModel       EventOrigin = "model"
	EventOriginProvider    EventOrigin = "provider"
	EventOriginSDKDriver   EventOrigin = "sdk_driver"
	EventOriginToolRuntime EventOrigin = "tool_runtime"
	EventOriginCompaction  EventOrigin = "compaction"
	EventOriginHost        EventOrigin = "host"
)

// EventEnvelope adds query-wide ordering metadata without replacing the typed
// Event payload. Sequence, not Timestamp, defines logical order.
type EventEnvelope struct {
	// Optional, producer-captured control provenance for this logical request.
	// Empty is unreported, not proof that the request has no other sources.
	// Compaction, steering and merged continuation content remain independent.
	RequestControlRelation      string `json:"RequestControlRelation,omitempty"`
	RequestControlSourceFrameID string `json:"RequestControlSourceFrameID,omitempty"`
	// Optional, producer-captured history provenance for this logical request:
	// an automatic compaction published since the previous Frame. It is set
	// only on the first Frame built after publication (retries reuse it) and
	// empty means unreported, not that history was unchanged.
	RequestHistoryRelation      string `json:"RequestHistoryRelation,omitempty"`
	RequestHistorySourceFrameID string `json:"RequestHistorySourceFrameID,omitempty"`
	// Optional, producer-captured recovery provenance for this logical
	// request: a recovery reminder appended after the source Frame stalled.
	// Set only on the first Frame built after it (retries reuse it); empty
	// means unreported.
	RequestRecoveryRelation      string `json:"RequestRecoveryRelation,omitempty"`
	RequestRecoverySourceFrameID string `json:"RequestRecoverySourceFrameID,omitempty"`
	// Optional, producer-captured steering provenance for this logical
	// request: accepted user steering appended after the source Frame.
	// Set only on the first Frame built after it (retries reuse it); empty
	// means unreported, not that no steering occurred.
	RequestSteeringRelation      string `json:"RequestSteeringRelation,omitempty"`
	RequestSteeringSourceFrameID string `json:"RequestSteeringSourceFrameID,omitempty"`
	// Optional, producer-captured continuation provenance for this logical
	// request (see RequestContinuationToolResults): the Frames, in build
	// order and at most MaxRequestContinuationSources, whose responses
	// produced the tool calls this request answers with tool results. It is
	// set on every Frame built while those results await an accepted model
	// response (a retry, or a Frame after steering or stream-idle recovery,
	// reuses it) and cleared once a response is accepted, a compaction
	// rewrites history or an ephemeral result is released. Empty means
	// unreported, not that the request carries no earlier output. It holds
	// Frame IDs only, never content, CallIDs or fingerprints; each envelope
	// owns its copy.
	RequestContinuationRelation       string   `json:"RequestContinuationRelation,omitempty"`
	RequestContinuationSourceFrameIDs []string `json:"RequestContinuationSourceFrameIDs,omitempty"`
	// HostPublicationRevision is the HistoryPublication.Revision whose system
	// messages this Frame's request carries: the latest host publication when
	// the request was built, provided the SDK had not changed the system
	// messages since (compaction, trim, configured SystemPrompt). Retries
	// reuse it. Zero is unknown, not "no publication"; it names a publication,
	// not its content, the final wire payload or delivery.
	HostPublicationRevision uint64 `json:"HostPublicationRevision,omitempty"`
	SchemaVersion           int
	QueryID                 string
	// FrameID identifies the explicitly correlated execution context, not the
	// sole provenance of aggregated continuation content. Empty means unknown.
	FrameID string `json:"FrameID,omitempty"`
	// InvokeAttempt counts SDK ChatModel entries within that logical frame;
	// zero means no invocation correlation. It does not count hidden HTTP retries.
	InvokeAttempt uint64 `json:"InvokeAttempt,omitempty"`
	// ToolBlockID names the accepted block of the finalizing Frame, not a
	// provider ToolCallID or proof of execution/delivery. Empty means unknown.
	ToolBlockID string `json:"ToolBlockID,omitempty"`
	// ToolCallOrdinal is one-based within that block; ToolBlockCallCount includes
	// accepted history-only tails that may produce no envelope. Zero is unknown.
	ToolCallOrdinal    uint64 `json:"ToolCallOrdinal,omitempty"`
	ToolBlockCallCount uint64 `json:"ToolBlockCallCount,omitempty"`
	// Intervention names an SDK intervention that was applied before this
	// event was produced; InterventionStage is then "applied" and
	// InterventionResult a fixed outcome label saying what was applied:
	//   - tool_suppressed: the suppressed tool result was committed to history.
	//   - reminder_queued: the suppression was committed and a reminder was
	//     queued; the reminder itself enters history only when the tool block
	//     closes, and never if the block fails first.
	//   - guard_downgraded: the repeat guard's state changed.
	//   - safety_fallback_accepted: the partial final answer was accepted; the
	//     strike counts the reminders appended before it.
	//   - recovery_reminder_appended: the stream-idle recovery reminder is in
	//     history.
	//   - history_compacted: overflow recovery replaced history before the new
	//     request.
	// Kinds: repeated_tool_signature (tool_suppressed, reminder_queued,
	// guard_downgraded), evidence_progress (tool_suppressed, reminder_queued),
	// require_done_reminder (safety_fallback_accepted), stream_idle_recovery
	// (recovery_reminder_appended) and context_overflow_recovery
	// (history_compacted). Empty means the event reports no intervention, not
	// that none was considered. It is not proof of delivery, of model
	// compliance or of full content provenance.
	//
	// The one exception to "applied" is the opt-in, observe-only
	// thinking_only kind (Config.ObserveThinkingOnlyResponses): its
	// InterventionStage is "detected" and its InterventionResult
	// "observed_only", on a "thinking_only_observed" WarnEvent, with no
	// strike. Detection is not application: nothing was changed.
	Intervention       string `json:"Intervention,omitempty"`
	InterventionStage  string `json:"InterventionStage,omitempty"`
	InterventionResult string `json:"InterventionResult,omitempty"`
	// InterventionStrike is the one-based applied strike of that intervention
	// within the Query; zero is unknown.
	InterventionStrike uint64 `json:"InterventionStrike,omitempty"`
	Sequence           uint64
	Origin             EventOrigin
	Kind               EventKind
	Timestamp          time.Time
	Event              Event
}

type eventOutput struct {
	legacy            chan Event
	enveloped         chan EventEnvelope
	queryID           string
	clock             func() time.Time
	sequence          atomic.Uint64
	dropStart         uint64
	criticalDropStart uint64
	// dropped and droppedCritical count this output's own allocated
	// envelopes that were not delivered; the receipt reports them at close.
	dropped         atomic.Uint64
	droppedCritical atomic.Uint64
	receipt         *QueryStreamReceipt
}

// QueryStreamReceipt is the producer's account of one Query stream. It is
// filled immediately before the stream channel closes, so a consumer that
// has observed the closed channel reads its final values and can compare the Sequence range it received
// with the range the SDK allocated. It reuses the Query's only Sequence; it
// is not a second counter, a delivery acknowledgement or a success report.
type QueryStreamReceipt struct {
	summary atomic.Pointer[QueryStreamSummary]
}

// QueryStreamSummary describes a closed Query stream.
type QueryStreamSummary struct {
	// QueryID is the QueryID carried by every envelope of the stream.
	QueryID string
	// LastSequence is the last Sequence allocated for the Query. Every
	// allocated envelope was either delivered or counted in DroppedEvents,
	// including any allocated after a terminal event.
	LastSequence uint64
	// DroppedEvents counts this stream's allocated envelopes that were not
	// delivered; DroppedCriticalEvents is the subset that were terminal or
	// consistency-critical. Unlike FinalResponseEvent's counts, which stop
	// at the final answer, they cover the whole stream.
	DroppedEvents         uint64
	DroppedCriticalEvents uint64
}

// Summary returns the stream's final summary. ok may become true just
// before the channel closes (the summary is published first); the values
// never change afterwards and are the complete account of the stream once
// the consumer has observed the close.
func (r *QueryStreamReceipt) Summary() (summary QueryStreamSummary, ok bool) {
	if r == nil {
		return QueryStreamSummary{}, false
	}
	if loaded := r.summary.Load(); loaded != nil {
		return *loaded, true
	}
	return QueryStreamSummary{}, false
}

// countDrop records one undelivered envelope of this output.
func (o *eventOutput) countDrop(ev Event) {
	if o == nil {
		return
	}
	o.dropped.Add(1)
	if isTerminalAgentEvent(ev) || isCriticalAgentEvent(ev) {
		o.droppedCritical.Add(1)
	}
}

func (o *eventOutput) setDropBaseline(dropped, critical uint64) {
	o.dropStart = dropped
	o.criticalDropStart = critical
}

func dropsSince(current, baseline uint64) uint64 {
	if current <= baseline {
		return 0
	}
	return current - baseline
}

func newEventOutput(bufferSize int, enveloped bool, queryID string, clock func() time.Time) *eventOutput {
	out := &eventOutput{queryID: queryID, clock: clock, receipt: &QueryStreamReceipt{}}
	if enveloped {
		out.enveloped = make(chan EventEnvelope, bufferSize)
	} else {
		out.legacy = make(chan Event, bufferSize)
	}
	return out
}

func (o *eventOutput) next(ev Event) EventEnvelope {
	return o.nextFrom(ev, "")
}

func (o *eventOutput) nextFrom(ev Event, origin EventOrigin) EventEnvelope {
	kind, classifiedOrigin := classifyEvent(ev)
	if origin == "" {
		origin = classifiedOrigin
	}
	now := time.Now()
	if o != nil && o.clock != nil {
		now = o.clock()
	}
	return EventEnvelope{
		SchemaVersion: EventEnvelopeSchemaVersion,
		QueryID:       o.queryID,
		Sequence:      o.sequence.Add(1),
		Origin:        origin,
		Kind:          kind,
		Timestamp:     now,
		Event:         ev,
	}
}

func (o *eventOutput) trySend(envelope EventEnvelope) bool {
	if o == nil {
		return false
	}
	if o.enveloped != nil {
		select {
		case o.enveloped <- envelope:
			return true
		default:
			return false
		}
	}
	select {
	case o.legacy <- envelope.Event:
		return true
	default:
		return false
	}
}

type eventSendOutcome uint8

const (
	eventSent eventSendOutcome = iota
	eventTurnCanceled
	eventSendTimedOut
)

func (o *eventOutput) sendUntil(envelope EventEnvelope, done <-chan struct{}, timeout <-chan time.Time) eventSendOutcome {
	if o.enveloped != nil {
		select {
		case o.enveloped <- envelope:
			return eventSent
		case <-done:
			return eventTurnCanceled
		case <-timeout:
			return eventSendTimedOut
		}
	}
	select {
	case o.legacy <- envelope.Event:
		return eventSent
	case <-done:
		return eventTurnCanceled
	case <-timeout:
		return eventSendTimedOut
	}
}

func (o *eventOutput) tryReceive() (EventEnvelope, bool) {
	if o.enveloped != nil {
		select {
		case envelope := <-o.enveloped:
			return envelope, true
		default:
			return EventEnvelope{}, false
		}
	}
	select {
	case event := <-o.legacy:
		return EventEnvelope{Event: event}, true
	default:
		return EventEnvelope{}, false
	}
}

func (o *eventOutput) sendAfterReceive(envelope EventEnvelope) {
	if o.enveloped != nil {
		o.enveloped <- envelope
		return
	}
	o.legacy <- envelope.Event
}

func (o *eventOutput) close() {
	// Publish the receipt before closing: the close happens before any
	// receive that observes it, so a consumer that saw the closed channel
	// reads the final values.
	if o.receipt != nil {
		o.receipt.summary.Store(&QueryStreamSummary{
			QueryID:               o.queryID,
			LastSequence:          o.sequence.Load(),
			DroppedEvents:         o.dropped.Load(),
			DroppedCriticalEvents: o.droppedCritical.Load(),
		})
	}
	if o.enveloped != nil {
		close(o.enveloped)
		return
	}
	close(o.legacy)
}

func classifyEvent(event Event) (EventKind, EventOrigin) {
	switch event := event.(type) {
	case TextEvent:
		return EventKindText, EventOriginModel
	case TextDeltaEvent:
		return EventKindTextDelta, EventOriginModel
	case ThinkingEvent:
		return EventKindThinking, EventOriginModel
	case ThinkingDeltaEvent:
		return EventKindThinkingDelta, EventOriginModel
	case ErrorEvent:
		return EventKindError, classifyErrorOrigin(event)
	case WarnEvent:
		return EventKindWarning, EventOriginSDKDriver
	case HiddenUserMessageEvent:
		return EventKindHiddenUserMessage, EventOriginSDKDriver
	case StepStartEvent:
		return EventKindStepStart, EventOriginToolRuntime
	case StepCompleteEvent:
		return EventKindStepComplete, EventOriginToolRuntime
	case ToolCallEvent:
		return EventKindToolCall, EventOriginToolRuntime
	case ToolResultEvent:
		return EventKindToolResult, EventOriginToolRuntime
	case FinalResponseEvent:
		return EventKindFinalResponse, EventOriginSDKDriver
	case UsageEvent:
		return EventKindUsage, EventOriginProvider
	case CompactionEvent:
		return EventKindCompaction, EventOriginCompaction
	case AccountingEvent:
		return EventKindAccounting, EventOriginSDKDriver
	case SteeringReceivedEvent:
		return EventKindSteeringReceived, EventOriginHost
	case AutoContinueEvent:
		return EventKindAutoContinue, EventOriginSDKDriver
	default:
		panic(fmt.Sprintf("agent: unclassified event type %T", event))
	}
}

func classifyErrorOrigin(event ErrorEvent) EventOrigin {
	switch strings.TrimSpace(event.Kind) {
	case "agent_busy", "invalid_tool_call_block", "max_iterations", "loop_guard", "doom_loop":
		return EventOriginSDKDriver
	}
	if strings.TrimSpace(event.Provider) != "" {
		return EventOriginProvider
	}
	return EventOriginSDKDriver
}

var fallbackQueryID atomic.Uint64

func newDefaultQueryID() string {
	var random [16]byte
	if _, err := rand.Read(random[:]); err == nil {
		return "query_" + hex.EncodeToString(random[:])
	}
	return fmt.Sprintf("query_fallback_%d_%d", time.Now().UnixNano(), fallbackQueryID.Add(1))
}
