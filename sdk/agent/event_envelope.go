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
	SchemaVersion int
	QueryID       string
	Sequence      uint64
	Origin        EventOrigin
	Kind          EventKind
	Timestamp     time.Time
	Event         Event
}

type eventOutput struct {
	legacy            chan Event
	enveloped         chan EventEnvelope
	queryID           string
	clock             func() time.Time
	sequence          atomic.Uint64
	dropStart         uint64
	criticalDropStart uint64
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
	out := &eventOutput{queryID: queryID, clock: clock}
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
	case "agent_busy", "canceled", "invalid_tool_call_block", "max_iterations", "loop_guard", "doom_loop":
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
