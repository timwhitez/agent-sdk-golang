package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net"
	"os"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unicode/utf8"

	sdkaccounting "github.com/timwhitez/agent-sdk-golang/sdk/accounting"
	"github.com/timwhitez/agent-sdk-golang/sdk/agent/compaction"
	"github.com/timwhitez/agent-sdk-golang/sdk/agent/messageorigin"
	"github.com/timwhitez/agent-sdk-golang/sdk/artifact"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

type ArtifactOwnerProvider func(context.Context) (artifact.Owner, error)

type Config struct {
	LLM          llm.ChatModel
	Tools        []tools.Tool
	SystemPrompt string
	// Warningf receives non-fatal runtime diagnostics. Empty uses log.Printf.
	Warningf func(format string, args ...any)

	// InitialMessages restores a previous conversation history.
	// If provided, the agent will not auto-insert SystemPrompt on first query unless you include it here.
	InitialMessages []llm.Message

	MaxIterations int
	// InvokeRetryMaxAttempts controls framework-level retries for transient invoke failures.
	// The value includes the initial attempt. Values <= 0 use a safe default.
	InvokeRetryMaxAttempts int
	// InvokeRetryBackoff is the base exponential backoff for framework invoke retries.
	// Values <= 0 disable backoff delay between retry attempts.
	InvokeRetryBackoff time.Duration
	// RepeatToolSignatureThreshold enables repeated tool-signature loop protection when > 0.
	// The guard emits a warning and resets when the same normalized signature appears
	// this many times within the rolling RepeatToolSignatureWindow.
	RepeatToolSignatureThreshold int
	// RepeatToolSignatureWindow controls how many recent tool signatures are tracked.
	// Values <= 0 use a safe default when RepeatToolSignatureThreshold is enabled.
	RepeatToolSignatureWindow int
	// LoopGuardStrikeThreshold controls how many repeated-signature strikes are
	// allowed before the generic guard downgrades. Deterministic read-only tools
	// use the per-query evidence-progress ledger instead.
	// Values <= 0 use a safe default when RepeatToolSignatureThreshold is enabled.
	LoopGuardStrikeThreshold int
	// LoopGuardUserMessage is injected into conversation history when a repeated-signature
	// strike is detected so the model can correct course. Empty uses a safe default.
	LoopGuardUserMessage string
	// MaxToolResultBytes bounds tool result text stored in history and emitted in ToolResultEvent.
	// Values <= 0 use a safe default.
	MaxToolResultBytes int
	// MaxToolResultTokens bounds the estimator-token size of provider-visible
	// tool result content. Values <= 0 derive a conservative budget from
	// MaxToolResultBytes.
	MaxToolResultTokens int
	// ToolResultTokenEstimator optionally supplies the estimator used for the
	// provider-visible tool-result token budget. Nil or non-positive estimates
	// fall back to the SDK byte approximation.
	ToolResultTokenEstimator func(string) int
	// AccountingEstimator identifies the estimator used for comparable
	// original/visible accounting. An incomplete identity leaves estimator-token
	// measurements unknown rather than inventing comparable values.
	AccountingEstimator sdkaccounting.Estimator
	// ArtifactOwner identifies the host session, agent, or run that owns
	// canonical objects created at the Agent tool-result boundary. It is the
	// compatibility fallback when ArtifactOwnerProvider is nil.
	ArtifactOwner artifact.Owner
	// ArtifactOwnerProvider resolves the current execution subject whenever an
	// oversized or existing canonical tool result reaches the Agent boundary.
	// It takes precedence over ArtifactOwner so one Agent can safely switch
	// sessions without assigning future objects to a stale owner.
	ArtifactOwnerProvider ArtifactOwnerProvider
	// ArtifactSink persists complete logical tool-result objects. The Agent
	// claims recoverability only when the sink succeeds and the resolver
	// capability below is registered.
	ArtifactSink artifact.Sink
	// ArtifactResolver validates canonical objects before local compaction
	// removes provider-history bytes. Hosts should bind it to the same store and
	// execution subject used by ArtifactSink.
	ArtifactResolver artifact.Resolver
	// ArtifactResolverCapability declares the model-callable recovery contract
	// that the host has actually registered.
	ArtifactResolverCapability artifact.ResolverCapability
	// ArtifactEnvelopeCodec serializes provider-visible canonical envelopes.
	// Nil uses artifact.JSONEnvelopeCodec.
	ArtifactEnvelopeCodec artifact.EnvelopeCodec
	// ToolResultDumpTTL controls how long oversized tool-result dump files are retained.
	// Values <= 0 use a safe default.
	ToolResultDumpTTL time.Duration
	// EventBufferSize controls the outbound event channel capacity used by QueryStream.
	// Values <= 0 use a safe default.
	EventBufferSize int
	// EventSendTimeout bounds how long event delivery waits when the outbound channel is full.
	// Values <= 0 use a safe default.
	EventSendTimeout time.Duration
	// EventDropLogEvery controls backpressure drop log frequency.
	// Values <= 0 use a safe default.
	EventDropLogEvery int
	// QueryIDGenerator optionally supplies opaque query IDs for EventEnvelope.
	// It may be called concurrently. Empty or whitespace-only results fall back
	// to an SDK-generated ID.
	QueryIDGenerator func() string
	// EventClock supplies observational EventEnvelope timestamps. Sequence is
	// the only ordering authority. It may be called concurrently. Nil uses
	// time.Now.
	EventClock func() time.Time
	// StreamIdleTimeout bounds how long a streaming response may go without any
	// event before it is treated as stalled (and auto-recovered). Values <= 0
	// use the package default (75s). Use a larger value for long single-step
	// reasoning that can legitimately stay silent for extended periods.
	StreamIdleTimeout time.Duration
	// StreamIdleMaxRecoveries caps how many times a stalled stream is
	// auto-recovered before surfacing an idle-timeout error. Values < 0 use the
	// package default; 0 disables auto-recovery.
	StreamIdleMaxRecoveries int
	ToolChoice              llm.ToolChoice

	Compaction *compaction.Config

	RequireDoneTool bool

	Deps *tools.Container
}

type Agent struct {
	llm                        llm.ChatModel
	systemPrompt               string
	maxIterations              int
	invokeRetryMax             int
	invokeRetryBackoff         time.Duration
	repeatSigThreshold         int
	repeatSigWindow            int
	loopGuardStrikeMax         int
	loopGuardUserMsg           string
	maxToolResultBytes         int
	maxToolResultTokens        int
	toolResultTokenEstimator   func(string) int
	accountingEstimator        sdkaccounting.Estimator
	artifactOwner              artifact.Owner
	artifactOwnerProvider      ArtifactOwnerProvider
	artifactSink               artifact.Sink
	artifactResolver           artifact.Resolver
	artifactResolverCapability artifact.ResolverCapability
	artifactEnvelopeCodec      artifact.EnvelopeCodec
	toolResultDumpTTL          time.Duration
	eventBufferSize            int
	eventSendTimeout           time.Duration
	eventDropLogEvery          uint64
	queryIDGenerator           func() string
	eventClock                 func() time.Time
	streamIdleTimeout          time.Duration
	streamIdleMaxRecov         int
	toolChoice                 llm.ToolChoice
	requireDone                bool
	warningf                   func(format string, args ...any)
	hasCompactor               bool

	compactionAdmissionObserved func()
	compactionShadowObserved    func()
	toolBlockStateObserved      func(*toolBlockState)

	repeatInterventionShadowObserved  func(interventionDecision, interventionDecision)
	repeatInterventionShadowEvaluator func(repeatedSignatureObservation) interventionDecision
	repeatResultRecycled              func(string) bool

	tools             []tools.Tool
	toolMap           map[string]tools.Tool
	toolMapNormalized map[string]tools.Tool
	deps              *tools.Container

	compactor *compaction.Service

	// compactionRuntimeMu protects installation of compactor/hasCompactor.
	// Complete operations hold a lifecycle use; a queued replacement is a
	// barrier for later top-level operations while retained child work may
	// finish against its parent's coherent generation.
	compactionRuntimeMu      sync.Mutex
	compactionRuntimeUses    int
	pendingCompactionRuntime *compactionRuntimeUpdate
	compactionRuntimeWaitCh  chan struct{}

	todoCompactionPending  atomic.Bool
	compactionRetryPending atomic.Bool
	compactionInFlight     atomic.Bool
	compactionGeneration   atomic.Uint64
	accountingSequence     atomic.Uint64

	// compactionFailureStreak counts consecutive compaction failures so the
	// loop can back off instead of re-running an expensive summary every turn.
	compactionFailureStreak atomic.Uint64
	compactionCooldownUntil atomic.Int64

	// compactionIdleCh is closed when an in-flight compaction finishes, so
	// boundary waits wake immediately instead of polling.
	compactionIdleMu sync.Mutex
	compactionIdleCh chan struct{}

	pendingCompactionMu sync.Mutex
	pendingCompaction   *pendingCompaction

	activeStageMu         sync.Mutex
	activeStageCancel     context.CancelFunc
	activeStageGeneration uint64
	activeStageSteering   bool

	mu              sync.Mutex
	messages        []llm.Message
	lastPromptCount int
	// Ephemeral cleanup state avoids full-history scans on every loop.
	// Results are grouped by a composite key of tool name + argument signature
	// (e.g. read|{path,offset,limit}) so that only genuinely redundant re-reads
	// of the *same* target are recycled; reading a different target never evicts
	// an earlier distinct result. ephemeralSigByCall maps a tool_call_id to its
	// argument signature, populated as assistant tool_calls are scanned.
	ephemeralByKey     map[string][]int
	ephemeralSigByCall map[string]string
	ephemeralScanFrom  int

	toolResultDumpsMu sync.Mutex
	toolResultDumps   map[string]toolResultDumpLifecycleEntry
	toolResultDumpDir string
	toolResultDumpID  string
	toolResultDumpIdx string

	eventDropCount atomic.Uint64
	// criticalEventDropCount counts drops of events whose loss makes the
	// delivered stream contradict the history the agent already mutated, as
	// opposed to merely omitting presentation content.
	criticalEventDropCount atomic.Uint64

	// turnCancelByOut maps an in-flight turn's event channel to that turn's
	// backpressure state (the context's Done channel plus the floor budget
	// already spent). emitEvent has no ctx parameter and is reached from
	// ~40 call sites plus the accounting/compaction helpers, so the signal is
	// looked up by output channel instead of threaded through all of them. It is
	// only consulted on the slow path (channel already full), which is exactly
	// where the bounded critical-event wait would otherwise be paid by a query
	// whose caller has already gone away.
	// turnActive is acquired synchronously before the turn goroutine starts,
	// preventing overlapping submissions from mutating shared turn state.
	turnActive      atomic.Bool
	turnCancelMu    sync.Mutex
	turnCancelByOut map[*eventOutput]*turnBackpressure
}

// turnBackpressure carries the per-turn state emitEvent needs on the slow path:
// whether the turn was abandoned, and how much of the turn's critical-event
// floor budget has already been spent waiting on a full channel.
type turnBackpressure struct {
	done <-chan struct{}
	// floorSpentNanos accumulates the wall time this turn spent inside the
	// extended critical-event floor. It is the quantity criticalEventFloorTurnBudget
	// bounds, so a tool-heavy turn cannot multiply the per-event floor by an
	// unbounded event count.
	floorSpentNanos atomic.Int64
	// floorBudgetWarned keeps the budget-exhaustion notice to one line per turn.
	floorBudgetWarned atomic.Bool
	// floorOverrideWarned keeps the notice that the critical-event floor
	// overrode the host's configured send budget to one line per turn.
	floorOverrideWarned atomic.Bool
}

type pendingCompaction struct {
	messages     []llm.Message
	snapshotLen  int
	result       compaction.Result
	triggerUsage *llm.Usage
}

// SteeringMsg represents a user message injected mid-turn for real-time steering.
// This enables the "Real-time steering" feature where users can send feedback
// while the agent is working, and the agent incorporates it immediately.
type SteeringMsg struct {
	Content string
}

const (
	defaultMaxToolResultBytes      = 50 * 1024
	toolResultTruncatedSuffix      = "\n...[truncated]"
	toolResultDumpPattern          = "agent-tool-result-*.txt"
	defaultToolResultDumpTTL       = 15 * time.Minute
	defaultEventBufferSize         = 32
	defaultEventSendTimeout        = 25 * time.Millisecond
	defaultEventDropLogEvery       = 25
	defaultInvokeRetryMax          = 2
	defaultRepeatSigWindow         = 24
	defaultLoopGuardStrikeMax      = 2
	defaultMaxContinuationTurns    = 3
	defaultRequireDoneMaxReminders = 2
	defaultStreamIdleTimeout       = 75 * time.Second
	defaultStreamIdleMaxRecoveries = 2
	maxInvokeRetryDelay            = 2 * time.Second
	// defaultDestroyedToolCompactThreshold triggers compaction once this many
	// tool results have been recycled to the ephemeral placeholder. A long turn
	// that reads hundreds of files accumulates that many zero-information
	// placeholders in context; compacting them out is exactly what the growing
	// placeholder count signals, even before the token watermark is reached.
	defaultDestroyedToolCompactThreshold = 24
	requireDoneReminderText              = messageorigin.RequireDoneReminderText
	defaultLoopGuardUserMsg              = messageorigin.DefaultLoopGuardText
	earlyStopReminderText                = messageorigin.EarlyStopReminderText
	streamIdleRecoveryText               = messageorigin.StreamIdleRecoveryText
	// ephemeralReleasedPlaceholder replaces an ephemeral tool result once it is
	// recycled to save context. It is intentionally actionable: the model must be
	// able to tell "recycled" apart from "read failed" and know how to continue
	// without re-issuing the identical call.
	ephemeralReleasedPlaceholder = "<earlier result released to save context; if still needed, re-read with a different offset/limit or continue from your notes — do NOT repeat the identical call>"
)

var writeToolResultDump = writeToolResultDumpFile
var agentStreamIdleTimeout = defaultStreamIdleTimeout
var agentStreamIdleMaxRecoveries = defaultStreamIdleMaxRecoveries

func New(cfg Config) (*Agent, error) {
	if cfg.LLM == nil {
		return nil, fmt.Errorf("agent: LLM is required")
	}
	// A negative MaxIterations means "unlimited" (no per-turn iteration cap);
	// the loop is then bounded only by tool-loop guards, idle detection, and
	// context cancellation. Zero falls back to the conservative default.
	if cfg.MaxIterations == 0 {
		cfg.MaxIterations = 200
	}
	if cfg.InvokeRetryMaxAttempts <= 0 {
		cfg.InvokeRetryMaxAttempts = defaultInvokeRetryMax
	}
	if cfg.RepeatToolSignatureThreshold > 0 {
		if cfg.RepeatToolSignatureWindow <= 0 {
			cfg.RepeatToolSignatureWindow = defaultRepeatSigWindow
		}
		if cfg.RepeatToolSignatureWindow < cfg.RepeatToolSignatureThreshold {
			cfg.RepeatToolSignatureWindow = cfg.RepeatToolSignatureThreshold
		}
		if cfg.LoopGuardStrikeThreshold < 0 {
			cfg.LoopGuardStrikeThreshold = 0
		}
		if cfg.LoopGuardStrikeThreshold == 0 {
			cfg.LoopGuardStrikeThreshold = defaultLoopGuardStrikeMax
		}
		cfg.LoopGuardUserMessage = strings.TrimSpace(cfg.LoopGuardUserMessage)
		if cfg.LoopGuardUserMessage == "" {
			cfg.LoopGuardUserMessage = defaultLoopGuardUserMsg
		}
	} else {
		cfg.RepeatToolSignatureThreshold = 0
		cfg.RepeatToolSignatureWindow = 0
		cfg.LoopGuardStrikeThreshold = 0
		cfg.LoopGuardUserMessage = ""
	}
	if cfg.MaxToolResultBytes <= 0 {
		cfg.MaxToolResultBytes = defaultMaxToolResultBytes
	}
	if cfg.MaxToolResultTokens <= 0 {
		cfg.MaxToolResultTokens = (cfg.MaxToolResultBytes + 3) / 4
	}
	if cfg.ArtifactEnvelopeCodec == nil {
		cfg.ArtifactEnvelopeCodec = artifact.JSONEnvelopeCodec{}
	}
	if cfg.ToolResultDumpTTL <= 0 {
		cfg.ToolResultDumpTTL = defaultToolResultDumpTTL
	}
	if cfg.EventBufferSize <= 0 {
		cfg.EventBufferSize = defaultEventBufferSize
	}
	if cfg.EventSendTimeout <= 0 {
		cfg.EventSendTimeout = defaultEventSendTimeout
	}
	if cfg.EventDropLogEvery <= 0 {
		cfg.EventDropLogEvery = defaultEventDropLogEvery
	}
	if cfg.StreamIdleTimeout <= 0 {
		cfg.StreamIdleTimeout = agentStreamIdleTimeout
	}
	if cfg.StreamIdleMaxRecoveries < 0 {
		cfg.StreamIdleMaxRecoveries = agentStreamIdleMaxRecoveries
	}
	if cfg.Deps == nil {
		cfg.Deps = tools.NewContainer()
	}
	if setter, ok := cfg.LLM.(llm.WarningSinkSetter); ok {
		setter.SetWarningf(cfg.Warningf)
	}

	ownedTools := make([]tools.Tool, len(cfg.Tools))
	toolMap := map[string]tools.Tool{}
	toolPositions := map[string]int{}
	for i, t := range cfg.Tools {
		if t.Name == "" {
			return nil, fmt.Errorf("agent: tool missing name")
		}
		if previous, exists := toolPositions[t.Name]; exists {
			return nil, fmt.Errorf("agent: duplicate tool name %q at positions %d and %d", t.Name, previous+1, i+1)
		}
		cloned, err := llm.CloneInvokeRequest(llm.InvokeRequest{Tools: []llm.ToolDefinition{t.Definition()}})
		if err != nil {
			return nil, fmt.Errorf("agent: clone tool %q schema: %w", t.Name, err)
		}
		t.Schema = cloned.Tools[0].Parameters
		ownedTools[i] = t
		toolPositions[t.Name] = i
		toolMap[t.Name] = t
	}

	compactionCfg := bindCompactionArtifactConfig(cfg.Compaction, cfg.Warningf, cfg.ArtifactOwner, cfg.ArtifactOwnerProvider, cfg.ArtifactSink, cfg.ArtifactResolver, cfg.ArtifactResolverCapability, cfg.ArtifactEnvelopeCodec)
	compSvc := compaction.NewService(compactionCfg)
	hasCompactor := compSvc != nil && compSvc.Config.Enabled
	if !hasCompactor {
		compSvc = nil
	}

	ag := &Agent{
		llm:                      cfg.LLM,
		systemPrompt:             cfg.SystemPrompt,
		maxIterations:            cfg.MaxIterations,
		invokeRetryMax:           cfg.InvokeRetryMaxAttempts,
		invokeRetryBackoff:       cfg.InvokeRetryBackoff,
		repeatSigThreshold:       cfg.RepeatToolSignatureThreshold,
		repeatSigWindow:          cfg.RepeatToolSignatureWindow,
		loopGuardStrikeMax:       cfg.LoopGuardStrikeThreshold,
		loopGuardUserMsg:         cfg.LoopGuardUserMessage,
		maxToolResultBytes:       cfg.MaxToolResultBytes,
		maxToolResultTokens:      cfg.MaxToolResultTokens,
		toolResultTokenEstimator: cfg.ToolResultTokenEstimator,
		accountingEstimator:      cfg.AccountingEstimator,
		artifactOwner:            cfg.ArtifactOwner,
		artifactOwnerProvider:    cfg.ArtifactOwnerProvider,
		artifactSink:             cfg.ArtifactSink,
		artifactResolver:         cfg.ArtifactResolver,
		artifactResolverCapability: artifact.ResolverCapability{
			Registered: cfg.ArtifactResolverCapability.Registered,
			Recovery:   cloneArtifactRecovery(cfg.ArtifactResolverCapability.Recovery),
		},
		artifactEnvelopeCodec: cfg.ArtifactEnvelopeCodec,
		toolResultDumpTTL:     cfg.ToolResultDumpTTL,
		eventBufferSize:       cfg.EventBufferSize,
		eventSendTimeout:      cfg.EventSendTimeout,
		eventDropLogEvery:     uint64(cfg.EventDropLogEvery),
		queryIDGenerator:      cfg.QueryIDGenerator,
		eventClock:            cfg.EventClock,
		streamIdleTimeout:     cfg.StreamIdleTimeout,
		streamIdleMaxRecov:    cfg.StreamIdleMaxRecoveries,
		toolChoice:            cfg.ToolChoice,
		requireDone:           cfg.RequireDoneTool,
		warningf:              cfg.Warningf,
		hasCompactor:          hasCompactor,
		tools:                 ownedTools,
		toolMap:               toolMap,
		toolMapNormalized:     buildNormalizedToolMap(toolMap, ownedTools),
		deps:                  cfg.Deps,
		compactor:             compSvc,
		toolResultDumps:       make(map[string]toolResultDumpLifecycleEntry),
		ephemeralByKey:        make(map[string][]int),
		ephemeralSigByCall:    make(map[string]string),
	}
	if len(cfg.InitialMessages) > 0 {
		ag.messages = llm.CloneMessages(cfg.InitialMessages)
	}
	ag.initToolResultDumpLifecycle(toolResultDumpNow())
	return ag, nil
}

func (a *Agent) warnf(format string, args ...any) {
	if a != nil && a.warningf != nil {
		a.warningf(format, args...)
		return
	}
	log.Printf(format, args...)
}

// UpdateCompactionConfig replaces the compaction service used by future
// operations. The call itself never blocks: while an old generation is in
// use, updates are coalesced (latest wins). Later top-level operations wait
// for that generation to drain, while child work already launched by an
// active operation finishes against the same coherent service/configuration.
func (a *Agent) UpdateCompactionConfig(cfg *compaction.Config) {
	if a == nil {
		return
	}
	cfg = bindCompactionArtifactConfig(cfg, a.warningf, a.artifactOwner, a.artifactOwnerProvider, a.artifactSink, a.artifactResolver, a.artifactResolverCapability, a.artifactEnvelopeCodec)
	compSvc := compaction.NewService(cfg)
	hasCompactor := compSvc != nil && compSvc.Config.Enabled
	if !hasCompactor {
		compSvc = nil
	}
	a.installOrQueueCompactionRuntime(&compactionRuntimeUpdate{service: compSvc, enabled: hasCompactor})
}

func (a *Agent) Messages() []llm.Message {
	a.mu.Lock()
	defer a.mu.Unlock()
	return llm.CloneMessages(a.messages)
}

func (a *Agent) ClearHistory() {
	a.mu.Lock()
	a.messages = nil
	a.resetEphemeralTrackingLocked()
	a.mu.Unlock()
	a.cleanupToolResultDumps(toolResultDumpNow(), true)
}

// ReplaceHistory replaces the current conversation history.
// Callers should include the system prompt message if they want it preserved.
func (a *Agent) ReplaceHistory(messages []llm.Message) {
	a.mu.Lock()
	a.messages = llm.CloneMessages(messages)
	a.resetEphemeralTrackingLocked()
	a.mu.Unlock()
	a.cleanupToolResultDumps(toolResultDumpNow(), true)
}

func (a *Agent) Query(ctx context.Context, text string) (string, error) {
	ch := a.QueryStream(ctx, llm.TextContent(text))
	final := ""
	var lastErr error
	for ev := range ch {
		if f, ok := ev.(FinalResponseEvent); ok {
			final = f.Content
		}
		if e, ok := ev.(ErrorEvent); ok {
			// Preserve provider/status info in the error string.
			if e.StatusCode != 0 {
				lastErr = fmt.Errorf("%s error (%d): %s", e.Provider, e.StatusCode, e.Message)
			} else if e.Provider != "" {
				lastErr = fmt.Errorf("%s error: %s", e.Provider, e.Message)
			} else {
				lastErr = fmt.Errorf("agent error: %s", e.Message)
			}
		}
	}
	return final, lastErr
}

func (a *Agent) QueryStream(ctx context.Context, input llm.Content) <-chan Event {
	return a.QueryStreamWithSteering(ctx, input, nil)
}

// QueryStreamEnveloped returns the same typed events as QueryStream with
// query-wide ordering metadata. It uses the same delivery and backpressure
// path; only the public channel projection differs.
func (a *Agent) QueryStreamEnveloped(ctx context.Context, input llm.Content) <-chan EventEnvelope {
	return a.QueryStreamEnvelopedWithSteering(ctx, input, nil)
}

// QueryStreamWithSteering is like QueryStream but accepts an optional steering channel.
// When steeringCh is non-nil, the agent checks for new user messages at natural breakpoints
// (before each LLM invocation and after each tool execution). Any received steering messages
// are appended to the conversation history as user messages, so the next LLM call will
// see them and can adjust its plan accordingly.
//
// The steering channel is caller-owned. The agent only reads from it and never closes it.
func (a *Agent) QueryStreamWithSteering(ctx context.Context, input llm.Content, steeringCh <-chan SteeringMsg) <-chan Event {
	return a.queryStreamWithSteering(ctx, input, steeringCh, false).legacy
}

// QueryStreamEnvelopedWithSteering is the enveloped form of
// QueryStreamWithSteering. The steering channel remains caller-owned.
func (a *Agent) QueryStreamEnvelopedWithSteering(ctx context.Context, input llm.Content, steeringCh <-chan SteeringMsg) <-chan EventEnvelope {
	return a.queryStreamWithSteering(ctx, input, steeringCh, true).enveloped
}

func (a *Agent) queryStreamWithSteering(ctx context.Context, input llm.Content, steeringCh <-chan SteeringMsg, enveloped bool) *eventOutput {
	bufferSize := defaultEventBufferSize
	if a != nil && a.eventBufferSize > 0 {
		bufferSize = a.eventBufferSize
	}
	out := newEventOutput(bufferSize, enveloped, a.newQueryID(), a.eventClock)
	if !a.turnActive.CompareAndSwap(false, true) {
		// Admission is synchronous and out is buffered, so callers receive a
		// deterministic terminal rejection without scheduling another goroutine.
		out.trySend(out.next(ErrorEvent{Kind: "agent_busy", Message: ErrAgentBusy.Error()}))
		out.close()
		return out
	}
	// Reserve synchronously when possible so an update made after this call
	// cannot overtake an already admitted turn. A pending replacement is
	// waited inside the goroutine so QueryStream remains non-blocking.
	runtimeRelease, runtimeAcquired := a.tryBeginCompactionRuntimeUse()
	unregisterTurn := a.registerTurnCancellation(out, ctx)
	go func(releaseCompactionRuntime func(), acquired bool) {
		// Later defers execute first. Runtime release happens before admission
		// is reopened and before channel close, so the next accepted operation
		// observes any queued replacement.
		defer out.close()
		defer a.turnActive.Store(false)
		defer unregisterTurn()
		if !acquired {
			var err error
			releaseCompactionRuntime, err = a.beginCompactionRuntimeUse(ctx)
			if err != nil {
				a.emitEventFrom(out, a.errEvent(err), EventOriginSDKDriver)
				return
			}
		}
		defer releaseCompactionRuntime()
		a.cleanupToolResultDumps(toolResultDumpNow(), false)

		if a.compactionInFlight.Load() {
			if err := a.waitForCompactionIdle(ctx, out); err != nil {
				return
			}
		}
		a.applyPendingCompaction(out)

		a.mu.Lock()
		if len(a.messages) == 0 && a.systemPrompt != "" {
			a.messages = append(a.messages, llm.NewSystemMessage(a.systemPrompt))
		}
		a.messages = append(a.messages, llm.Message{Role: llm.RoleUser, Content: input})
		a.mu.Unlock()

		earlyStopReminderSent := false
		requireDoneReminderLogged := false
		requireDoneReminders := 0
		forceRequireDoneToolChoice := false
		requireDoneRecoveryDisableThinkingActive := false
		seenToolCallHistory := false
		lastResponseID := ""
		pendingTextContinuation := ""
		pendingRequireDoneFinalText := ""
		pendingRequireDoneFinalResponseID := ""
		streamIdleRecoveries := 0
		streamIdleRecoveryTotal := 0
		usageFallbackWarned := false
		cont := newToolCallContinuation(defaultMaxContinuationTurns)
		repeatGuard := newRepeatedToolSignatureGuard(a.repeatSigThreshold, a.repeatSigWindow)
		progressLedger := newEvidenceProgressLedger(a.deps, a.compactionGeneration.Load())
		loopGuardStrikes := 0
		hasDoneTool := a.hasToolNamed("done")
		out.setDropBaseline(a.eventDropCount.Load(), a.criticalEventDropCount.Load())
		droppedThisTurn := func() uint64 {
			return dropsSince(a.eventDropCount.Load(), out.dropStart)
		}
		criticalDroppedThisTurn := func() uint64 {
			return dropsSince(a.criticalEventDropCount.Load(), out.criticalDropStart)
		}
		emitFinal := func(content, responseID string) {
			a.emitEvent(out, FinalResponseEvent{
				Content:               content,
				ResponseID:            responseID,
				Status:                "complete",
				StallRecoveries:       streamIdleRecoveryTotal,
				DroppedEvents:         droppedThisTurn(),
				DroppedCriticalEvents: criticalDroppedThisTurn(),
			})
		}
		emitPartialFinal := func(content, responseID, reason string) {
			a.emitEvent(out, FinalResponseEvent{
				Content:               content,
				ResponseID:            responseID,
				Status:                "partial",
				Reason:                strings.TrimSpace(reason),
				StallRecoveries:       streamIdleRecoveryTotal,
				DroppedEvents:         droppedThisTurn(),
				DroppedCriticalEvents: criticalDroppedThisTurn(),
			})
		}
		emitErr := func(e ErrorEvent, origin EventOrigin) {
			e.StallRecoveries = streamIdleRecoveryTotal
			a.emitEventFrom(out, e, origin)
		}
		emitSDKErr := func(e ErrorEvent) {
			emitErr(e, EventOriginSDKDriver)
		}
		emitCompactionErr := func(e ErrorEvent) {
			origin := EventOriginCompaction
			if ctx.Err() != nil {
				origin = EventOriginSDKDriver
			}
			emitErr(e, origin)
		}
		var activeToolBlock *toolBlockState
		finishToolBlock := func() {
			block := activeToolBlock
			activeToolBlock = nil
			if block == nil {
				return
			}
			if err := block.validateClosed(); err != nil {
				a.warnf("warning: tool block shadow invariant mismatch: %v", err)
			}
			if a.toolBlockStateObserved != nil {
				a.toolBlockStateObserved(block)
			}
		}
		defer finishToolBlock()

		// maxIterations < 0 means unlimited: the loop is then bounded only by
		// tool-loop guards, idle detection, and context cancellation.
		unlimitedIter := a.maxIterations < 0
		for iter := 0; unlimitedIter || iter < a.maxIterations; iter++ {
			// A synchronous tool callback may cancel the turn after returning its
			// result. Enforce cancellation at the provider-admission boundary so a
			// context-ignoring model cannot receive one more stale request.
			if err := ctx.Err(); err != nil {
				emitSDKErr(a.errEvent(err))
				return
			}
			if a.compactionInFlight.Load() {
				if err := a.waitForCompactionIdle(ctx, out); err != nil {
					emitSDKErr(a.errEvent(err))
					return
				}
			}
			a.applyPendingCompaction(out)

			// *** Boundary-aware steering: check for new user messages before each LLM call ***
			if a.drainSteering(steeringCh, out) > 0 {
				requireDoneReminders = 0
				forceRequireDoneToolChoice = false
				requireDoneRecoveryDisableThinkingActive = false
				pendingRequireDoneFinalText = ""
				pendingRequireDoneFinalResponseID = ""
			}

			// Remove old ephemeral messages before the next LLM call.
			a.destroyEphemeralMessages()

			messages := a.Messages()
			toolDefs := make([]llm.ToolDefinition, 0, len(a.tools))
			for _, t := range a.tools {
				if t.Hidden {
					continue
				}
				toolDefs = append(toolDefs, t.Definition())
			}

			requestToolChoice := a.toolChoice
			if forceRequireDoneToolChoice {
				switch strings.ToLower(strings.TrimSpace(string(requestToolChoice))) {
				case "", "auto":
					requestToolChoice = llm.ToolChoice("required")
				}
			}
			// Steering delivery and request preparation above can synchronously
			// unblock a host that cancels the root turn. Recheck at the actual
			// provider-admission boundary, not only at iteration entry.
			if err := ctx.Err(); err != nil {
				emitSDKErr(a.errEvent(err))
				return
			}
			comp, streamedText, err := a.invokeCompletionWithRetryAndSteering(ctx, llm.InvokeRequest{
				Messages:   messages,
				Tools:      toolDefs,
				ToolChoice: requestToolChoice,
				// A forced tool_choice is illegal on some providers (Anthropic) while
				// extended thinking is enabled. Keep thinking disabled for the whole
				// require-done recovery subloop: if the forced call chooses an ordinary
				// work tool, its follow-up request must stay in the same thinking mode
				// until done, steering, or turn termination closes the subloop.
				DisableThinking: requireDoneRecoveryDisableThinkingActive,
			}, out, steeringCh)
			if err != nil {
				// Check for steering interrupt - handle specially
				var steerErr *llm.SteeringInterruptError
				if errors.As(err, &steerErr) {
					requireDoneReminders = 0
					forceRequireDoneToolChoice = false
					requireDoneRecoveryDisableThinkingActive = false
					pendingTextContinuation = ""
					pendingRequireDoneFinalText = ""
					pendingRequireDoneFinalResponseID = ""
					// Save partial assistant output, including a terminal
					// provider-state-only event received just before steering.
					if comp != nil && (!comp.Content.IsEmpty() || llm.HasProviderState(comp.Content)) {
						a.mu.Lock()
						a.messages = append(a.messages, llm.Message{
							Role:    llm.RoleAssistant,
							Content: llm.CloneContent(comp.Content),
						})
						a.mu.Unlock()
					}
					// The provider already billed what it produced before the
					// interrupt. This path continues the loop instead of ending
					// the turn, so the tokens are only recorded here.
					a.emitPartialUsage(out, comp)
					if msg := strings.TrimSpace(steerErr.Message); msg != "" {
						a.mu.Lock()
						a.messages = append(a.messages, llm.NewUserMessage(msg))
						a.mu.Unlock()
					}
					// A non-empty message was consumed directly from the channel. An
					// empty message means the host canceled a later stage after that
					// steering had already been appended at an earlier boundary.
					if strings.TrimSpace(steerErr.Message) != "" {
						a.emitEvent(out, SteeringReceivedEvent{Content: steerErr.Message})
					}
					continue
				}

				var idleErr *llm.StreamIdleTimeoutError
				if errors.As(err, &idleErr) {
					maxRecov := agentStreamIdleMaxRecoveries
					if a != nil {
						maxRecov = a.streamIdleMaxRecov
					}
					if streamIdleRecoveries < maxRecov {
						streamIdleRecoveries++
						streamIdleRecoveryTotal++
						if comp != nil && !comp.Content.IsEmpty() {
							a.mu.Lock()
							a.messages = append(a.messages, llm.Message{
								Role:    llm.RoleAssistant,
								Content: comp.Content,
							})
							a.mu.Unlock()
						}
						// Same as the steering path: the stalled response was
						// already billed, and the recovery continues the loop
						// rather than ending the turn, so emit it here.
						a.emitPartialUsage(out, comp)
						a.mu.Lock()
						a.messages = append(a.messages, messageorigin.NewInternalUserMessage(messageorigin.KindStreamIdleRecovery, streamIdleRecoveryText))
						a.mu.Unlock()
						a.warnf(
							"warning: response stream idle-timed out after %s; auto-recovering (%d/%d)",
							idleErr.Duration,
							streamIdleRecoveries,
							maxRecov,
						)
						continue
					}
				}

				// Save partial assistant message if any text was streamed,
				// so the conversation history reflects what the user saw.
				if comp != nil && !comp.Content.IsEmpty() {
					a.mu.Lock()
					a.messages = append(a.messages, llm.Message{
						Role:    llm.RoleAssistant,
						Content: comp.Content,
					})
					a.mu.Unlock()
				}
				_ = streamedText
				// The provider already billed whatever it produced before the
				// error, and the partial completion carries that usage. Emit it
				// before the terminal error or the tokens never reach the
				// accounting journal, which silently under-counts the ledger.
				a.emitPartialUsage(out, comp)
				origin := EventOriginProvider
				if ctx.Err() != nil {
					origin = EventOriginSDKDriver
				}
				emitErr(a.errEvent(err), origin)
				return
			}
			streamIdleRecoveries = 0
			responseID := strings.TrimSpace(comp.ResponseID)
			if responseID != "" {
				lastResponseID = responseID
			}
			comp.ToolCalls = ensureSyntheticToolCallIDs(comp.ToolCalls)
			comp.Usage = llm.NormalizeUsage(comp.Usage)
			if comp.Usage != nil && !llm.PromptUsageIsProviderValid(comp.Usage) {
				estimatedPrompt := llm.EstimateMessagesTokens(messages)
				comp.Usage = llm.WithPromptEstimate(comp.Usage, estimatedPrompt)
				if comp.Usage != nil && comp.Usage.PromptTokensSource == llm.PromptTokensSourceEstimate && !usageFallbackWarned {
					usageFallbackWarned = true
					a.emitEvent(out, WarnEvent{
						Kind:    "provider_usage_prompt_tokens_missing",
						Message: "provider reported zero or missing prompt tokens for a non-empty request; using a local prompt-token estimate for context, compaction, and budget accounting",
						Metadata: map[string]any{
							"prompt_tokens_source":    comp.Usage.PromptTokensSource,
							"prompt_tokens_semantics": comp.Usage.PromptTokensSemantics,
							"prompt_tokens_effective": comp.Usage.PromptTokens,
							"total_tokens_effective":  comp.Usage.TotalTokens,
							"provider_prompt_tokens":  comp.Usage.ProviderPromptTokens,
							"provider_total_tokens":   comp.Usage.ProviderTotalTokens,
						},
					})
				}
			}

			for _, diag := range comp.Diagnostics {
				if strings.TrimSpace(diag.Message) == "" {
					continue
				}
				kind := strings.TrimSpace(diag.Kind)
				if kind == "" {
					kind = "provider_diagnostic"
				}
				a.emitEvent(out, WarnEvent{Kind: kind, Message: strings.TrimSpace(diag.Message)})
			}

			if comp.Usage != nil {
				a.emitUsageWithAccounting(out, *comp.Usage, responseID)
			}
			if first, second, duplicate := duplicateToolCallIDPositions(comp.ToolCalls); duplicate {
				if cont.hasPending() {
					a.mu.Lock()
					cont.discardPartialToolCalls(a.messages)
					a.mu.Unlock()
				}
				emitErr(ErrorEvent{
					Provider: a.llm.Provider(),
					Kind:     "invalid_tool_call_block",
					Message:  fmt.Sprintf("provider returned duplicate tool_call_id values at positions %d and %d; no tools were executed", first+1, second+1),
				}, EventOriginSDKDriver)
				return
			}

			if comp.Thinking != "" {
				a.emitEvent(out, ThinkingEvent{Content: comp.Thinking})
			}
			if !streamedText {
				if txt := comp.PlainText(); txt != "" {
					a.emitEvent(out, TextEvent{Content: txt})
				}
			}

			// Append assistant message.
			a.mu.Lock()
			a.messages = append(a.messages, llm.Message{
				Role:      llm.RoleAssistant,
				Content:   llm.CloneContent(comp.Content),
				ToolCalls: comp.ToolCalls,
			})
			msgIndex := len(a.messages) - 1
			a.mu.Unlock()
			postCompletionEstimate := 0
			if a.hasCompactor && a.compactor != nil {
				postCompletionEstimate = a.compactor.EstimateMessages(a.Messages())
			}
			additionalSinceCompletion := func() int {
				if postCompletionEstimate <= 0 || a.compactor == nil {
					return 0
				}
				current := a.compactor.EstimateMessages(a.Messages())
				if current <= postCompletionEstimate {
					return 0
				}
				return current - postCompletionEstimate
			}
			pendingMessageTokens := func(message llm.Message) int {
				if a.compactor == nil {
					return 0
				}
				return a.compactor.EstimateMessages([]llm.Message{message})
			}

			if comp.HasToolCalls() {
				pendingTextContinuation = ""
			}

			// Tool call continuation: if model returned tool calls with max_tokens,
			// save partial and auto-continue to get the rest.
			if comp.HasToolCalls() && comp.StopReason == "max_tokens" {
				seenToolCallHistory = true
				turn, allowed := cont.nextTurn()
				if !allowed {
					a.emitEvent(out, WarnEvent{
						Message: fmt.Sprintf("tool-call continuation exceeded %d turns; resetting continuation state and requesting a split response", cont.maxTurns),
						Kind:    "continuation_limit",
					})
					// The truncated tool_use blocks will never receive a
					// tool_result, and a user reminder is appended next: strip
					// them from history or every later request is invalid.
					a.discardContinuationToolCalls(&cont, msgIndex)
					reminder := messageorigin.NewInternalUserMessage(messageorigin.KindToolCallContinuation, messageorigin.ToolCallContinuationLimitText)
					if a.hasCompactor {
						if err := a.checkAndCompactWithGrowth(ctx, comp, out, additionalSinceCompletion(), pendingMessageTokens(reminder)); err != nil {
							emitCompactionErr(a.errEvent(err))
							return
						}
					}
					a.mu.Lock()
					a.messages = append(a.messages, reminder)
					a.mu.Unlock()
					continue
				}
				cont.addPartial(msgIndex, comp.ToolCalls)
				a.emitEvent(out, WarnEvent{
					Message: fmt.Sprintf("continuing truncated tool-call arguments (%d/%d)", turn, cont.maxTurns),
					Kind:    "continuation",
				})
				reminder := messageorigin.NewInternalUserMessage(messageorigin.KindToolCallContinuation, messageorigin.ResponseTruncatedContinuationText)
				if a.hasCompactor {
					if err := a.checkAndCompactWithGrowth(ctx, comp, out, additionalSinceCompletion(), pendingMessageTokens(reminder)); err != nil {
						emitCompactionErr(a.errEvent(err))
						return
					}
				}
				a.mu.Lock()
				a.messages = append(a.messages, reminder)
				a.mu.Unlock()
				a.emitAutoContinue(out, "max_tokens", responseID)
				continue
			}

			continuationMergeDiagnostics := map[string][]string(nil)

			// Merge pending tool call continuations.
			if comp.HasToolCalls() && cont.hasPending() {
				seenToolCallHistory = true
				merged := ensureSyntheticToolCallIDs(cont.mergeToolCalls(comp.ToolCalls))
				continuationMergeDiagnostics = cont.mergeDiagnosticsForCalls(merged)
				if !allToolArgsValid(merged) {
					turn, allowed := cont.nextTurn()
					if !allowed {
						a.emitEvent(out, WarnEvent{
							Message: fmt.Sprintf("tool-call continuation exceeded %d turns; resetting continuation state and requesting a split response", cont.maxTurns),
							Kind:    "continuation_limit",
						})
						// The merged tool_use blocks are being abandoned and a
						// user reminder follows: strip them from history so no
						// tool_use is left without a tool_result.
						a.discardContinuationToolCalls(&cont, msgIndex)
						reminder := messageorigin.NewInternalUserMessage(messageorigin.KindToolCallContinuation, messageorigin.InvalidToolCallContinuationText)
						if a.hasCompactor {
							if err := a.checkAndCompactWithGrowth(ctx, comp, out, additionalSinceCompletion(), pendingMessageTokens(reminder)); err != nil {
								emitCompactionErr(a.errEvent(err))
								return
							}
						}
						a.mu.Lock()
						a.messages = append(a.messages, reminder)
						a.mu.Unlock()
						continue
					}
					// Still invalid JSON — keep accumulating.
					cont.setAccumulated(merged, msgIndex)
					a.emitEvent(out, WarnEvent{
						Message: fmt.Sprintf("tool-call merge remained invalid; requesting continuation (%d/%d)", turn, cont.maxTurns),
						Kind:    "continuation",
					})
					reminder := messageorigin.NewInternalUserMessage(messageorigin.KindToolCallContinuation, messageorigin.ResponseTruncatedContinuationText)
					if a.hasCompactor {
						if err := a.checkAndCompactWithGrowth(ctx, comp, out, additionalSinceCompletion(), pendingMessageTokens(reminder)); err != nil {
							emitCompactionErr(a.errEvent(err))
							return
						}
					}
					a.mu.Lock()
					a.messages = append(a.messages, reminder)
					a.mu.Unlock()
					a.emitAutoContinue(out, "max_tokens", responseID)
					continue
				}
				// Valid merged tool calls — clean up partials and update.
				a.mu.Lock()
				cont.clearPartialToolCalls(a.messages, msgIndex)
				comp.ToolCalls = merged
				a.messages[msgIndex] = llm.Message{Role: llm.RoleAssistant, Content: comp.Content, ToolCalls: merged}
				a.mu.Unlock()
			}
			if comp.HasToolCalls() {
				seenToolCallHistory = true
				requireDoneReminders = 0 // reset safety valve on tool usage
				forceRequireDoneToolChoice = false
			}
			if comp.StopReason != "max_tokens" {
				// The provider finished this response, so the truncation episode
				// is over and a later truncation may use the budget again.
				cont.rearm()
				// A completed response without tool calls abandons whatever
				// continuation was still pending: its argument fragments will
				// never arrive. Strip those tool_use blocks from history now —
				// otherwise they stay permanently unpaired and every later
				// request only remains sendable because the outbound repair
				// masks the malformed history.
				if !comp.HasToolCalls() && cont.hasPending() {
					a.mu.Lock()
					cont.discardPartialToolCalls(a.messages)
					a.mu.Unlock()
					cont.reset()
				}
			}
			// Stopping condition.
			if !comp.HasToolCalls() {
				plainText := comp.PlainText()
				combinedText := pendingTextContinuation + plainText
				clearPendingTextContinuation := func() {
					pendingTextContinuation = ""
				}
				// Auto-continue: if the response was truncated due to max_tokens,
				// check compaction and send a continuation prompt.
				if comp.StopReason == "max_tokens" {
					pendingTextContinuation = combinedText
					reminder := messageorigin.NewInternalUserMessage(messageorigin.KindMaxTokensContinuation, messageorigin.ResponseTruncatedContinuationText)
					if a.hasCompactor {
						if err := a.checkAndCompactWithGrowth(ctx, comp, out, additionalSinceCompletion(), pendingMessageTokens(reminder)); err != nil {
							emitCompactionErr(a.errEvent(err))
							return
						}
					}
					a.mu.Lock()
					a.messages = append(a.messages, reminder)
					a.mu.Unlock()
					a.emitAutoContinue(out, "max_tokens", responseID)
					continue
				}
				if !a.requireDone {
					// Generic early-stop reminder: once tools have been called in this run,
					// ask for an explicit done tool completion before ending.
					if hasDoneTool && seenToolCallHistory && !earlyStopReminderSent {
						earlyStopReminderSent = true
						a.emitEvent(out, WarnEvent{
							Message: "detected text-only stop after tool usage; prompting explicit done-tool completion",
							Kind:    "early_stop",
						})
						reminder := messageorigin.NewInternalUserMessage(messageorigin.KindEarlyStop, earlyStopReminderText)
						if a.hasCompactor {
							if err := a.checkAndCompactWithGrowth(ctx, comp, out, additionalSinceCompletion(), pendingMessageTokens(reminder)); err != nil {
								emitCompactionErr(a.errEvent(err))
								return
							}
						}
						a.mu.Lock()
						a.messages = append(a.messages, reminder)
						a.mu.Unlock()
						continue
					}
					// compaction check
					if a.hasCompactor {
						_ = a.checkAndCompact(ctx, comp, out)
					}
					clearPendingTextContinuation()
					emitFinal(combinedText, responseID)
					return
				}
				// require done tool: only enforce when tools have actually been
				// used in this run. Pure text Q&A (no tool calls ever) terminates
				// naturally — this prevents self-exciting loops on simple greetings.
				if a.requireDone && !comp.HasToolCalls() {
					if !seenToolCallHistory {
						// No tools were ever called — accept text-only as terminal.
						if a.hasCompactor {
							_ = a.checkAndCompact(ctx, comp, out)
						}
						clearPendingTextContinuation()
						emitFinal(combinedText, responseID)
						return
					}
					// Tools were used earlier; enforce done-tool completion.
					// Preserve the latest non-empty post-tool text so that if the
					// safety valve fires we surface the model's most recent answer
					// rather than a stale earlier one.
					if txt := strings.TrimSpace(combinedText); txt != "" {
						pendingRequireDoneFinalText = txt
						pendingRequireDoneFinalResponseID = responseID
					}
					requireDoneReminders++
					if !requireDoneReminderLogged {
						a.warnf("warning: RequireDoneTool is true but model stopped with text-only after tool usage; prompting done-tool reminder")
						requireDoneReminderLogged = true
					}
					// Safety valve: cap consecutive reminders to prevent runaway loops.
					// This is an expected, bounded fallback (e.g. a provider that keeps
					// answering in text under extended thinking, where forcing a tool is
					// not permitted), not a fatal abort: accept the model's latest
					// post-tool text and complete the turn.
					if requireDoneReminders > defaultRequireDoneMaxReminders {
						a.emitEvent(out, WarnEvent{
							Message: "require-done: model kept answering with text after tool usage and could not be pushed to the done tool (e.g. forced tool choice is unavailable under extended thinking); accepting its latest response and completing the turn",
							Kind:    "require_done_safety",
						})
						if a.hasCompactor {
							_ = a.checkAndCompact(ctx, comp, out)
						}
						finalContent := combinedText
						finalResponseID := responseID
						if preserved := strings.TrimSpace(pendingRequireDoneFinalText); preserved != "" {
							finalContent = preserved
							if preservedResponseID := strings.TrimSpace(pendingRequireDoneFinalResponseID); preservedResponseID != "" {
								finalResponseID = preservedResponseID
							}
						}
						clearPendingTextContinuation()
						requireDoneRecoveryDisableThinkingActive = false
						emitPartialFinal(finalContent, finalResponseID, "require_done_safety")
						return
					}
					forceRequireDoneToolChoice = true
					requireDoneRecoveryDisableThinkingActive = true
					reminder := messageorigin.NewInternalUserMessage(messageorigin.KindRequireDone, requireDoneReminderText)
					if a.hasCompactor {
						if err := a.checkAndCompactWithGrowth(ctx, comp, out, additionalSinceCompletion(), pendingMessageTokens(reminder)); err != nil {
							emitCompactionErr(a.errEvent(err))
							return
						}
					}
					a.mu.Lock()
					a.messages = append(a.messages, reminder)
					a.mu.Unlock()
					continue
				}
				continue
			}

			// Execute tool calls with alias resolution and unknown-tool fallback.
			//
			// pendingBlockMessages buffers user-role messages (loop-guard and
			// evidence-recovery reminders, drained steering) that are produced
			// while the assistant tool-call block is still open. A provider
			// rejects a tool_use block whose tool_result messages are
			// interleaved with user text, and the malformed history would stay
			// in place and fail every later turn — so these are only appended
			// once every tool_use in the block has a matching tool_result.
			activeToolBlock = newToolBlockState(comp.ToolCalls)
			var pendingBlockMessages []llm.Message
			for idx, tc := range comp.ToolCalls {
				if err := ctx.Err(); err != nil {
					// Close every unstarted tool call before terminating so a later
					// turn never inherits an unpaired assistant tool-call block.
					a.appendCancellationSkippedToolResults(comp.ToolCalls[idx:])
					activeToolBlock.markTerminalRange(idx, toolCallAccepted, "root_cancel_before_start")
					a.appendMessages(pendingBlockMessages)
					pendingBlockMessages = nil
					emitSDKErr(a.errEvent(err))
					return
				}
				step := idx + 1
				originalName := tc.Function.Name

				// Resolve tool: exact match → normalized/alias match → fallback
				tool, resolvedName, found, _ := a.resolveToolByName(tc.Function.Name)
				unknownToolFallback := false
				execArgs := tc.Function.Arguments

				if !found {
					unknownToolFallback = true
					resolvedName = "invalid"
					execArgs = wrapInvalidToolArgs(originalName, tc.Function.Arguments)
					if inv, ok := a.toolMap["invalid"]; ok {
						tool = inv
					} else {
						tool = autoInvalidTool()
					}
				}

				norm := tools.NormalizeToolArgs(resolvedName, execArgs, tool.Schema)
				evidenceReq, evidenceTool := newEvidenceRequest(resolvedName, norm.Normalized, execArgs, a.deps)
				if !strings.EqualFold(strings.TrimSpace(resolvedName), "done") {
					pendingRequireDoneFinalText = ""
					pendingRequireDoneFinalResponseID = ""
				}
				if mergeWarnings := continuationMergeDiagnostics[tc.ID]; len(mergeWarnings) > 0 {
					norm.Meta = appendToolArgMergeDiagnostics(norm.Meta, mergeWarnings)
					for _, warning := range mergeWarnings {
						a.warnf("warning: tool-call argument merge conflict for call %q: %s", tc.ID, warning)
					}
				}
				if repeatGuard != nil && !evidenceTool {
					signature := normalizeToolSignature(resolvedName, norm.Normalized, execArgs)
					seen, blocked := repeatGuard.observe(signature)
					detected := blocked
					lastResultRecycled := false
					if blocked && repeatGuard.exhausted {
						if a.repeatResultRecycled != nil {
							lastResultRecycled = a.repeatResultRecycled(signature)
						} else {
							lastResultRecycled = a.lastResultForSignatureIsRecycled(signature)
						}
					}
					if blocked && repeatGuard.exhausted && !lastResultRecycled {
						// In downgraded mode the guard only keeps intercepting the
						// pathological subclass (re-issuing a call whose previous
						// result is a known recycled placeholder — which can never
						// make progress). All other repeats, e.g. legitimate
						// re-reads after compaction, are allowed through.
						blocked = false
					}
					reminderConfigured := blocked && strings.TrimSpace(a.loopGuardUserMsg) != ""
					legacyAction := interventionActionProceed
					if blocked {
						legacyAction = interventionActionSuppressTool
					}
					legacyDecision := interventionDecision{
						detection:      interventionDetection{kind: interventionKindRepeatedSignature, active: detected},
						action:         legacyAction,
						queueReminder:  blocked && (repeatGuard.exhausted || reminderConfigured),
						downgradeGuard: blocked && !repeatGuard.exhausted && a.loopGuardStrikeMax > 0 && loopGuardStrikes+1 >= a.loopGuardStrikeMax,
					}
					observation := repeatedSignatureObservation{
						count:              seen,
						threshold:          repeatGuard.threshold,
						exhausted:          repeatGuard.exhausted,
						lastResultRecycled: lastResultRecycled,
						reminderConfigured: reminderConfigured,
						nextStrike:         loopGuardStrikes + 1,
						strikeLimit:        a.loopGuardStrikeMax,
					}
					shadowDecision := shadowRepeatedSignatureIntervention(observation)
					if a.repeatInterventionShadowEvaluator != nil {
						shadowDecision = a.repeatInterventionShadowEvaluator(observation)
					}
					a.observeRepeatedSignatureIntervention(legacyDecision, shadowDecision)
					if blocked {
						loopGuardStrikes++
						a.appendLoopGuardSkippedToolResult(tc, resolvedName)
						activeToolBlock.markTerminal(idx, toolCallAccepted, "loop_guard")
						reminder := strings.TrimSpace(a.loopGuardUserMsg)
						if repeatGuard.exhausted {
							// The default reminder can mislead here ("reuse prior
							// results"), because the prior result was recycled and
							// cannot be reused. Inject a diagnostic that matches
							// reality and is aligned with the placeholder wording.
							reminder = messageorigin.RecycledToolResultRecoveryText
						}
						if reminder != "" {
							// Buffered, not appended: this is a user-role
							// message and the assistant tool-call block is not
							// closed yet. Appending here would interleave user
							// text between two tool results of the same block
							// and permanently invalidate the history.
							pendingBlockMessages = append(pendingBlockMessages, messageorigin.NewInternalUserMessage(messageorigin.KindLoopGuard, reminder))
							a.emitEvent(out, HiddenUserMessageEvent{Content: reminder})
						}
						a.emitEvent(out, WarnEvent{
							Message: fmt.Sprintf(
								"detected repeated tool-call signature (%d/%d within %d calls); strike %d/%d; injected loop-break reminder and skipping execution",
								seen,
								repeatGuard.threshold,
								repeatGuard.window,
								loopGuardStrikes,
								a.loopGuardStrikeMax,
							),
							Kind: "loop_guard",
						})
						repeatGuard.reset()
						if !repeatGuard.exhausted && a.loopGuardStrikeMax > 0 && loopGuardStrikes >= a.loopGuardStrikeMax {
							// Repeat protection budget is spent. Rather than
							// aborting the run (which kills legitimate work — e.g. a
							// long research turn that re-reads a file after context
							// compaction evicted the earlier result) or fully
							// disabling the guard (which nurtured 60-minute spins in
							// the self-bootstrap case), downgrade it: normal repeats
							// are allowed through, but the pathological subclass
							// (re-reading a recycled placeholder) is still
							// intercepted so the run stays bounded.
							a.emitEvent(out, WarnEvent{
								Message: fmt.Sprintf(
									"repeated tool-call loop protection budget spent after %d strike(s); downgrading guard: normal repeats allowed, recycled-placeholder re-reads still blocked",
									loopGuardStrikes,
								),
								Kind: "loop_guard",
							})
							repeatGuard.exhausted = true
						}
						a.emitEvent(out, StepStartEvent{StepID: tc.ID, Title: resolvedName, StepNumber: step})
						a.emitEvent(out, ToolCallEvent{
							Tool: resolvedName, Args: norm.Display, ArgsJSON: norm.Normalized,
							ArgsMeta: norm.Meta, ToolCallID: tc.ID, DisplayName: resolvedName,
						})
						guardResult := ToolResultEvent{
							Tool: resolvedName, ToolCallID: tc.ID, IsError: true,
							Result:   "[ERROR] Tool call skipped by loop guard - Repeated identical tool call blocked before execution.",
							Metadata: map[string]any{"loop_guard_suppressed": true},
						}
						a.emitToolResultWithAccounting(out, guardResult, guardResult.Result, 0)
						a.emitEvent(out, StepCompleteEvent{StepID: tc.ID, Status: "error"})
						continue
					}
				}
				a.emitEvent(out, StepStartEvent{StepID: tc.ID, Title: resolvedName, StepNumber: step})
				argsMap := norm.Display
				if argsMap == nil {
					argsMap = map[string]any{"__raw": execArgs}
				}
				a.emitEvent(out, ToolCallEvent{
					Tool:        resolvedName,
					Args:        argsMap,
					ArgsJSON:    norm.Normalized,
					ArgsMeta:    norm.Meta,
					ToolCallID:  tc.ID,
					DisplayName: resolvedName,
				})
				if evidenceTool {
					decision := progressLedger.preflight(evidenceReq, a.compactionGeneration.Load())
					if decision.suppress {
						content := llm.TextContent(decision.content)
						a.mu.Lock()
						a.messages = append(a.messages, llm.Message{
							Role: llm.RoleTool, ToolCallID: tc.ID, ToolName: resolvedName,
							Content: content, IsError: false, Ephemeral: tool.EphemeralKeep > 0,
						})
						a.mu.Unlock()
						activeToolBlock.markTerminal(idx, toolCallAccepted, "evidence_suppressed")
						suppressedResult := ToolResultEvent{
							Tool: resolvedName, Result: content.PlainText(), ToolCallID: tc.ID,
							IsError: false, Metadata: decision.metadata,
						}
						a.emitToolResultWithAccounting(out, suppressedResult, suppressedResult.Result, 0)
						a.emitEvent(out, StepCompleteEvent{StepID: tc.ID, Status: "completed"})
						if decision.recovery {
							reminder := evidenceRecoveryMessage(evidenceReq)
							// Deferred until the tool-call block is closed; see
							// the loop-guard reminder above.
							pendingBlockMessages = append(pendingBlockMessages, messageorigin.NewInternalUserMessage(messageorigin.KindEvidenceRecovery, reminder))
							a.emitEvent(out, HiddenUserMessageEvent{Content: reminder})
							a.emitEvent(out, WarnEvent{
								Kind:     "no_progress_recovery",
								Metadata: decision.metadata,
								Message: fmt.Sprintf(
									"suppressed repeated %s evidence call after %d execution(s); target=%s fingerprint=%s action=change_target_range_or_action",
									evidenceReq.family,
									decision.metadata["evidence_executed"],
									evidenceReq.target,
									evidenceReq.fingerprint,
								),
							})
						}
						continue
					}
				}

				start := time.Now()
				ctxToolBase := tools.WithToolCallID(ctx, tc.ID)
				ctxToolBase = tools.WithToolResultMetadata(ctxToolBase)
				ctxTool, finishToolStage := a.beginSteeringInterruptibleStage(ctxToolBase)
				activeToolBlock.markRunning(idx)
				content, toolErr := a.executeToolSafely(ctxTool, tool, execArgs)
				stageInterruptedForSteering := finishToolStage()
				rootCancelErr := ctx.Err()
				activeToolBlock.markAttemptReturned(idx, rootCancelErr != nil)
				if rootCancelErr != nil {
					// Root cancellation outranks a tool's ordinary or task-complete
					// return. Keep a contiguous result for this call, then terminate.
					toolErr = rootCancelErr
					content = llm.TextContent("Tool execution canceled before turn continuation: " + rootCancelErr.Error())
				}
				isError := toolErr != nil
				if unknownToolFallback {
					isError = true
				}
				status := "completed"
				if isError {
					status = "error"
				}
				meta := tools.TakeToolResultMetadataSnapshot(ctxTool)
				if unknownToolFallback {
					if meta == nil {
						meta = map[string]any{}
					}
					meta["error_kind"] = "tool_not_found"
					meta["tool"] = originalName
				}

				// If tool is configured ephemeral, mark tool message accordingly.
				ephemeral := tool.EphemeralKeep > 0

				// Tool completion special-case.
				var tce *tools.TaskCompleteError
				if rootCancelErr == nil && errors.As(toolErr, &tce) {
					isError = false
					status = "completed"
					content = llm.TextContent("Task completed: " + tce.Message)
					originalResult := content.PlainText()
					content, meta = a.applyToolResultTruncation(ctx, content, meta, resolvedName, tc.ID)
					// append tool message and finish
					a.mu.Lock()
					a.messages = append(a.messages, llm.Message{Role: llm.RoleTool, ToolCallID: tc.ID, ToolName: resolvedName, Content: content, IsError: false, Ephemeral: ephemeral})
					a.mu.Unlock()
					activeToolBlock.markTerminal(idx, toolCallRunning, "task_complete")
					a.emitToolResultWithAccounting(out, ToolResultEvent{Tool: resolvedName, Result: content.PlainText(), ToolCallID: tc.ID, IsError: false, Metadata: meta}, originalResult, time.Since(start))
					a.emitEvent(out, StepCompleteEvent{StepID: tc.ID, Status: status, DurationMS: time.Since(start).Milliseconds()})
					if err := ctx.Err(); err != nil {
						a.appendCancellationSkippedToolResults(comp.ToolCalls[idx+1:])
						activeToolBlock.markTerminalRange(idx+1, toolCallAccepted, "root_cancel_after_task_complete")
						a.appendMessages(pendingBlockMessages)
						pendingBlockMessages = nil
						emitSDKErr(a.errEvent(err))
						return
					}
					// The turn ends here, but the assistant tool-call block must
					// still be closed: a parallel `done` that is not the last
					// call would otherwise leave tool_use blocks with no result,
					// making the *next* turn's first provider request invalid.
					a.appendTurnEndSkippedToolResults(comp.ToolCalls[idx+1:])
					activeToolBlock.markTerminalRange(idx+1, toolCallAccepted, "task_complete_tail")
					a.appendMessages(pendingBlockMessages)
					pendingBlockMessages = nil
					if a.hasCompactor {
						_ = a.checkAndCompact(ctx, comp, out, additionalSinceCompletion())
					}
					finalContent := strings.TrimSpace(tce.Message)
					finalResponseID := responseID
					if preserved := strings.TrimSpace(pendingRequireDoneFinalText); preserved != "" {
						finalContent = preserved
						if preservedResponseID := strings.TrimSpace(pendingRequireDoneFinalResponseID); preservedResponseID != "" {
							finalResponseID = preservedResponseID
						}
					}
					requireDoneRecoveryDisableThinkingActive = false
					finishToolBlock()
					emitFinal(finalContent, finalResponseID)
					return
				}

				if evidenceTool {
					evidenceMeta := progressLedger.observe(evidenceReq, content.PlainText(), isError)
					meta = mergeToolResultMetadata(meta, evidenceMeta)
				}
				progressLedger.invalidateAfter(resolvedName, isError)
				originalResult := content.PlainText()
				content, meta = a.applyToolResultTruncation(ctx, content, meta, resolvedName, tc.ID)

				// append tool message
				a.mu.Lock()
				a.messages = append(a.messages, llm.Message{Role: llm.RoleTool, ToolCallID: tc.ID, ToolName: resolvedName, Content: content, IsError: isError, Ephemeral: ephemeral})
				a.mu.Unlock()
				activeToolBlock.markTerminal(idx, toolCallRunning, "handler_return")

				a.emitToolResultWithAccounting(out, ToolResultEvent{Tool: resolvedName, Result: content.PlainText(), ToolCallID: tc.ID, IsError: isError, Metadata: meta}, originalResult, time.Since(start))
				a.emitEvent(out, StepCompleteEvent{StepID: tc.ID, Status: status, DurationMS: time.Since(start).Milliseconds()})
				if rootCancelErr == nil {
					rootCancelErr = ctx.Err()
				}
				if rootCancelErr != nil {
					a.appendCancellationSkippedToolResults(comp.ToolCalls[idx+1:])
					activeToolBlock.markTerminalRange(idx+1, toolCallAccepted, "root_cancel_after_handler")
					a.appendMessages(pendingBlockMessages)
					pendingBlockMessages = nil
					emitSDKErr(a.errEvent(rootCancelErr))
					return
				}
				if strings.EqualFold(strings.TrimSpace(resolvedName), "done") {
					requireDoneRecoveryDisableThinkingActive = false
				}

				// *** Boundary-aware steering: check for new user messages after each tool execution ***
				steeringMessages := a.collectSteering(steeringCh, out)
				if len(steeringMessages) > 0 || stageInterruptedForSteering {
					requireDoneReminders = 0
					forceRequireDoneToolChoice = false
					requireDoneRecoveryDisableThinkingActive = false
					pendingRequireDoneFinalText = ""
					pendingRequireDoneFinalResponseID = ""
					// Close the tool_result block for every remaining call
					// *before* the steering text enters history. A user message
					// between two tool results of the same assistant block makes
					// the whole conversation permanently unsendable.
					a.appendSteeringSkippedToolResults(comp.ToolCalls[idx+1:])
					activeToolBlock.markTerminalRange(idx+1, toolCallAccepted, "steering")
					pendingBlockMessages = append(pendingBlockMessages, steeringMessages...)
					break
				}
			}
			// The assistant tool-call block is now closed: every tool_use has a
			// tool_result. Only here may user-role messages enter history.
			finishToolBlock()
			a.appendMessages(pendingBlockMessages)
			pendingBlockMessages = nil
			if a.hasCompactor {
				if err := a.checkAndCompact(ctx, comp, out, additionalSinceCompletion()); err != nil {
					emitCompactionErr(a.errEvent(err))
					return
				}
			}
		}

		// Max iterations reached — emit both error and final events.
		// Unreachable when maxIterations < 0 (unlimited).
		msg := fmt.Sprintf("Max iterations reached (%d)", a.maxIterations)
		emitErr(ErrorEvent{Provider: a.llm.Provider(), Message: msg, Kind: "max_iterations"}, EventOriginSDKDriver)
		emitFinal(fmt.Sprintf("[Max iterations reached] %d", a.maxIterations), lastResponseID)
	}(runtimeRelease, runtimeAcquired)
	return out
}

func (a *Agent) newQueryID() string {
	if a != nil && a.queryIDGenerator != nil {
		if id := strings.TrimSpace(a.queryIDGenerator()); id != "" {
			return id
		}
	}
	return newDefaultQueryID()
}

func (a *Agent) applyToolResultTruncation(ctx context.Context, content llm.Content, meta map[string]any, toolName, toolCallID string) (llm.Content, map[string]any) {
	return a.applyToolResultBoundary(ctx, content, meta, toolName, toolCallID)
}

func truncateToolResultContent(content llm.Content, meta map[string]any, maxBytes int, warnf func(string, ...any)) (llm.Content, map[string]any, string) {
	if maxBytes <= 0 {
		return content, meta, ""
	}
	plain := content.PlainText()
	originalBytes := len(plain)
	if originalBytes <= maxBytes {
		return content, meta, ""
	}
	truncated := truncateStringWithSuffix(plain, maxBytes, toolResultTruncatedSuffix)
	truncatedBytes := len(truncated)
	if meta == nil {
		meta = map[string]any{}
	}
	dumpPath, err := writeToolResultDump(plain)
	if err != nil && warnf != nil {
		warnf("warning: failed to persist full tool result for truncation: %v", err)
	}
	meta["result_truncated"] = true
	meta["result_bytes"] = truncatedBytes
	meta["result_original_bytes"] = originalBytes
	meta["result_max_bytes"] = maxBytes
	meta["truncated"] = true
	meta["originalSize"] = originalBytes
	if dumpPath != "" {
		meta["result_output_path"] = dumpPath
		meta["outputPath"] = dumpPath
	}
	return llm.TextContent(truncated), meta, dumpPath
}

func writeToolResultDumpFile(fullOutput string) (string, error) {
	dir, err := resolveToolResultDumpDir()
	if err != nil {
		return "", err
	}
	f, err := os.CreateTemp(dir, toolResultDumpPattern)
	if err != nil {
		return "", err
	}
	path := f.Name()
	if err := f.Chmod(0o600); err != nil {
		_ = f.Close()
		_ = os.Remove(path)
		return "", err
	}
	if _, err := f.WriteString(fullOutput); err != nil {
		_ = f.Close()
		_ = os.Remove(path)
		return "", err
	}
	if err := f.Close(); err != nil {
		_ = os.Remove(path)
		return "", err
	}
	return path, nil
}

func truncateStringWithSuffix(s string, maxBytes int, suffix string) string {
	if maxBytes <= 0 {
		return ""
	}
	if len(s) <= maxBytes {
		return s
	}
	if len(suffix) >= maxBytes {
		return truncateUTF8PrefixByBytes(suffix, maxBytes)
	}
	prefix := truncateUTF8PrefixByBytes(s, maxBytes-len(suffix))
	return prefix + suffix
}

func truncateUTF8PrefixByBytes(s string, maxBytes int) string {
	if maxBytes <= 0 {
		return ""
	}
	if len(s) <= maxBytes {
		return s
	}
	if utf8.ValidString(s[:maxBytes]) {
		return s[:maxBytes]
	}
	for cut := maxBytes; cut > 0; cut-- {
		if utf8.ValidString(s[:cut]) {
			return s[:cut]
		}
	}
	return ""
}

type toolCallBuilder struct {
	id   string
	name strings.Builder
	args strings.Builder
}

type toolCallAccumulator struct {
	items []*toolCallBuilder
}

type structuredThinkingBlockBuilder struct {
	blockType string
	thinking  strings.Builder
	signature strings.Builder
	data      strings.Builder
}

type structuredThinkingBlockAccumulator struct {
	order []int
	items map[int]*structuredThinkingBlockBuilder
}

func (a *structuredThinkingBlockAccumulator) apply(d llm.StreamThinkingDeltaEvent) {
	blockType := strings.TrimSpace(d.BlockType)
	if blockType == "" && d.SignatureDelta == "" && d.Data == "" {
		// Legacy/provider-display-only thinking deltas remain in Completion.Thinking
		// but do not become replayable content blocks.
		return
	}
	if blockType == "" {
		blockType = "thinking"
	}
	if a.items == nil {
		a.items = map[int]*structuredThinkingBlockBuilder{}
	}
	builder, ok := a.items[d.Index]
	if !ok {
		builder = &structuredThinkingBlockBuilder{blockType: blockType}
		a.items[d.Index] = builder
		a.order = append(a.order, d.Index)
	} else if builder.blockType == "" {
		builder.blockType = blockType
	}
	if d.Delta != "" {
		builder.thinking.WriteString(d.Delta)
	}
	if d.SignatureDelta != "" {
		builder.signature.WriteString(d.SignatureDelta)
	}
	if d.Data != "" {
		builder.data.WriteString(d.Data)
	}
}

func (a *structuredThinkingBlockAccumulator) finalize() []llm.ContentBlock {
	if a == nil || len(a.order) == 0 {
		return nil
	}
	blocks := make([]llm.ContentBlock, 0, len(a.order))
	for _, index := range a.order {
		builder := a.items[index]
		if builder == nil {
			continue
		}
		switch strings.TrimSpace(builder.blockType) {
		case "thinking":
			thinking := builder.thinking.String()
			signature := builder.signature.String()
			if thinking == "" && signature == "" {
				continue
			}
			blocks = append(blocks, llm.ContentBlock{Type: "thinking", Thinking: thinking, Signature: signature})
		case "redacted_thinking":
			data := builder.data.String()
			if data == "" {
				continue
			}
			blocks = append(blocks, llm.ContentBlock{Type: "redacted_thinking", Data: data})
		}
	}
	return blocks
}

func (a *toolCallAccumulator) ensure(index int) *toolCallBuilder {
	if index < 0 {
		index = 0
	}
	for len(a.items) <= index {
		a.items = append(a.items, &toolCallBuilder{})
	}
	return a.items[index]
}

func (a *toolCallAccumulator) apply(d llm.StreamToolCallDeltaEvent) {
	it := a.ensure(d.Index)
	if id := strings.TrimSpace(d.ID); id != "" && strings.TrimSpace(it.id) == "" {
		it.id = id
	}
	if d.NameDelta != "" {
		it.name.WriteString(d.NameDelta)
	}
	if d.ArgumentsDelta != "" {
		it.args.WriteString(d.ArgumentsDelta)
	}
}

const syntheticToolCallIDPrefix = "call_"

func ensureSyntheticToolCallIDs(toolCalls []llm.ToolCall) []llm.ToolCall {
	if len(toolCalls) == 0 {
		return toolCalls
	}
	out := append([]llm.ToolCall(nil), toolCalls...)
	used := make(map[string]struct{}, len(out))
	for _, tc := range out {
		if id := strings.TrimSpace(tc.ID); id != "" {
			used[id] = struct{}{}
		}
	}
	nextIndex := 0
	for i := range out {
		if id := strings.TrimSpace(out[i].ID); id != "" {
			out[i].ID = id
			continue
		}
		for {
			candidate := fmt.Sprintf("%s%d", syntheticToolCallIDPrefix, nextIndex)
			nextIndex++
			if _, exists := used[candidate]; exists {
				continue
			}
			out[i].ID = candidate
			used[candidate] = struct{}{}
			break
		}
	}
	return out
}

func duplicateToolCallIDPositions(toolCalls []llm.ToolCall) (first, second int, duplicate bool) {
	seen := make(map[string]int, len(toolCalls))
	for i, toolCall := range toolCalls {
		id := strings.TrimSpace(toolCall.ID)
		if previous, exists := seen[id]; exists {
			return previous, i, true
		}
		seen[id] = i
	}
	return 0, 0, false
}

func (a *toolCallAccumulator) finalize() []llm.ToolCall {
	out := []llm.ToolCall{}
	for _, it := range a.items {
		name := strings.TrimSpace(it.name.String())
		args := strings.TrimSpace(it.args.String())
		if name == "" {
			continue
		}
		id := strings.TrimSpace(it.id)
		out = append(out, llm.ToolCall{ID: id, Type: "function", Function: llm.FunctionCall{Name: name, Arguments: args}})
	}
	return ensureSyntheticToolCallIDs(out)
}

type repeatedToolSignatureGuard struct {
	threshold int
	window    int
	recent    []string
	counts    map[string]int
	// exhausted downgrades the guard after the strike budget is spent: instead
	// of aborting or fully disabling protection, the guard stops blocking normal
	// repeats but keeps intercepting the pathological subclass (re-issuing a call
	// whose previous result is a known recycled placeholder, which can never make
	// progress). This keeps the run bounded without a hard iteration cap.
	exhausted bool
}

func newRepeatedToolSignatureGuard(threshold, window int) *repeatedToolSignatureGuard {
	if threshold <= 0 {
		return nil
	}
	if window <= 0 {
		window = defaultRepeatSigWindow
	}
	if window < threshold {
		window = threshold
	}
	return &repeatedToolSignatureGuard{
		threshold: threshold,
		window:    window,
		counts:    make(map[string]int, window),
	}
}

func (g *repeatedToolSignatureGuard) observe(signature string) (count int, triggered bool) {
	if g == nil {
		return 0, false
	}
	signature = strings.TrimSpace(signature)
	if signature == "" {
		return 0, false
	}
	g.recent = append(g.recent, signature)
	g.counts[signature]++
	if len(g.recent) > g.window {
		old := g.recent[0]
		g.recent = g.recent[1:]
		if g.counts[old] <= 1 {
			delete(g.counts, old)
		} else {
			g.counts[old]--
		}
	}
	count = g.counts[signature]
	return count, count >= g.threshold
}

func (g *repeatedToolSignatureGuard) reset() {
	if g == nil {
		return
	}
	g.recent = nil
	for key := range g.counts {
		delete(g.counts, key)
	}
}

func normalizeToolSignature(name string, normalizedArgs json.RawMessage, rawArgs string) string {
	name = strings.TrimSpace(name)
	if name == "" {
		name = "<tool>"
	}
	args := strings.TrimSpace(string(normalizedArgs))
	if args == "" {
		args = strings.TrimSpace(rawArgs)
	}
	if args == "" {
		return name
	}
	var decoded any
	if err := json.Unmarshal([]byte(args), &decoded); err == nil {
		if canon, err := json.Marshal(decoded); err == nil {
			args = string(canon)
		}
	}
	return name + "|" + args
}

func hasVisiblePartialCompletion(comp *llm.Completion) bool {
	if comp == nil {
		return false
	}
	if !comp.Content.IsEmpty() {
		return true
	}
	if strings.TrimSpace(comp.Thinking) != "" {
		return true
	}
	return len(comp.ToolCalls) > 0 || llm.HasProviderState(comp.Content)
}

type streamMetadataBuffer struct {
	events        []llm.StreamEvent
	usage         *llm.Usage
	responseID    string
	providerState []llm.ProviderState
}

func (b *streamMetadataBuffer) add(ev llm.StreamEvent) bool {
	switch e := ev.(type) {
	case llm.StreamUsageEvent:
		u := e.Usage
		b.usage = &u
		b.events = append(b.events, ev)
		return true
	case llm.StreamResponseEvent:
		if id := strings.TrimSpace(e.ResponseID); id != "" {
			b.responseID = id
		}
		b.events = append(b.events, ev)
		return true
	case llm.StreamProviderStateEvent:
		b.providerState = append(b.providerState, llm.CloneProviderState(e.State)...)
		b.events = append(b.events, ev)
		return true
	case llm.StreamDoneEvent:
		b.events = append(b.events, ev)
		return true
	default:
		return false
	}
}

func (b *streamMetadataBuffer) flush(process func(llm.StreamEvent) error) error {
	if b == nil || len(b.events) == 0 {
		return nil
	}
	events := b.events
	b.events = nil
	for _, ev := range events {
		if err := process(ev); err != nil {
			return err
		}
	}
	return nil
}

func isVisibleProviderStreamEvent(ev llm.StreamEvent) bool {
	switch e := ev.(type) {
	case llm.StreamTextDeltaEvent:
		return e.Delta != ""
	case llm.StreamThinkingDeltaEvent:
		return e.Delta != ""
	case llm.StreamToolCallDeltaEvent:
		return strings.TrimSpace(e.ID) != "" || e.NameDelta != "" || e.ArgumentsDelta != ""
	default:
		return false
	}
}

// invokeCompletion calls the provider using streaming when available.
// It returns the completion (possibly partial on error), whether text was streamed, and error.
// On streaming errors, the partial completion contains whatever text/tools were accumulated
// before the error occurred, allowing callers to preserve partial output in history.
func (a *Agent) invokeCompletion(ctx context.Context, req llm.InvokeRequest, out *eventOutput) (*llm.Completion, bool, error) {
	return a.invokeCompletionWithSteering(ctx, req, out, nil)
}

// invokeCompletionWithSteering extends invokeCompletion with real-time steering support.
// When steeringCh is non-nil, the stream can be interrupted mid-generation if the user
// sends a steering message. The function returns a SteeringInterruptError in that case,
// allowing the caller to immediately incorporate the steering message into the conversation.
func (a *Agent) invokeCompletionWithSteering(ctx context.Context, req llm.InvokeRequest, out *eventOutput, steeringCh <-chan SteeringMsg) (*llm.Completion, bool, error) {
	if a == nil {
		return nil, false, fmt.Errorf("agent: nil llm")
	}
	return a.invokeModelCompletionWithSteering(ctx, a.llm, req, out, steeringCh)
}

func (a *Agent) invokeModelCompletionWithSteering(ctx context.Context, model llm.ChatModel, req llm.InvokeRequest, out *eventOutput, steeringCh <-chan SteeringMsg) (*llm.Completion, bool, error) {
	if model == nil {
		return nil, false, fmt.Errorf("agent: nil llm")
	}
	var err error
	req, err = llm.CloneInvokeRequest(req)
	if err != nil {
		return nil, false, fmt.Errorf("agent: clone invoke request: %w", err)
	}
	if ctx == nil {
		ctx = context.Background()
	}
	// Last line of defense for the tool_use/tool_result pairing invariant. Any
	// loop-level defect that leaves a tool_use without its result (or an orphan
	// result) would otherwise become an unrecoverable provider 400 that replays
	// on every later turn, because the malformed history stays in place. Repair
	// the outgoing copy only — history keeps the original messages — and warn
	// loudly so the underlying defect stays visible.
	if repaired, changed, unexpected := repairToolCallPairsDetailed(req.Messages); changed {
		if unexpected {
			// Only a repair that is not an in-flight tool-call continuation
			// indicates a genuinely malformed history; warning on the healthy
			// continuation boundary would make this signal useless.
			a.warnf("warning: repaired tool_use/tool_result pairing in the outgoing request; the conversation history had an unpaired tool-call block")
			a.emitEvent(out, WarnEvent{
				Kind:    "tool_pairing_repaired",
				Message: "conversation history contained an unpaired tool_use/tool_result block; the outgoing request was repaired before sending",
			})
		}
		req.Messages = repaired
	}
	if err := ctx.Err(); err != nil {
		return nil, false, err
	}
	if sm, ok := model.(llm.StreamingChatModel); ok {
		invokeCtx, finishStage := a.beginSteeringInterruptibleStage(ctx)
		defer finishStage()
		if err := invokeCtx.Err(); err != nil {
			return nil, false, err
		}
		finishProviderStage := func(comp *llm.Completion, streamedText bool, fallbackErr error) (*llm.Completion, bool, error) {
			stageInterruptedForSteering := finishStage()
			if ctx.Err() == nil && stageInterruptedForSteering {
				if msg, ok := takeNextSteering(steeringCh); ok {
					return comp, streamedText, &llm.SteeringInterruptError{Message: msg.Content}
				}
				return comp, streamedText, &llm.SteeringInterruptError{}
			}
			if fallbackErr == nil && ctx.Err() != nil {
				fallbackErr = ctx.Err()
			}
			return comp, streamedText, fallbackErr
		}
		ch, err := sm.InvokeStream(invokeCtx, req)
		if err != nil {
			return finishProviderStage(nil, false, err)
		}
		var text strings.Builder
		var thinking strings.Builder
		thinkingBlocks := &structuredThinkingBlockAccumulator{}
		acc := &toolCallAccumulator{}
		var usage *llm.Usage
		var providerState []llm.ProviderState
		stopReason := ""
		responseID := ""
		sawDone := false
		streamedText := false
		emittedVisible := false
		metadata := &streamMetadataBuffer{}
		partialCompletion := func() (*llm.Completion, error) {
			textContent := text.String()
			structuredThinking := thinkingBlocks.finalize()
			content := llm.TextContent(textContent)
			if len(structuredThinking) > 0 {
				blocks := append([]llm.ContentBlock(nil), structuredThinking...)
				if textContent != "" {
					blocks = append(blocks, llm.ContentBlock{Type: "text", Text: textContent})
				}
				content = llm.Content{Blocks: blocks}
			}
			thinkingText := strings.TrimSpace(thinking.String())
			toolCalls := acc.finalize()
			visible := !content.IsEmpty() || thinkingText != "" || len(toolCalls) > 0 || len(providerState) > 0
			completionUsage := usage
			if completionUsage == nil && !visible {
				completionUsage = metadata.usage
			}
			completionResponseID := responseID
			if strings.TrimSpace(completionResponseID) == "" && !visible {
				completionResponseID = metadata.responseID
			}
			completionProviderState := providerState
			if len(completionProviderState) == 0 && !visible {
				completionProviderState = metadata.providerState
			}
			content, err := llm.WithProviderState(content, completionProviderState)
			completion := &llm.Completion{
				Content:    content,
				Thinking:   thinkingText,
				ToolCalls:  toolCalls,
				Usage:      completionUsage,
				StopReason: stopReason,
				ResponseID: completionResponseID,
			}
			if err != nil {
				return completion, fmt.Errorf("agent: invalid streamed provider state: %w", err)
			}
			return completion, nil
		}
		finishPartial := func(fallbackErr error) (*llm.Completion, bool, error) {
			comp, stateErr := partialCompletion()
			if stateErr != nil {
				fallbackErr = stateErr
			}
			return finishProviderStage(comp, streamedText, fallbackErr)
		}
		processStreamEvent := func(ev llm.StreamEvent) error {
			switch e := ev.(type) {
			case llm.StreamTextDeltaEvent:
				if strings.TrimSpace(e.Delta) != "" || e.Delta == "\n" {
					text.WriteString(e.Delta)
					streamedText = true
					a.emitEvent(out, TextDeltaEvent{Delta: e.Delta})
				} else {
					// preserve whitespace as-is
					text.WriteString(e.Delta)
					if e.Delta != "" {
						streamedText = true
						a.emitEvent(out, TextDeltaEvent{Delta: e.Delta})
					}
				}
			case llm.StreamThinkingDeltaEvent:
				thinkingBlocks.apply(e)
				if e.Delta != "" {
					thinking.WriteString(e.Delta)
					a.emitEvent(out, ThinkingDeltaEvent{Delta: e.Delta})
				}
			case llm.StreamToolCallDeltaEvent:
				acc.apply(e)
			case llm.StreamUsageEvent:
				u := e.Usage
				usage = &u
			case llm.StreamResponseEvent:
				if id := strings.TrimSpace(e.ResponseID); id != "" {
					responseID = id
				}
			case llm.StreamProviderStateEvent:
				candidate := append(llm.CloneProviderState(providerState), llm.CloneProviderState(e.State)...)
				if _, err := llm.WithProviderState(llm.Content{}, candidate); err != nil {
					return fmt.Errorf("agent: invalid streamed provider state: %w", err)
				}
				providerState = candidate
			case llm.StreamRetryEvent:
				msg := strings.TrimSpace(e.Message)
				if msg == "" {
					msg = "provider request was rate limited"
				}
				if e.Attempt > 0 && e.MaxRetries > 0 {
					msg = fmt.Sprintf("%s; retry %d/%d", msg, e.Attempt, e.MaxRetries)
				}
				if e.RetryAfter > 0 {
					msg = fmt.Sprintf("%s in %s", msg, e.RetryAfter.Round(time.Second))
				}
				a.emitEvent(out, WarnEvent{Kind: "rate_limit_retry", Message: msg})
			case llm.StreamErrorEvent:
				return e.AsError()
			case llm.StreamDoneEvent:
				stopReason = e.StopReason
				sawDone = true
			}
			return nil
		}
		idleTimeout := agentStreamIdleTimeout
		if a != nil && a.streamIdleTimeout > 0 {
			idleTimeout = a.streamIdleTimeout
		}
		var idleTimer *time.Timer
		var idleC <-chan time.Time
		if idleTimeout > 0 {
			idleTimer = time.NewTimer(idleTimeout)
			idleC = idleTimer.C
			defer stopTimerDrain(idleTimer)
		}
		resetIdleTimer := func() {
			if idleTimer == nil {
				return
			}
			stopTimerDrain(idleTimer)
			idleTimer.Reset(idleTimeout)
		}

		for {
			select {
			case ev, ok := <-ch:
				if !ok {
					if err := metadata.flush(processStreamEvent); err != nil {
						return finishPartial(err)
					}
					if !sawDone {
						return finishPartial(&llm.IncompleteStreamError{
							Provider: model.Provider(),
							Model:    model.Model(),
							Message:  "provider event channel closed before StreamDoneEvent",
						})
					}
					return finishPartial(nil)
				}
				resetIdleTimer()
				if !emittedVisible && metadata.add(ev) {
					continue
				}
				if isVisibleProviderStreamEvent(ev) {
					if err := metadata.flush(processStreamEvent); err != nil {
						return finishPartial(err)
					}
					emittedVisible = true
				}
				if err := processStreamEvent(ev); err != nil {
					return finishPartial(err)
				}

			case msg, ok := <-steeringCh:
				if !ok {
					steeringCh = nil
					continue
				}

				// Steering message received - interrupt the stream
				if strings.TrimSpace(msg.Content) != "" {
					// Return partial completion with special error.
					// Cancel and drain the provider stream so streaming goroutines do not remain
					// blocked writing into an abandoned channel after steering interrupts the turn.
					comp, stateErr := partialCompletion()
					finishStage()
					drainStreamAsync(ch)
					if stateErr != nil {
						return comp, streamedText, stateErr
					}
					return comp, streamedText, &llm.SteeringInterruptError{Message: msg.Content}
				}

			case <-invokeCtx.Done():
				comp, streamedText, err := finishPartial(invokeCtx.Err())
				drainStreamAsync(ch)
				return comp, streamedText, err

			case <-idleC:
				comp, streamedText, err := finishPartial(&llm.StreamIdleTimeoutError{Duration: idleTimeout})
				drainStreamAsync(ch)
				return comp, streamedText, err
			}
		}
	}
	invokeCtx, finishStage := a.beginSteeringInterruptibleStage(ctx)
	if err := invokeCtx.Err(); err != nil {
		finishStage()
		return nil, false, err
	}
	comp, err := model.Invoke(invokeCtx, req)
	stageInterruptedForSteering := finishStage()
	if ctx.Err() == nil && stageInterruptedForSteering {
		if msg, ok := takeNextSteering(steeringCh); ok {
			return comp, false, &llm.SteeringInterruptError{Message: msg.Content}
		}
		return comp, false, &llm.SteeringInterruptError{}
	}
	if err == nil && ctx.Err() != nil {
		err = ctx.Err()
	}
	return comp, false, err
}

func (a *Agent) invokeCompletionWithRetry(ctx context.Context, req llm.InvokeRequest, out *eventOutput) (*llm.Completion, bool, error) {
	return a.invokeCompletionWithRetryAndSteering(ctx, req, out, nil)
}

// invokeCompletionWithRetryAndSteering extends invokeCompletionWithRetry with steering support.
// When a steering interrupt occurs, it immediately returns the partial completion along with
// the SteeringInterruptError, allowing the agent loop to process the steering message.
func (a *Agent) invokeCompletionWithRetryAndSteering(ctx context.Context, req llm.InvokeRequest, out *eventOutput, steeringCh <-chan SteeringMsg) (*llm.Completion, bool, error) {
	if a == nil {
		return nil, false, fmt.Errorf("agent: nil llm")
	}
	return a.invokeModelCompletionWithRetryAndSteering(ctx, a.llm, req, out, steeringCh)
}

func (a *Agent) invokeModelCompletionWithRetryAndSteering(ctx context.Context, model llm.ChatModel, req llm.InvokeRequest, out *eventOutput, steeringCh <-chan SteeringMsg) (*llm.Completion, bool, error) {
	maxAttempts := defaultInvokeRetryMax
	if a != nil && a.invokeRetryMax > 0 {
		maxAttempts = a.invokeRetryMax
	}
	if maxAttempts <= 1 {
		return a.invokeModelCompletionWithSteering(ctx, model, req, out, steeringCh)
	}

	var lastComp *llm.Completion
	var lastStreamed bool
	var lastErr error
	for attempt := 1; attempt <= maxAttempts; attempt++ {
		if err := ctx.Err(); err != nil {
			return lastComp, lastStreamed, err
		}
		comp, streamedText, err := a.invokeModelCompletionWithSteering(ctx, model, req, out, steeringCh)
		if err == nil {
			return comp, streamedText, nil
		}

		// Check for steering interrupt - don't retry, return immediately
		var steerErr *llm.SteeringInterruptError
		if errors.As(err, &steerErr) {
			return comp, streamedText, err
		}
		var idleErr *llm.StreamIdleTimeoutError
		if errors.As(err, &idleErr) {
			return comp, streamedText, err
		}

		lastComp = comp
		lastStreamed = streamedText
		lastErr = err
		if attempt >= maxAttempts {
			break
		}
		delay, retry := a.transientInvokeRetryDelay(err, comp, streamedText, attempt)
		if !retry {
			break
		}
		if delay > 0 {
			a.warnf("agent invoke transient failure (attempt %d/%d): %v; retrying in %s", attempt, maxAttempts, err, delay)
			t := time.NewTimer(delay)
			select {
			case <-ctx.Done():
				if !t.Stop() {
					<-t.C
				}
				return lastComp, lastStreamed, ctx.Err()
			case <-t.C:
			}
			continue
		}
		a.warnf("agent invoke transient failure (attempt %d/%d): %v; retrying", attempt, maxAttempts, err)
	}
	return lastComp, lastStreamed, lastErr
}

func (a *Agent) transientInvokeRetryDelay(err error, partial *llm.Completion, streamedText bool, attempt int) (time.Duration, bool) {
	if err == nil {
		return 0, false
	}
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return 0, false
	}
	if streamedText || hasVisiblePartialCompletion(partial) {
		return 0, false
	}

	transient := false
	retryAfter := time.Duration(0)

	var rl *llm.RateLimitError
	if errors.As(err, &rl) {
		transient = true
		retryAfter = rl.RetryAfter
	}
	if !transient {
		var pe *llm.ProviderError
		if errors.As(err, &pe) {
			transient = retryableProviderStatus(pe.StatusCode)
			if pe.RetryAfter > 0 {
				retryAfter = pe.RetryAfter
			}
		}
	}
	if !transient {
		if status, ok := statusCodeInText(strings.ToLower(strings.TrimSpace(err.Error()))); ok && retryableProviderStatus(status) {
			transient = true
		}
	}
	if !transient {
		kind := classifyGenericErrorKind(err)
		switch kind {
		case "network", "timeout", "rate_limit":
			transient = true
		case "provider":
			if status, ok := statusCodeInText(strings.ToLower(strings.TrimSpace(err.Error()))); !ok || retryableProviderStatus(status) {
				transient = true
			}
		}
	}
	if !transient {
		return 0, false
	}

	if retryAfter > 0 {
		if retryAfter > maxInvokeRetryDelay {
			return maxInvokeRetryDelay, true
		}
		return retryAfter, true
	}
	if a == nil || a.invokeRetryBackoff <= 0 {
		return 0, true
	}
	if attempt < 1 {
		attempt = 1
	}
	delay := a.invokeRetryBackoff
	for i := 1; i < attempt; i++ {
		delay *= 2
		if delay >= maxInvokeRetryDelay {
			return maxInvokeRetryDelay, true
		}
	}
	if delay > maxInvokeRetryDelay {
		delay = maxInvokeRetryDelay
	}
	return delay, true
}

func drainStreamAsync(ch <-chan llm.StreamEvent) {
	if ch == nil {
		return
	}
	go func() {
		for range ch {
		}
	}()
}

func stopTimerDrain(t *time.Timer) {
	if t == nil {
		return
	}
	if !t.Stop() {
		select {
		case <-t.C:
		default:
		}
	}
}

func (a *Agent) appendLoopGuardSkippedToolResult(call llm.ToolCall, currentResolvedName string) {
	id := strings.TrimSpace(call.ID)
	if id == "" {
		return
	}
	content := llm.TextContent("[ERROR] Tool call skipped by loop guard - Repeated identical tool call blocked before execution. Reuse previous results, change arguments, or call done if the task is complete.")
	name := strings.TrimSpace(currentResolvedName)
	if name == "" {
		name = strings.TrimSpace(call.Function.Name)
	}
	if name == "" {
		name = "unknown"
	}
	a.mu.Lock()
	a.messages = append(a.messages, llm.NewToolMessage(id, name, content, true))
	a.mu.Unlock()
}

func (a *Agent) appendSteeringSkippedToolResults(calls []llm.ToolCall) {
	if a == nil || len(calls) == 0 {
		return
	}
	a.appendMessages(skippedToolResults(calls, toolSkippedBySteeringText))
}

// appendTurnEndSkippedToolResults closes a tool-call block whose remaining calls
// were never executed because the turn completed early (e.g. a parallel `done`).
func (a *Agent) appendTurnEndSkippedToolResults(calls []llm.ToolCall) {
	if a == nil || len(calls) == 0 {
		return
	}
	a.appendMessages(skippedToolResults(calls, toolSkippedByTurnEndText))
}

// appendCancellationSkippedToolResults closes a tool-call block whose
// remaining calls were never executed because the root turn was canceled.
func (a *Agent) appendCancellationSkippedToolResults(calls []llm.ToolCall) {
	if a == nil || len(calls) == 0 {
		return
	}
	a.appendMessages(skippedToolResults(calls, toolSkippedByCancellationText))
}

const (
	toolSkippedBySteeringText     = "[INFO] Tool call skipped because user steering changed the current direction before execution."
	toolSkippedByTurnEndText      = "[INFO] Tool call skipped because the task was completed before this call ran."
	toolSkippedByCancellationText = "[ERROR] Tool call skipped because the active turn was canceled before this call ran."
)

// skippedToolResults builds the synthetic tool results that close an assistant
// tool-call block whose remaining calls were never executed.
func skippedToolResults(calls []llm.ToolCall, text string) []llm.Message {
	if len(calls) == 0 {
		return nil
	}
	messages := make([]llm.Message, 0, len(calls))
	for _, call := range calls {
		id := strings.TrimSpace(call.ID)
		if id == "" {
			continue
		}
		name := strings.TrimSpace(call.Function.Name)
		if name == "" {
			name = "unknown"
		}
		messages = append(messages, llm.NewToolMessage(id, name, llm.TextContent(text), true))
	}
	return messages
}

// repairToolCallPairs enforces the provider-level tool_use/tool_result pairing
// invariant on an outgoing history: an assistant tool-call block is kept only
// when the contiguous tool messages that follow it contain exactly one result
// per unique, non-empty call ID; orphaned tool results are dropped. It is
// idempotent and reports whether anything had to change, so callers can log the
// underlying defect instead of silently masking it.
func repairToolCallPairs(messages []llm.Message) ([]llm.Message, bool) {
	out, changed, _ := repairToolCallPairsDetailed(messages)
	return out, changed
}

// repairToolCallPairsDetailed additionally reports whether any repair went
// beyond an in-flight tool-call continuation. A truncated tool_use block that is
// followed by the framework's continuation reminder is a designed, transient
// state — its arguments are still being accumulated — so repairing it says
// nothing about history health. Every other repair does indicate a malformed
// history, which is what callers should warn about.
func repairToolCallPairsDetailed(messages []llm.Message) (out []llm.Message, changed bool, unexpected bool) {
	if len(messages) == 0 {
		return messages, false, false
	}
	out = make([]llm.Message, 0, len(messages))
	for i := 0; i < len(messages); i++ {
		m := messages[i]
		if m.Role == llm.RoleTool {
			// A tool result reached here without a preceding assistant block
			// that claimed it.
			changed = true
			unexpected = true
			continue
		}
		if m.Role != llm.RoleAssistant || len(m.ToolCalls) == 0 {
			out = append(out, m)
			continue
		}
		expected, validCalls := toolCallIDSet(m.ToolCalls)
		j := i + 1
		for j < len(messages) && messages[j].Role == llm.RoleTool {
			j++
		}
		if validCalls && toolResultBlockIsComplete(messages[i+1:j], expected) {
			out = append(out, m)
			out = append(out, messages[i+1:j]...)
		} else {
			pendingContinuation := isPendingContinuationBoundary(messages, i, j)
			m.ToolCalls = nil
			if !pendingContinuation {
				m.Content = llm.WithoutProviderState(m.Content)
			}
			out = append(out, m)
			changed = true
			if !pendingContinuation {
				unexpected = true
			}
		}
		i = j - 1
	}
	return out, changed, unexpected
}

// isPendingContinuationBoundary reports whether the assistant tool-call block at
// index is an in-flight truncated continuation: no tool results were produced
// for it yet and the next message is the framework's continuation reminder.
func isPendingContinuationBoundary(messages []llm.Message, index, resultsEnd int) bool {
	if resultsEnd != index+1 || resultsEnd >= len(messages) {
		return false
	}
	next := messages[resultsEnd]
	if next.Role != llm.RoleUser {
		return false
	}
	switch strings.TrimSpace(next.Name) {
	case messageorigin.Name(messageorigin.KindToolCallContinuation),
		messageorigin.Name(messageorigin.KindMaxTokensContinuation):
		return true
	default:
		return false
	}
}

// toolCallIDSet collects the tool call IDs of one assistant block. It reports
// false when an ID is empty or duplicated, because such a block can never be
// paired unambiguously.
func toolCallIDSet(calls []llm.ToolCall) (map[string]bool, bool) {
	ids := make(map[string]bool, len(calls))
	for _, call := range calls {
		id := strings.TrimSpace(call.ID)
		if id == "" {
			return nil, false
		}
		if _, ok := ids[id]; ok {
			return nil, false
		}
		ids[id] = false
	}
	return ids, len(ids) > 0
}

// toolResultBlockIsComplete reports whether results covers every expected call
// ID exactly once and contains nothing else.
func toolResultBlockIsComplete(results []llm.Message, expected map[string]bool) bool {
	if len(expected) == 0 {
		return false
	}
	for _, m := range results {
		id := strings.TrimSpace(m.ToolCallID)
		seen, ok := expected[id]
		if !ok || seen {
			return false
		}
		expected[id] = true
	}
	for _, seen := range expected {
		if !seen {
			return false
		}
	}
	return true
}

func mergeToolResultMetadata(base, extra map[string]any) map[string]any {
	if len(extra) == 0 {
		return base
	}
	if base == nil {
		base = make(map[string]any, len(extra))
	}
	for key, value := range extra {
		base[key] = value
	}
	return base
}

func (a *Agent) executeToolSafely(ctx context.Context, tool tools.Tool, raw string) (content llm.Content, err error) {
	defer func() {
		if recovered := recover(); recovered != nil {
			panicMsg := fmt.Sprintf("tool %q panicked: %v", tool.Name, recovered)
			a.warnf("error: recovered panic from tool %q: %v", tool.Name, recovered)
			tools.UpsertToolResultMetadata(ctx, map[string]any{
				"panic":         true,
				"panic_message": panicMsg,
				"tool":          tool.Name,
			})
			content = llm.TextContent("Error: " + panicMsg)
			err = fmt.Errorf("%s", panicMsg)
		}
	}()
	return tool.Execute(ctx, raw, a.deps)
}

// emitPartialUsage records the usage of a completion that did not reach the
// normal success path (terminal stream error, steering interrupt, idle-timeout
// recovery, context cancellation). The provider already billed the tokens it
// produced, so skipping this leaves the accounting journal under-counted.
func (a *Agent) emitPartialUsage(out *eventOutput, comp *llm.Completion) {
	if comp == nil || comp.Usage == nil {
		return
	}
	usage := llm.NormalizeUsage(comp.Usage)
	if usage == nil {
		return
	}
	a.emitUsageWithAccounting(out, *usage, strings.TrimSpace(comp.ResponseID))
}

func (a *Agent) emitUsageWithAccounting(out *eventOutput, usage llm.Usage, responseID string) {
	if !a.emitEvent(out, UsageEvent{Usage: usage, ResponseID: responseID}) {
		return
	}
	a.emitAccounting(out, AccountingEvent{
		Payload:         sdkaccounting.ProjectUsage(usage, a.accountingEstimator),
		CorrelationKind: "response",
		ResponseID:      strings.TrimSpace(responseID),
	})
}

func (a *Agent) emitToolResultWithAccounting(out *eventOutput, event ToolResultEvent, original string, duration time.Duration) {
	if !a.emitEvent(out, event) {
		return
	}
	a.emitAccounting(out, AccountingEvent{
		Payload: sdkaccounting.ProjectToolResult(sdkaccounting.ToolResultInput{
			Tool:     event.Tool,
			Original: original,
			Visible:  event.Result,
			IsError:  event.IsError,
			Metadata: event.Metadata,
		}, a.accountingEstimator),
		CorrelationKind: "tool_call",
		ToolCallID:      strings.TrimSpace(event.ToolCallID),
		DurationMS:      duration.Milliseconds(),
	})
}

func (a *Agent) emitCompactionWithAccounting(out *eventOutput, event CompactionEvent) {
	if !a.emitEvent(out, event) {
		return
	}
	a.emitAccounting(out, AccountingEvent{
		Payload:         sdkaccounting.ProjectCompaction(event.Result, a.accountingEstimator),
		CorrelationKind: "compaction",
	})
}

func (a *Agent) emitAccounting(out *eventOutput, event AccountingEvent) {
	event.Sequence = a.accountingSequence.Add(1)
	a.emitEvent(out, event)
}

func (a *Agent) emitEvent(out *eventOutput, ev Event) bool {
	if out == nil {
		return false
	}
	return a.emitEnvelope(out, ev, out.next(ev))
}

func (a *Agent) emitEventFrom(out *eventOutput, ev Event, origin EventOrigin) bool {
	if out == nil {
		return false
	}
	return a.emitEnvelope(out, ev, out.nextFrom(ev, origin))
}

func (a *Agent) emitEnvelope(out *eventOutput, ev Event, envelope EventEnvelope) bool {
	if out.trySend(envelope) {
		return true
	}

	timeout := defaultEventSendTimeout
	if a != nil && a.eventSendTimeout > 0 {
		timeout = a.eventSendTimeout
	}
	// The extended floor below is only worth paying while a consumer can still
	// read. Once the turn's context is cancelled the caller has abandoned the
	// stream, so waiting cannot preserve the record it was meant to preserve: the
	// event would be delivered into a channel nobody drains, or dropped anyway
	// after the full wait. Falling back to the configured budget keeps the
	// deliberate floor for live turns while a cancelled query stops paying
	// 250ms per critical event.
	state := a.turnBackpressureState(out)
	var turnDone <-chan struct{}
	if state != nil {
		turnDone = state.done
	}
	turnCancelled := false
	if turnDone != nil {
		select {
		case <-turnDone:
			turnCancelled = true
		default:
		}
	}
	floored := false
	if isTerminalAgentEvent(ev) {
		switch a.tryEnqueueTerminalEvent(out, envelope) {
		case terminalEnqueued:
			return true
		case terminalRejected:
			return false
		}
		// Terminal events are not charged against the per-turn floor budget:
		// there is at most one FinalResponseEvent/ErrorEvent per turn, so they
		// cannot multiply the floor the way per-step events can, and losing the
		// turn's outcome is the single worst drop.
		if !turnCancelled && timeout < criticalEventSendTimeoutFloor {
			a.warnFloorOverride(state, ev, timeout)
			timeout = criticalEventSendTimeoutFloor
		}
	} else if isCriticalAgentEvent(ev) {
		// These events are the only trace a consumer gets of a history or
		// ledger mutation the agent already performed. Dropping one on the
		// default 25ms budget leaves the UI, the JSONL audit log, and the
		// accounting journal describing a turn that did not happen, so they
		// get the same bounded floor as terminal events. The wait stays
		// bounded: an abandoned consumer must not hang the turn.
		//
		// The floor is charged against a per-turn budget because these kinds are
		// per-step (~7 critical events per tool call): paying it unconditionally
		// makes a tool-heavy turn against a stalled consumer cost the floor times
		// the event count (a 100-call turn is ~700 events, i.e. minutes of pure
		// waiting). Once the budget is spent the turn keeps the ordinary event
		// budget and drops are still counted exactly and reported through
		// FinalResponseEvent.DroppedCriticalEvents, so nothing is lost silently.
		if !turnCancelled && timeout < criticalEventSendTimeoutFloor &&
			a.criticalFloorBudgetAvailable(state) {
			a.warnFloorOverride(state, ev, timeout)
			timeout = criticalEventSendTimeoutFloor
			floored = true
		}
	}
	if timeout <= 0 {
		a.logDroppedEvent(ev, "channel_full")
		return false
	}

	if floored {
		start := time.Now()
		defer func() { state.chargeFloorSpent(time.Since(start)) }()
	}
	timer := time.NewTimer(timeout)
	defer func() {
		if !timer.Stop() {
			select {
			case <-timer.C:
			default:
			}
		}
	}()
	// A turn cancelled while this send is already waiting aborts the wait for the
	// same reason: nothing is going to read the channel any more.
	switch out.sendUntil(envelope, turnDone, timer.C) {
	case eventSent:
		return true
	case eventTurnCanceled:
		a.logDroppedEvent(ev, "turn_canceled")
		return false
	default:
		a.logDroppedEvent(ev, fmt.Sprintf("send_timeout_%s", timeout))
		return false
	}
}

// criticalEventFloorTurnBudget caps the total wall time one turn may spend inside
// the extended critical-event floor. It allows a consumer that stalls briefly
// (GC pause, a slow render, a blocking write to the audit log) to keep every
// consistency-critical event, while a consumer that has stopped reading
// altogether costs the turn a bounded delay instead of criticalEventSendTimeoutFloor
// per event for the rest of the turn.
const criticalEventFloorTurnBudget = 2 * time.Second

// criticalFloorBudgetAvailable reports whether the turn may still pay the
// extended critical-event floor. Channels that belong to no registered turn
// (direct library or unit-test calls) are not budgeted: there is no turn scope to
// charge, so they keep the unconditional floor.
func (a *Agent) criticalFloorBudgetAvailable(state *turnBackpressure) bool {
	if state == nil {
		return true
	}
	if state.floorSpentNanos.Load() < int64(criticalEventFloorTurnBudget) {
		return true
	}
	if state.floorBudgetWarned.CompareAndSwap(false, true) {
		a.warnf("warning: critical-event backpressure floor budget of %s is spent for this turn; consistency-critical events now use the ordinary %s send budget and any drop is still counted in DroppedCriticalEvents",
			criticalEventFloorTurnBudget, a.configuredEventSendTimeout())
	}
	return false
}

// configuredEventSendTimeout reports the ordinary per-event send budget, i.e. the
// budget critical events fall back to once the turn's floor budget is spent.
func (a *Agent) configuredEventSendTimeout() time.Duration {
	if a != nil && a.eventSendTimeout > 0 {
		return a.eventSendTimeout
	}
	return defaultEventSendTimeout
}

// warnFloorOverride reports, once per turn, that the critical-event floor is
// being used in place of the host's configured EventSendTimeout.
//
// The floor deliberately wins over the configured budget: a host tunes
// EventSendTimeout for high-rate text deltas, and applying that budget to the
// events that are a consumer's only record of a mutation the agent already
// performed is what ISS-129b exists to prevent. Silently multiplying a host's
// 1ms or 13ms setting by up to 250x is nonetheless a surprise worth surfacing,
// so the substitution is announced instead of being invisible. It stays a
// warning rather than a clamp because bounding the floor to a multiple of the
// configured value would hand a host that configures 1ms a 1ms budget for
// consistency-critical events, i.e. exactly the silent-loss behavior the floor
// prevents. The total cost per turn is already bounded by
// criticalEventFloorTurnBudget.
func (a *Agent) warnFloorOverride(state *turnBackpressure, ev Event, configured time.Duration) {
	if a == nil {
		return
	}
	// Channels outside any registered turn (direct library or unit-test calls)
	// have no turn scope to deduplicate against, so they are not warned about:
	// there is no turn whose log the notice would belong to.
	if state == nil || !state.floorOverrideWarned.CompareAndSwap(false, true) {
		return
	}
	a.warnf("warning: consistency-critical event %T waited up to the %s critical-event floor instead of the configured EventSendTimeout of %s; the floor keeps history/ledger mutations from being dropped silently and the turn's total floor wait is capped at %s",
		ev, criticalEventSendTimeoutFloor, configured, criticalEventFloorTurnBudget)
}

// chargeFloorSpent adds the wall time spent in one floored wait to the turn's
// budget.
func (t *turnBackpressure) chargeFloorSpent(d time.Duration) {
	if t == nil || d <= 0 {
		return
	}
	t.floorSpentNanos.Add(int64(d))
}

// registerTurnCancellation associates a turn's event channel with its context so
// emitEvent can stop paying the critical-event floor once the turn is abandoned,
// and gives the turn a fresh floor budget (see criticalEventFloorTurnBudget).
// The returned function must be called when the turn ends.
func (a *Agent) registerTurnCancellation(out *eventOutput, ctx context.Context) func() {
	if a == nil || out == nil || ctx == nil || ctx.Done() == nil {
		return func() {}
	}
	a.turnCancelMu.Lock()
	if a.turnCancelByOut == nil {
		a.turnCancelByOut = make(map[*eventOutput]*turnBackpressure, 1)
	}
	a.turnCancelByOut[out] = &turnBackpressure{done: ctx.Done()}
	a.turnCancelMu.Unlock()
	return func() {
		a.turnCancelMu.Lock()
		delete(a.turnCancelByOut, out)
		a.turnCancelMu.Unlock()
	}
}

// turnBackpressureState returns the backpressure state of the turn that owns out,
// or nil when the channel belongs to no registered turn (e.g. a direct unit-test
// call or a library caller driving the loop itself).
func (a *Agent) turnBackpressureState(out *eventOutput) *turnBackpressure {
	if a == nil || out == nil {
		return nil
	}
	a.turnCancelMu.Lock()
	defer a.turnCancelMu.Unlock()
	return a.turnCancelByOut[out]
}

// turnCancellation returns the Done channel of the turn that owns out, or nil
// when the channel belongs to no registered turn (e.g. a direct unit-test call).
// A nil channel blocks forever in a select, which is exactly the previous
// behavior for unregistered channels.
func (a *Agent) turnCancellation(out *eventOutput) <-chan struct{} {
	if state := a.turnBackpressureState(out); state != nil {
		return state.done
	}
	return nil
}

type terminalEnqueueOutcome uint8

const (
	terminalUnavailable terminalEnqueueOutcome = iota
	terminalEnqueued
	terminalRejected
)

func (a *Agent) tryEnqueueTerminalEvent(out *eventOutput, envelope EventEnvelope) terminalEnqueueOutcome {
	buffered, ok := out.tryReceive()
	if !ok {
		return terminalUnavailable
	}
	if isTerminalAgentEvent(buffered.Event) && terminalEventPriority(buffered.Event) > terminalEventPriority(envelope.Event) {
		out.sendAfterReceive(buffered)
		a.logDroppedEvent(envelope.Event, "terminal_priority_loss")
		return terminalRejected
	}
	a.logDroppedEvent(buffered.Event, "evicted_for_terminal")
	if final, ok := envelope.Event.(FinalResponseEvent); ok {
		final.DroppedEvents = dropsSince(a.eventDropCount.Load(), out.dropStart)
		final.DroppedCriticalEvents = dropsSince(a.criticalEventDropCount.Load(), out.criticalDropStart)
		envelope.Event = final
	}
	out.sendAfterReceive(envelope)
	return terminalEnqueued
}

func terminalEventPriority(ev Event) int {
	switch ev.(type) {
	case ErrorEvent:
		return 3
	case FinalResponseEvent:
		return 2
	default:
		return 0
	}
}

func isTerminalAgentEvent(ev Event) bool {
	switch ev.(type) {
	case FinalResponseEvent, ErrorEvent:
		return true
	default:
		return false
	}
}

// criticalEventSendTimeoutFloor is the minimum delivery budget for terminal and
// critical events. It is deliberately far above defaultEventSendTimeout (25ms,
// tuned for high-rate text deltas) yet still bounded, so a stalled consumer
// slows the turn instead of losing the record of a completed mutation.
const criticalEventSendTimeoutFloor = 250 * time.Millisecond

// isCriticalAgentEvent reports whether losing this event would leave the
// consumer's view inconsistent with the history/ledger the agent already
// mutated, rather than merely incomplete.
//
// The classification covers every event kind in events.go:
//
//   - ToolResultEvent: history holds the tool_result; a dropped event makes the
//     UI and the headless JSONL audit log disagree with history, and its
//     AccountingEvent projection is the only record of the truncation layers.
//   - StepStartEvent / StepCompleteEvent: unpaired step lifecycle leaves a
//     tool spinner active forever in the TUI.
//   - ToolCallEvent: history holds the assistant tool_use, and the runtime
//     keys step state off the call.
//   - HiddenUserMessageEvent: the agent appended an internal user message to
//     history; losing it hides an injected instruction entirely.
//   - CompactionEvent: history was replaced and a checkpoint/ledger entry was
//     written; losing it desynchronizes the context gauge and the checkpoint
//     lineage.
//   - AccountingEvent: the accounting journal is derived from this stream, so a
//     drop silently under-counts the ledger and breaks its sequence.
//   - UsageEvent: the billing/budget totals are accumulated from it.
//   - SteeringReceivedEvent: steering was merged into history mid-turn.
//   - AutoContinueEvent: a continuation prompt was appended to history.
//
// Text/thinking deltas and warnings are intentionally excluded: they are
// high-rate presentation content whose loss is reported in aggregate through
// FinalResponseEvent.DroppedEvents, and giving them the longer floor would
// let a slow consumer stall the whole turn.
func isCriticalAgentEvent(ev Event) bool {
	switch ev.(type) {
	case ToolResultEvent, ToolCallEvent, StepStartEvent, StepCompleteEvent,
		HiddenUserMessageEvent, CompactionEvent, AccountingEvent, UsageEvent,
		SteeringReceivedEvent, AutoContinueEvent:
		return true
	default:
		return false
	}
}

func (a *Agent) logDroppedEvent(ev Event, reason string) {
	logEvery := uint64(defaultEventDropLogEvery)
	if a != nil && a.eventDropLogEvery > 0 {
		logEvery = a.eventDropLogEvery
	}
	if logEvery == 0 {
		logEvery = 1
	}
	if a == nil {
		log.Printf("warning: dropping agent event %T due to backpressure (%s)", ev, reason)
		return
	}
	dropped := a.eventDropCount.Add(1)
	// Terminal and critical drops leave the consumer inconsistent with history,
	// so their exact count is kept separately and reported in full through
	// FinalResponseEvent.DroppedCriticalEvents. The log line is sampled on that
	// dedicated counter (always logging the first): sustained backpressure would
	// otherwise emit one warn line per drop, which is a log-volume regression
	// that buys nothing over the exact count already carried by the event.
	if isTerminalAgentEvent(ev) || isCriticalAgentEvent(ev) {
		critical := a.criticalEventDropCount.Add(1)
		if critical == 1 || critical%logEvery == 0 {
			a.warnf("warning: dropping consistency-critical agent event %T due to backpressure (%s); dropped_critical_total=%d dropped_total=%d", ev, reason, critical, dropped)
		}
		return
	}
	if dropped == 1 || dropped%logEvery == 0 {
		a.warnf("warning: dropping agent event %T due to backpressure (%s); dropped_total=%d", ev, reason, dropped)
	}
}

func (a *Agent) emitAutoContinue(out *eventOutput, reason string, responseID string) {
	a.emitEvent(out, AutoContinueEvent{Reason: reason, ResponseID: strings.TrimSpace(responseID)})
}

// lastResultForSignatureIsRecycled reports whether the most recent tool result
// whose originating call matches the given tool name + argument signature was
// recycled to the ephemeral placeholder. Re-issuing such a call cannot make
// progress (the model would just get the placeholder again), so this identifies
// the pathological subclass the loop guard keeps intercepting even after its
// strike budget is exhausted.
func (a *Agent) lastResultForSignatureIsRecycled(signature string) bool {
	signature = strings.TrimSpace(signature)
	if signature == "" {
		return false
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	for i := len(a.messages) - 1; i >= 0; i-- {
		m := a.messages[i]
		if m.Role != llm.RoleTool {
			continue
		}
		id := strings.TrimSpace(m.ToolCallID)
		if id == "" {
			continue
		}
		sig, ok := a.ephemeralSigByCall[id]
		if !ok {
			continue
		}
		if sig != signature {
			continue
		}
		// Most recent matching result found; report whether it was recycled.
		return m.Destroyed && m.Content.PlainText() == ephemeralReleasedPlaceholder
	}
	return false
}

func (a *Agent) destroyEphemeralMessages() {
	a.mu.Lock()
	defer a.mu.Unlock()

	if a.ephemeralByKey == nil {
		a.ephemeralByKey = make(map[string][]int)
	}
	if a.ephemeralSigByCall == nil {
		a.ephemeralSigByCall = make(map[string]string)
	}
	if a.ephemeralScanFrom < 0 || a.ephemeralScanFrom > len(a.messages) {
		a.resetEphemeralTrackingLocked()
	}

	for i := a.ephemeralScanFrom; i < len(a.messages); i++ {
		m := a.messages[i]
		// Record argument signatures for every tool call so that tool results
		// can be grouped by their concrete target rather than only by tool name.
		if m.Role == llm.RoleAssistant {
			for _, tc := range m.ToolCalls {
				id := strings.TrimSpace(tc.ID)
				if id == "" {
					continue
				}
				a.ephemeralSigByCall[id] = normalizeToolSignature(tc.Function.Name, nil, tc.Function.Arguments)
			}
			continue
		}
		if m.Role != llm.RoleTool {
			continue
		}
		if !m.Ephemeral || m.Destroyed {
			continue
		}
		toolName := strings.TrimSpace(m.ToolName)
		if toolName == "" {
			continue
		}
		a.ephemeralByKey[a.ephemeralGroupKeyLocked(m, toolName)] = append(a.ephemeralByKey[a.ephemeralGroupKeyLocked(m, toolName)], i)
	}
	a.ephemeralScanFrom = len(a.messages)

	for key, idxs := range a.ephemeralByKey {
		keep := 1
		if t, ok := a.toolMap[toolNameFromGroupKey(key)]; ok {
			if t.EphemeralKeep > 0 {
				keep = t.EphemeralKeep
			}
		}
		if keep <= 0 {
			keep = 1
		}
		for len(idxs) > keep {
			i := idxs[0]
			idxs = idxs[1:]
			if i < 0 || i >= len(a.messages) {
				continue
			}
			m := a.messages[i]
			if m.Role != llm.RoleTool || !m.Ephemeral || m.Destroyed {
				continue
			}
			m.Destroyed = true
			m.Content = llm.TextContent(ephemeralReleasedPlaceholder)
			a.messages[i] = m
		}
		if len(idxs) == 0 {
			delete(a.ephemeralByKey, key)
			continue
		}
		a.ephemeralByKey[key] = idxs
	}
}

// ephemeralGroupKeyLocked returns the grouping key for a tool result. Results
// with the same tool name and argument signature (e.g. reading the same
// path/offset/limit) share a key so redundant re-reads collapse to the newest
// entry, while results targeting different arguments keep independent keep
// windows and never evict one another. Falls back to the tool name alone when
// the originating call's arguments are unknown.
func (a *Agent) ephemeralGroupKeyLocked(m llm.Message, toolName string) string {
	if id := strings.TrimSpace(m.ToolCallID); id != "" {
		if sig, ok := a.ephemeralSigByCall[id]; ok && sig != "" {
			return toolName + "\x00" + sig
		}
	}
	return toolName
}

// toolNameFromGroupKey extracts the tool name from a group key produced by
// ephemeralGroupKeyLocked.
func toolNameFromGroupKey(key string) string {
	if idx := strings.IndexByte(key, '\x00'); idx >= 0 {
		return key[:idx]
	}
	return key
}

func (a *Agent) resetEphemeralTrackingLocked() {
	a.ephemeralScanFrom = 0
	if a.ephemeralByKey == nil {
		a.ephemeralByKey = make(map[string][]int)
	} else {
		for key := range a.ephemeralByKey {
			delete(a.ephemeralByKey, key)
		}
	}
	if a.ephemeralSigByCall == nil {
		a.ephemeralSigByCall = make(map[string]string)
	} else {
		for id := range a.ephemeralSigByCall {
			delete(a.ephemeralSigByCall, id)
		}
	}
}

// NotifyTodoCompletion records a deterministic checkpoint signal. It does not
// bypass the normal compaction watermarks or force an LLM summary at low usage.
func (a *Agent) NotifyTodoCompletion() {
	a.todoCompactionPending.Store(true)
}

func (a *Agent) checkAndCompact(ctx context.Context, last *llm.Completion, out *eventOutput, additionalTokens ...int) error {
	currentHistoryGrowth := 0
	for _, value := range additionalTokens {
		if value > 0 {
			currentHistoryGrowth += value
		}
	}
	return a.checkAndCompactWithGrowth(ctx, last, out, currentHistoryGrowth, 0)
}

func (a *Agent) checkAndCompactWithGrowth(ctx context.Context, last *llm.Completion, out *eventOutput, currentHistoryGrowth, pendingHistoryGrowth int) error {
	if !a.hasCompactor || a.compactor == nil || last == nil {
		return nil
	}
	if ctx != nil && ctx.Err() != nil {
		return ctx.Err()
	}
	a.applyPendingCompaction(out)
	decisionUsage := a.effectiveCompactionUsageWithGrowth(last.Usage, currentHistoryGrowth, pendingHistoryGrowth)
	trigger, watermark := a.compactionTriggerAndWatermarkForUsage(decisionUsage)
	overflow := watermark == "overflow"
	ordinaryAdmission := false
	if !overflow {
		ordinaryAdmission = a.shouldAttemptCompactionUsage(ctx, decisionUsage)
	}
	legacyDecision := compactionDecision{
		run:             overflow || ordinaryAdmission,
		trigger:         trigger,
		targetWatermark: watermark,
	}
	a.observeAutomaticCompactionDecision(legacyDecision, shadowAutomaticCompactionDecision(automaticCompactionObservation{
		overflow:          overflow,
		ordinaryAdmission: ordinaryAdmission,
		trigger:           trigger,
		targetWatermark:   watermark,
	}))
	if legacyDecision.targetWatermark == "overflow" {
		return a.compactSyncOverflow(ctx, last, decisionUsage, out)
	}
	if !legacyDecision.run {
		return nil
	}
	if !a.compactionInFlight.CompareAndSwap(false, true) {
		return nil
	}
	a.emitCompactionDecisionProvenance(out, decisionUsage)
	a.mu.Lock()
	messages := make([]llm.Message, len(a.messages))
	copy(messages, a.messages)
	a.mu.Unlock()
	snapshotLen := len(messages)
	triggerUsage := cloneUsage(last.Usage)
	// Retain synchronously before launch; otherwise the parent could release
	// and install a queued replacement before the child reads the service.
	releaseCompactionRuntime := a.retainCompactionRuntimeUse()
	go func() {
		defer releaseCompactionRuntime()
		a.runCompactionAsync(ctx, messages, snapshotLen, decisionUsage, triggerUsage, trigger, watermark)
	}()
	return nil
}

// emitCompactionDecisionProvenance publishes the calibration sample for a
// compaction decision whose token value adds a local history-growth estimate to
// an exact provider count. The decision total is a conservative estimate rather
// than a measurement, and the exact provider numbers it was built from only
// survive in the diagnostic Provider* fields of a usage object that never
// leaves the SDK: the pipeline result reports the raw trigger usage instead. So
// without this event the estimator-versus-provider drift that moves every
// watermark stays unobservable, which is what makes the mixed decision unsafe.
func (a *Agent) emitCompactionDecisionProvenance(out *eventOutput, decisionUsage *llm.Usage) {
	if a == nil || out == nil || decisionUsage == nil {
		return
	}
	if decisionUsage.ProviderPromptTokens == nil && decisionUsage.ProviderTotalTokens == nil {
		return
	}
	metadata := map[string]any{
		"decision_tokens":        decisionUsage.PromptTokens,
		"prompt_tokens_source":   decisionUsage.PromptTokensSource,
		"prompt_tokens_valid":    decisionUsage.PromptTokensValid,
		"provider_prompt_tokens": decisionUsage.ProviderPromptTokens,
		"provider_total_tokens":  decisionUsage.ProviderTotalTokens,
	}
	if decisionUsage.ProviderTotalTokens != nil {
		metadata["estimated_growth_tokens"] = decisionUsage.PromptTokens - *decisionUsage.ProviderTotalTokens
	}
	a.emitEvent(out, WarnEvent{
		Kind:     "compaction_decision_estimate_mixed",
		Message:  "compaction watermark decision added a local history-growth estimate to an exact provider token count; the decision total is an estimate, not a measurement",
		Metadata: metadata,
	})
}

func (a *Agent) runCompactionAsync(ctx context.Context, snapshot []llm.Message, snapshotLen int, decisionUsage *llm.Usage, triggerUsage *llm.Usage, trigger string, watermark string) {
	defer a.releaseCompactionInFlight()

	compactCtx, cancelCompact := asyncCompactionContext(ctx)
	defer cancelCompact()
	newMsgs, res, err := a.compactWithRetry(compactCtx, snapshot, compaction.PipelineRequest{
		Trigger:         trigger,
		Usage:           decisionUsage,
		TargetWatermark: watermark,
		AllowSummary:    a.compactionSummaryAllowed(),
	})
	if err != nil {
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			return
		}
		a.warnf("compaction failed after %d attempt(s): %v", a.compactionMaxAttempts(), err)
		a.compactionRetryPending.Store(true)
		a.noteCompactionFailure()
		return
	}
	switch strings.TrimSpace(trigger) {
	case "todo_checkpoint":
		a.todoCompactionPending.Store(false)
	case "retry_checkpoint":
		a.compactionRetryPending.Store(false)
	}
	if !res.Compacted {
		return
	}
	a.noteCompactionSuccess()
	newMsgs = a.withPreservedSystem(snapshot, newMsgs)
	res = a.reconcileCompactionTelemetry(res, snapshot, newMsgs, 0)

	a.pendingCompactionMu.Lock()
	a.pendingCompaction = &pendingCompaction{
		messages:     newMsgs,
		snapshotLen:  snapshotLen,
		result:       a.withCompactionTelemetry(res, trigger, watermark, triggerUsage),
		triggerUsage: triggerUsage,
	}
	a.pendingCompactionMu.Unlock()
}

const asyncCompactionCancelGrace = 100 * time.Millisecond

func asyncCompactionContext(parent context.Context) (context.Context, context.CancelFunc) {
	if parent == nil {
		return context.WithCancel(context.Background())
	}
	ctx, cancel := context.WithCancel(context.WithoutCancel(parent))
	go func() {
		select {
		case <-parent.Done():
		case <-ctx.Done():
			return
		}
		timer := time.NewTimer(asyncCompactionCancelGrace)
		defer timer.Stop()
		select {
		case <-timer.C:
			cancel()
		case <-ctx.Done():
		}
	}()
	return ctx, cancel
}

func (a *Agent) compactSyncOverflow(ctx context.Context, last *llm.Completion, decisionUsage *llm.Usage, out *eventOutput) error {
	if !a.hasCompactor || a.compactor == nil || last == nil {
		return nil
	}
	if ctx != nil && ctx.Err() != nil {
		return ctx.Err()
	}
	a.applyPendingCompaction(out)
	if !a.compactionInFlight.CompareAndSwap(false, true) {
		if err := a.waitForCompactionIdle(ctx, out); err != nil {
			return err
		}
		a.applyPendingCompaction(out)
		if !a.compactionInFlight.CompareAndSwap(false, true) {
			return nil
		}
	}
	defer a.releaseCompactionInFlight()

	a.emitCompactionDecisionProvenance(out, decisionUsage)
	a.mu.Lock()
	messages := make([]llm.Message, len(a.messages))
	copy(messages, a.messages)
	a.mu.Unlock()

	snapshotLen := len(messages)
	triggerUsage := cloneUsage(last.Usage)
	trigger, watermark := a.compactionTriggerAndWatermarkForUsage(decisionUsage)
	allowSummary, plannedAttempts := a.overflowSummaryPlan()
	newMsgs, res, err := a.compactOverflowWithPlan(ctx, messages, decisionUsage, allowSummary, plannedAttempts)
	if err != nil {
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			return err
		}
		a.warnf("compaction failed after %d attempt(s): %v", plannedAttempts, err)
		a.compactionRetryPending.Store(true)
		a.noteCompactionFailure()
		// Overflow means the next provider request cannot be sent as-is, so
		// returning the error aborts the turn and every later turn identically.
		// Fall back to an emergency trim so the session keeps an escape path.
		if !a.applyEmergencyTrim(messages, snapshotLen, triggerUsage, err, out) {
			return err
		}
		return nil
	}
	if !res.Compacted {
		if !allowSummary {
			// The summary tier was suppressed by the failure cooldown and the
			// local tiers could not reduce anything, so the next request would
			// still exceed the window. Trim as the last escape hatch; if even
			// that is impossible the caller must not be told compaction worked -
			// returning nil here would mask the overflow and let the turn send a
			// request that is still over the window.
			overflowErr := errors.New("summary tier suppressed by the compaction failure cooldown and local reduction changed nothing")
			if !a.applyEmergencyTrim(messages, snapshotLen, triggerUsage, overflowErr, out) {
				return overflowErr
			}
		}
		return nil
	}
	a.noteCompactionSuccess()
	if strings.TrimSpace(res.Watermark) == "summarize" {
		a.todoCompactionPending.Store(false)
		a.compactionRetryPending.Store(false)
	}
	newMsgs = a.withPreservedSystem(messages, newMsgs)
	res = a.reconcileCompactionTelemetry(res, messages, newMsgs, 0)

	a.pendingCompactionMu.Lock()
	a.pendingCompaction = &pendingCompaction{
		messages:     newMsgs,
		snapshotLen:  snapshotLen,
		result:       a.withCompactionTelemetry(res, trigger, watermark, triggerUsage),
		triggerUsage: triggerUsage,
	}
	a.pendingCompactionMu.Unlock()
	a.applyPendingCompaction(out)
	return nil
}

// applyEmergencyTrim publishes an emergency-trimmed history as a compaction
// result. It reports false when the trim could not produce a legal sendable
// history, in which case the caller must keep surfacing the original failure.
func (a *Agent) applyEmergencyTrim(messages []llm.Message, snapshotLen int, triggerUsage *llm.Usage, cause error, out *eventOutput) bool {
	trimmed, trimmedOK := a.emergencyTrimHistory(messages)
	if !trimmedOK {
		return false
	}
	a.warnf("overflow compaction failed; applying emergency history trim (%d -> %d messages)", len(messages), len(trimmed))
	a.emitEvent(out, WarnEvent{
		Kind:    "compaction_emergency_trim",
		Message: fmt.Sprintf("compaction failed at the overflow boundary; emergency-trimmed history from %d to %d messages to keep the turn runnable", len(messages), len(trimmed)),
	})
	emergencyRes := compaction.Result{
		Compacted:    true,
		Trigger:      "overflow",
		Watermark:    "overflow",
		TiersApplied: []string{"emergency_trim"},
		Warnings: []string{fmt.Sprintf(
			"[WARN] Overflow compaction failed - emergency-trimmed history to the newest messages that fit the prompt budget: %v", cause)},
	}
	emergencyRes = a.reconcileCompactionTelemetry(emergencyRes, messages, trimmed, 0)
	a.pendingCompactionMu.Lock()
	a.pendingCompaction = &pendingCompaction{
		messages:     trimmed,
		snapshotLen:  snapshotLen,
		result:       a.withCompactionTelemetry(emergencyRes, "overflow", "overflow", triggerUsage),
		triggerUsage: triggerUsage,
	}
	a.pendingCompactionMu.Unlock()
	a.applyPendingCompaction(out)
	return true
}

// compactionFailureCooldown parameters bound the cost of a compaction pipeline
// that keeps failing (e.g. a summary model whose output never satisfies the
// quality gate). Without them the loop re-runs the full summary pipeline on
// every turn while token usage stays above the watermark.
const (
	compactionSummaryDisableStreak = 2
	compactionCooldownBase         = 30 * time.Second
	compactionCooldownMax          = 10 * time.Minute
)

// noteCompactionFailure records a failed compaction run and arms an exponential
// cooldown so the next turns do not immediately repeat the same expensive work.
func (a *Agent) noteCompactionFailure() {
	if a == nil {
		return
	}
	streak := a.compactionFailureStreak.Add(1)
	cooldown := compactionCooldownBase
	for i := uint64(1); i < streak && cooldown < compactionCooldownMax; i++ {
		cooldown *= 2
	}
	if cooldown > compactionCooldownMax {
		cooldown = compactionCooldownMax
	}
	a.compactionCooldownUntil.Store(time.Now().Add(cooldown).UnixNano())
	if streak == compactionSummaryDisableStreak {
		a.warnf(
			"warning: compaction failed %d consecutive times; disabling the summary tier and running local reduction only until it succeeds again",
			streak,
		)
	}
}

// noteCompactionSuccess clears the failure streak and any pending cooldown.
func (a *Agent) noteCompactionSuccess() {
	if a == nil {
		return
	}
	a.compactionFailureStreak.Store(0)
	a.compactionCooldownUntil.Store(0)
}

// compactionInCooldown reports whether a recent failure streak still suppresses
// opportunistic (non-overflow) compaction attempts.
func (a *Agent) compactionInCooldown() bool {
	if a == nil {
		return false
	}
	until := a.compactionCooldownUntil.Load()
	if until <= 0 {
		return false
	}
	if time.Now().UnixNano() >= until {
		a.compactionCooldownUntil.Store(0)
		return false
	}
	return true
}

// compactionSummaryAllowed reports whether the expensive summary tier should
// still be attempted. After a streak of failures only the local tiers run.
func (a *Agent) compactionSummaryAllowed() bool {
	if a == nil {
		return true
	}
	return a.compactionFailureStreak.Load() < compactionSummaryDisableStreak
}

// overflowSummaryPlan bounds what the overflow path may spend on the summary
// tier. Overflow is a hard boundary — the next request cannot be sent without
// reducing history — so compaction itself is never skipped; only the expensive
// summary retries are. A failure streak drops the summary tier entirely (local
// tiers plus the emergency trim remain), and an armed cooldown allows a single
// summary attempt instead of the full retry budget.
func (a *Agent) overflowSummaryPlan() (allowSummary bool, attempts int) {
	attempts = a.compactionMaxAttempts()
	if attempts <= 0 {
		attempts = 1
	}
	if a == nil {
		return true, attempts
	}
	if !a.compactionSummaryAllowed() {
		// Local tiers do not call the summary model, so retrying them is
		// pointless: one pass is all the pipeline can achieve.
		return false, 1
	}
	if a.compactionInCooldown() {
		return true, 1
	}
	return true, attempts
}

// emergencyTrimHistory is the last-resort escape hatch for the overflow path:
// when both the summary tier and local reduction fail to produce a smaller
// history, the turn would otherwise abort with an error on every subsequent
// user input. It keeps the protected messages plus the newest tail that fits the
// prompt budget, then repairs tool-call pairing. It reports compacted=false
// when nothing could be dropped, so a genuinely irreducible history still
// surfaces the original error instead of silently continuing over the window.
//
// Acceptance is decided on the final result, not per message: a trim is only
// applied when the repaired history is still a legal, sendable conversation and
// actually fits the prompt budget (see emergencyTrimResultUsable). Trimming to a
// system-only history would be strictly worse than the original error - it
// guarantees a provider rejection and destroys the user request along with the
// session - and so would reporting success for a history that still overflows.
// Rejection is per candidate though: the search keeps giving up more of the
// newest content until something fits, and only reports failure once every legal
// candidate down to the protected messages alone has been ruled out.
func (a *Agent) emergencyTrimHistory(messages []llm.Message) ([]llm.Message, bool) {
	if a == nil || a.compactor == nil || len(messages) == 0 {
		return messages, false
	}
	budget := a.compactor.ThresholdTokens()
	if budget <= 0 {
		return messages, false
	}
	protected := make([]llm.Message, 0, 1)
	firstUnprotected := len(messages)
	for i, m := range messages {
		if emergencyTrimProtected(m) {
			protected = append(protected, m)
			continue
		}
		if firstUnprotected == len(messages) {
			firstUnprotected = i
		}
	}
	remaining := budget - a.compactor.EstimateMessages(protected)
	if remaining <= 0 {
		return messages, false
	}
	blocks := emergencyTrimBlocks(messages, firstUnprotected)
	// Rejection is per candidate, never wholesale. skip counts how many of the
	// newest blocks a candidate gives up: skip == 0 is the tail that still
	// contains the newest block (what the model must answer), and each later
	// candidate drops one more of the newest blocks. An oversized newest block
	// therefore no longer vetoes a trim the older content would have fit into -
	// that veto aborted turns even though a legal in-budget history existed,
	// which is the failure mode the emergency trim exists to prevent.
	// skip == len(blocks) is the protected-only candidate, offered so a history
	// whose every block is individually oversized can still be reduced;
	// emergencyTrimResultUsable is what keeps it from degenerating into an
	// unsendable (system-only) history.
	for skip := 0; skip <= len(blocks); skip++ {
		trimmed, ok := a.emergencyTrimCandidate(messages, blocks, skip, remaining)
		if !ok {
			continue
		}
		if len(trimmed) >= len(messages) {
			// Nothing outside the protected prefix could be dropped. Only the
			// skip == 0 candidate can retain every message and every later
			// candidate retains strictly less, so there is no trim to report.
			return messages, false
		}
		if a.emergencyTrimResultUsable(messages, trimmed, budget) {
			return trimmed, true
		}
	}
	return messages, false
}

// emergencyTrimCandidate builds one trim candidate: the protected messages plus
// the newest run of whole tool blocks that fits remaining, starting at
// blocks[skip] and walking towards the oldest. It reports false when not even
// blocks[skip] fits on its own, which is the signal for the caller to give up one
// more of the newest blocks and try again. skip == len(blocks) yields the
// protected-only candidate.
//
// Whole blocks are the unit because a window that starts between an assistant
// tool_use and its tool results would lose the pairing and be discarded by the
// repair pass.
func (a *Agent) emergencyTrimCandidate(messages []llm.Message, blocks []emergencyTrimBlock, skip, remaining int) ([]llm.Message, bool) {
	keepFrom, keepTo := -1, -1
	if skip < len(blocks) {
		keepTo = blocks[skip].end
		for _, block := range blocks[skip:] {
			cost := a.compactor.EstimateMessages(messages[block.start : block.end+1])
			if cost > remaining {
				break
			}
			remaining -= cost
			keepFrom = block.start
		}
		if keepFrom < 0 {
			return nil, false
		}
	}
	// Keep chronological order: hoisting protected messages in front of the tail
	// would reorder a user turn ahead of the assistant turn that answered it.
	trimmed := make([]llm.Message, 0, len(messages))
	for i, m := range messages {
		if (keepFrom >= 0 && i >= keepFrom && i <= keepTo) || emergencyTrimProtected(m) {
			trimmed = append(trimmed, m)
		}
	}
	return repairToolCallPairsPreservingPendingContinuation(trimmed), true
}

// emergencyTrimBlock is one atomic unit of the unprotected history: an assistant
// tool_use message together with its tool results, or a single standalone
// message. Both bounds are inclusive.
type emergencyTrimBlock struct {
	start int
	end   int
}

// emergencyTrimBlocks splits the unprotected part of the history at and after
// lowerBound into atomic blocks, newest first. Protected messages are retained
// unconditionally, so here they only act as separators and belong to no block.
func emergencyTrimBlocks(messages []llm.Message, lowerBound int) []emergencyTrimBlock {
	if lowerBound >= len(messages) {
		return nil
	}
	blocks := make([]emergencyTrimBlock, 0, len(messages)-lowerBound)
	for i := len(messages) - 1; i >= lowerBound; {
		if emergencyTrimProtected(messages[i]) {
			i--
			continue
		}
		start := emergencyTrimBlockStart(messages, i, lowerBound)
		blocks = append(blocks, emergencyTrimBlock{start: start, end: i})
		i = start - 1
	}
	return blocks
}

// repairToolCallPairsPreservingPendingContinuation repairs tool-call pairing
// without touching a trailing in-flight tool-call continuation. Compaction can
// run at the overflow boundary while a truncated tool_use block is still
// accumulating its arguments (see the continuation paths in the query loop), and
// that block is not yet followed by any tool_result. A plain repair pass would
// clear its ToolCalls, which destroys the partial arguments the loop has to
// merge the next chunk into and leaves the continuation unable to ever complete.
// The outgoing request is repaired separately just before it is sent, so keeping
// the block in history is safe.
func repairToolCallPairsPreservingPendingContinuation(messages []llm.Message) []llm.Message {
	tail := pendingContinuationTailIndex(messages)
	if tail < 0 {
		repaired, _ := repairToolCallPairs(messages)
		return repaired
	}
	head, _ := repairToolCallPairs(messages[:tail])
	out := make([]llm.Message, 0, len(head)+len(messages)-tail)
	out = append(out, head...)
	out = append(out, messages[tail:]...)
	return out
}

// pendingContinuationTailIndex returns the index of a trailing assistant
// tool_use block whose arguments are still being accumulated by an in-flight
// tool-call continuation, or -1 when the history has no such block. The block
// has no tool results yet and is either the newest message or followed only by
// the framework's continuation reminder: compaction at the overflow boundary
// runs before that reminder is appended, so both shapes must be recognized.
func pendingContinuationTailIndex(messages []llm.Message) int {
	end := len(messages)
	if end > 0 && isContinuationReminderMessage(messages[end-1]) {
		end--
	}
	if end == 0 {
		return -1
	}
	last := messages[end-1]
	if last.Role != llm.RoleAssistant || len(last.ToolCalls) == 0 {
		return -1
	}
	return end - 1
}

// isContinuationReminderMessage reports whether the message is the internal user
// reminder the loop appends to request the rest of a truncated tool call.
func isContinuationReminderMessage(m llm.Message) bool {
	if m.Role != llm.RoleUser {
		return false
	}
	switch strings.TrimSpace(m.Name) {
	case messageorigin.Name(messageorigin.KindToolCallContinuation),
		messageorigin.Name(messageorigin.KindMaxTokensContinuation):
		return true
	default:
		return false
	}
}

// emergencyTrimBlockStart returns the first index of the atomic block that ends
// at index. Tool results and the assistant tool_use message that owns them form
// one unit: keeping only part of it drops the whole block in the repair pass.
func emergencyTrimBlockStart(messages []llm.Message, index, lowerBound int) int {
	start := index
	if messages[index].Role != llm.RoleTool {
		return start
	}
	for start-1 >= lowerBound && messages[start-1].Role == llm.RoleTool {
		start--
	}
	if start-1 >= lowerBound &&
		messages[start-1].Role == llm.RoleAssistant &&
		len(messages[start-1].ToolCalls) > 0 {
		start--
	}
	return start
}

// emergencyTrimResultUsable validates the trimmed history as a whole instead of
// trusting the per-message protection rules. A trim is only usable when it
// leaves a conversation the provider can actually accept: at least one sendable
// non-system message, legal tool-call pairing, every protected message of the
// original still present, and a post-trim size that actually fits the prompt
// budget.
//
// The budget re-check is what makes the reported success meaningful: without it
// the caller publishes compaction.Result{Compacted: true} and returns nil, and
// the turn then sends a request that is still over the window - the overflow
// error the trim was supposed to escape gets masked instead of surfaced. It
// rejects only the candidate it is given: emergencyTrimHistory answers a
// rejection by giving up more of the newest content and asking again, so an
// oversized newest block no longer discards the older content that would have
// fit.
func (a *Agent) emergencyTrimResultUsable(original, trimmed []llm.Message, budget int) bool {
	if len(trimmed) == 0 {
		return false
	}
	sendable := false
	for _, m := range trimmed {
		if m.Role == llm.RoleSystem {
			continue
		}
		if !m.Content.IsEmpty() || len(m.ToolCalls) > 0 {
			sendable = true
			break
		}
	}
	if !sendable {
		// System-only (or content-free) histories are rejected by the provider,
		// so the original error is the better outcome.
		return false
	}
	// The repair is idempotent: a second pass that still reports a change means
	// the trimmed history is not a legal conversation. An in-flight tool-call
	// continuation is exempt in both passes, so it does not read as illegal here.
	if _, changed := repairToolCallPairs(
		stripPendingContinuationTail(trimmed)); changed {
		return false
	}
	kept := 0
	for _, m := range trimmed {
		if emergencyTrimProtected(m) {
			kept++
		}
	}
	want := 0
	for _, m := range original {
		if emergencyTrimProtected(m) {
			want++
		}
	}
	if kept < want {
		return false
	}
	if budget > 0 && a != nil && a.compactor != nil {
		if estimate := a.compactor.EstimateMessages(trimmed); estimate > budget {
			// The retained newest block alone still overflows the window, so
			// reporting success would only hide the real overflow from the caller.
			a.warnf("emergency history trim still exceeds the prompt budget (%d > %d tokens); surfacing the overflow instead of reporting compaction success", estimate, budget)
			return false
		}
	}
	return true
}

// stripPendingContinuationTail drops a trailing in-flight tool-call continuation
// so pairing legality can be validated on the part of the history the invariant
// actually applies to.
func stripPendingContinuationTail(messages []llm.Message) []llm.Message {
	if tail := pendingContinuationTailIndex(messages); tail >= 0 {
		return messages[:tail]
	}
	return messages
}

// emergencyTrimProtected reports whether a message must survive an emergency
// trim regardless of the token budget:
//   - the system prefix, which also carries host-injected runtime context
//     (memory, wide-research protocol) as system-role messages;
//   - the compaction summary, the only remaining record of the history it
//     replaced;
//   - real user messages, i.e. the actual requests of the session. Dropping one
//     silently discards the task while framework-authored reminders survive.
//
// Framework-authored user messages (loop-guard, continuation, require-done
// reminders and the host's goode_internal_* injections) are regenerated by the
// loop when they are still needed, so they stay droppable. Destroyed ephemeral
// tool results are droppable placeholders by construction.
func emergencyTrimProtected(m llm.Message) bool {
	if m.Role == llm.RoleSystem {
		return true
	}
	if m.Role == llm.RoleUser && strings.TrimSpace(m.Name) == compaction.CompactionSummaryMessageName {
		return true
	}
	return messageorigin.IsRealUserMessage(m)
}

func (a *Agent) compactOverflowWithRetry(ctx context.Context, messages []llm.Message, usage *llm.Usage) ([]llm.Message, compaction.Result, error) {
	allowSummary, attempts := a.overflowSummaryPlan()
	return a.compactOverflowWithPlan(ctx, messages, usage, allowSummary, attempts)
}

// compactOverflowWithPlan runs the overflow pipeline under an explicit summary
// budget so the caller can report exactly how many attempts were planned.
func (a *Agent) compactOverflowWithPlan(ctx context.Context, messages []llm.Message, usage *llm.Usage, allowSummary bool, attempts int) ([]llm.Message, compaction.Result, error) {
	if attempts <= 0 {
		attempts = 1
	}
	var lastErr error
	var lastRes compaction.Result
	var fallbackMsgs []llm.Message
	var fallbackRes compaction.Result
	for attempt := 1; attempt <= attempts; attempt++ {
		if err := ctx.Err(); err != nil {
			return messages, lastRes, err
		}
		pipelineMsgs, pipelineRes, err := a.compactor.CompactPipeline(ctx, a.llm, messages, compaction.PipelineRequest{
			Trigger:         "overflow",
			Usage:           usage,
			TargetWatermark: "overflow",
			AllowSummary:    allowSummary,
		})
		if pipelineRes.Compacted {
			fallbackMsgs = pipelineMsgs
			fallbackRes = pipelineRes
		}
		if err == nil {
			return pipelineMsgs, pipelineRes, nil
		}
		lastErr = err
		lastRes = pipelineRes
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) || ctx.Err() != nil {
			if ctxErr := ctx.Err(); ctxErr != nil {
				return messages, lastRes, ctxErr
			}
			return messages, lastRes, err
		}
		if attempt >= attempts {
			break
		}
		delay := a.compactionRetryDelay(attempt)
		a.warnf("overflow summary compaction failed (attempt %d/%d): %v", attempt, attempts, err)
		if delay <= 0 {
			continue
		}
		t := time.NewTimer(delay)
		select {
		case <-ctx.Done():
			if !t.Stop() {
				<-t.C
			}
			return messages, lastRes, ctx.Err()
		case <-t.C:
		}
	}
	if len(fallbackMsgs) > 0 && fallbackRes.Compacted {
		fallbackRes.Warnings = append(fallbackRes.Warnings, fmt.Sprintf("[WARN] Overflow summary compaction failed after local reduction - continuing with local compaction and scheduling retry: %v", lastErr))
		a.compactionRetryPending.Store(true)
		return fallbackMsgs, fallbackRes, nil
	}
	if lastErr == nil {
		lastErr = errors.New("overflow summary compaction failed")
	}
	return messages, lastRes, lastErr
}

// waitForCompactionIdle blocks until no compaction is in flight. Completion is
// signalled through a broadcast channel, so the wait normally wakes up
// immediately instead of spinning; a slow backoff poll remains as a safety net
// for callers that flip the flag without signalling. The wait is bounded: an
// in-flight compaction that never completes must not hang the turn forever,
// because its result is only published into pendingCompaction and is picked up
// at the next boundary anyway.
func (a *Agent) waitForCompactionIdle(ctx context.Context, out *eventOutput) error {
	if a == nil || !a.compactionInFlight.Load() {
		return nil
	}
	const (
		minPollInterval = 25 * time.Millisecond
		maxPollInterval = 200 * time.Millisecond
	)
	limit := a.compactionIdleWaitLimit()
	deadline := time.NewTimer(limit)
	defer stopTimerDrain(deadline)
	poll := time.NewTimer(minPollInterval)
	defer stopTimerDrain(poll)
	interval := minPollInterval
	for a.compactionInFlight.Load() {
		idle := a.compactionIdleSignal()
		if !a.compactionInFlight.Load() {
			return nil
		}
		var done <-chan struct{}
		if ctx != nil {
			done = ctx.Done()
		}
		select {
		case <-done:
			return ctx.Err()
		case <-idle:
		case <-deadline.C:
			a.warnf("warning: giving up waiting for in-flight compaction after %s; continuing without it", limit)
			a.emitEvent(out, WarnEvent{
				Kind:    "compaction_wait_timeout",
				Message: fmt.Sprintf("in-flight compaction did not finish within %s; continuing the turn without waiting for it", limit),
			})
			return nil
		case <-poll.C:
			if interval < maxPollInterval {
				interval *= 2
				if interval > maxPollInterval {
					interval = maxPollInterval
				}
			}
			poll.Reset(interval)
		}
	}
	return nil
}

// compactionIdleWaitLimit bounds waitForCompactionIdle by the worst-case
// runtime of one compaction run (provider timeout times the attempt budget)
// plus the async cancellation grace.
func (a *Agent) compactionIdleWaitLimit() time.Duration {
	timeout := compaction.DefaultCompactionTimeout
	if a != nil && a.compactor != nil && a.compactor.Config.CompactionTimeout > 0 {
		timeout = a.compactor.Config.CompactionTimeout
	}
	attempts := 1
	if a != nil {
		if n := a.compactionMaxAttempts(); n > 1 {
			attempts = n
		}
	}
	return timeout*time.Duration(attempts) + asyncCompactionCancelGrace
}

// compactionIdleSignal returns a channel that is closed when the current
// in-flight compaction reports idle.
func (a *Agent) compactionIdleSignal() <-chan struct{} {
	a.compactionIdleMu.Lock()
	defer a.compactionIdleMu.Unlock()
	if a.compactionIdleCh == nil {
		a.compactionIdleCh = make(chan struct{})
	}
	return a.compactionIdleCh
}

// releaseCompactionInFlight clears the in-flight flag and wakes every waiter.
func (a *Agent) releaseCompactionInFlight() {
	a.compactionInFlight.Store(false)
	a.compactionIdleMu.Lock()
	ch := a.compactionIdleCh
	a.compactionIdleCh = nil
	a.compactionIdleMu.Unlock()
	if ch != nil {
		close(ch)
	}
}

func (a *Agent) hasPendingCompaction() bool {
	a.pendingCompactionMu.Lock()
	defer a.pendingCompactionMu.Unlock()
	return a.pendingCompaction != nil
}

func (a *Agent) applyPendingCompaction(out *eventOutput) {
	if !a.hasCompactor {
		return
	}
	a.pendingCompactionMu.Lock()
	pending := a.pendingCompaction
	if pending != nil {
		a.pendingCompaction = nil
	}
	a.pendingCompactionMu.Unlock()
	if pending == nil || !pending.result.Compacted {
		return
	}

	// Build an immutable candidate under the history lock, then release the
	// lock before any host-controlled ledger/checkpoint I/O. The source snapshot
	// is compared again before publication so a concurrent history mutation is
	// never overwritten by a stale compaction result.
	a.mu.Lock()
	source := llm.CloneMessages(a.messages)
	currentLen := len(source)
	if currentLen < pending.snapshotLen {
		a.mu.Unlock()
		a.warnf("compaction apply skipped: history shrank (%d < %d); scheduling retry", currentLen, pending.snapshotLen)
		a.requeuePendingCompaction(pending)
		a.compactionRetryPending.Store(true)
		return
	}
	tailCap := currentLen - pending.snapshotLen
	pairingRepaired := false
	merged := llm.CloneMessages(pending.messages)
	if tailCap > 0 {
		merged = append(merged, llm.CloneMessages(source[pending.snapshotLen:])...)
		// pending.messages dropped every assistant tool_use block, so a tail
		// that starts inside a tool block would splice orphaned tool results
		// onto the summary. Repair rather than trust the caller to compact only
		// on user-message boundaries.
		if repaired, changed := repairToolCallPairs(merged); changed {
			pairingRepaired = true
			merged = repaired
		}
	}
	a.mu.Unlock()

	if pairingRepaired {
		a.warnf("compaction apply repaired tool-call pairing at the summary/tail splice point")
	}
	// Estimation may invoke a host-supplied TokenEstimator; keep it outside the
	// history lock for the same re-entrancy reason as checkpoint persistence.
	pending.result = a.reconcileCompactionTelemetry(pending.result, source, merged, 0)
	commit, commitErr := a.persistCompactionCheckpoint(context.Background(), merged, pending.result)
	if commitErr != nil {
		a.requeuePendingCompaction(pending)
		a.compactionRetryPending.Store(true)
		warning := commitErr.Error()
		if len(commit.result.Warnings) > 0 {
			warning = commit.result.Warnings[len(commit.result.Warnings)-1]
		}
		a.warnf("%s", warning)
		return
	}

	a.mu.Lock()
	if !reflect.DeepEqual(a.messages, source) {
		a.mu.Unlock()
		rollbackErr := error(nil)
		if commit.persisted {
			rollbackErr = a.compactor.RollbackPendingLedger(context.Background(), &commit.transaction)
		}
		a.requeuePendingCompaction(pending)
		a.compactionRetryPending.Store(true)
		if rollbackErr != nil {
			a.warnf("compaction apply deferred because history changed while checkpoint persistence was running; ledger rollback failed: %v", rollbackErr)
		} else {
			a.warnf("compaction apply deferred because history changed while checkpoint persistence was running; ledger state was rolled back and the unreferenced checkpoint can be garbage-collected")
		}
		return
	}
	if commit.persisted {
		a.compactor.FinalizePendingLedger(&commit.transaction)
	}
	a.messages = merged
	a.resetEphemeralTrackingLocked()
	a.compactionGeneration.Add(1)
	a.mu.Unlock()

	a.emitCompactionWithAccounting(out, CompactionEvent{Result: commit.result, TriggerUsage: pending.triggerUsage})
}

func (a *Agent) requeuePendingCompaction(pending *pendingCompaction) {
	if a == nil || pending == nil {
		return
	}
	a.pendingCompactionMu.Lock()
	if a.pendingCompaction == nil {
		a.pendingCompaction = pending
	}
	a.pendingCompactionMu.Unlock()
}

// CompactNow forces a compaction run regardless of current token usage.
func (a *Agent) CompactNow(ctx context.Context) (compaction.Result, error) {
	return a.CompactPipelineNow(ctx, compaction.PipelineRequest{
		Trigger:         "manual",
		TargetWatermark: "summarize",
		AllowSummary:    true,
		ForceSummary:    true,
	})
}

// CompactPipelineNow applies the canonical compaction pipeline synchronously.
// Hosts use this for preflight so they do not duplicate local-versus-summary
// decisions outside the SDK.
func (a *Agent) CompactPipelineNow(ctx context.Context, req compaction.PipelineRequest) (compaction.Result, error) {
	releaseCompactionRuntime, err := a.beginCompactionRuntimeUse(ctx)
	if err != nil {
		return compaction.Result{Compacted: false}, err
	}
	defer releaseCompactionRuntime()
	if !a.hasCompactor || a.compactor == nil {
		return compaction.Result{Compacted: false}, nil
	}
	a.applyPendingCompaction(nil)
	if !a.compactionInFlight.CompareAndSwap(false, true) {
		return compaction.Result{Compacted: false}, fmt.Errorf("compaction already in progress")
	}
	defer a.releaseCompactionInFlight()

	a.mu.Lock()
	orig := make([]llm.Message, len(a.messages))
	copy(orig, a.messages)
	a.mu.Unlock()

	newMsgs, res, err := a.compactWithRetry(ctx, orig, req)
	if err != nil {
		return res, err
	}
	a.todoCompactionPending.Store(false)
	a.compactionRetryPending.Store(false)
	newMsgs = a.withPreservedSystem(orig, newMsgs)
	if res.Compacted {
		res = a.reconcileCompactionTelemetry(res, orig, newMsgs, req.AdditionalTokens)
		res, err = a.commitCompactionCheckpoint(ctx, newMsgs, res)
		if err != nil {
			return res, err
		}
		a.mu.Lock()
		a.messages = newMsgs
		a.resetEphemeralTrackingLocked()
		a.compactionGeneration.Add(1)
		a.mu.Unlock()
	}
	return res, nil
}

// CommitCompactionCheckpoint durably records compacted provider history before
// callers replace in-memory history. A persistence failure is fail-closed: the
// returned result is not reported as compacted and the caller keeps old state.
type pendingCheckpointCommit struct {
	result      compaction.Result
	transaction compaction.Result
	persisted   bool
}

// persistCompactionCheckpoint performs all potentially blocking persistence
// without finalizing the deferred ledger transaction. The caller finalizes only
// after it has atomically published the matching in-memory history.
func (a *Agent) persistCompactionCheckpoint(ctx context.Context, messages []llm.Message, res compaction.Result) (pendingCheckpointCommit, error) {
	commit := pendingCheckpointCommit{result: res, transaction: res}
	if a == nil || !res.Compacted || a.compactor == nil || a.compactor.Config.CheckpointWriter == nil {
		return commit, nil
	}
	if ctx == nil {
		ctx = context.Background()
	}
	transaction := res
	if err := a.compactor.CommitPendingLedger(ctx, &transaction); err != nil {
		warning := fmt.Sprintf("[WARN] Compaction ledger persistence failed before runtime checkpoint - original in-memory history was preserved and compaction remains retryable. (stage=save_compaction_ledger action=check ledger storage and retry: %v)", err)
		failed := res
		failed.Compacted = false
		failed.CheckpointID = ""
		failed.Warnings = append(failed.Warnings, warning)
		commit.result = failed
		return commit, fmt.Errorf("compaction ledger persistence failed: %w", err)
	}
	checkpoint, err := compaction.NewCompactionCheckpoint(messages, transaction)
	if err == nil {
		err = a.compactor.Config.CheckpointWriter.SaveCompactionCheckpoint(ctx, checkpoint)
	}
	if err != nil {
		rollbackErr := a.compactor.RollbackPendingLedger(context.Background(), &transaction)
		warning := fmt.Sprintf("[WARN] Compaction checkpoint persistence failed - original in-memory history was preserved and compaction remains retryable. (stage=append_compaction_checkpoint action=check checkpoint storage and retry: %v)", err)
		failed := res
		failed.Compacted = false
		failed.CheckpointID = ""
		failed.Warnings = append(failed.Warnings, warning)
		if rollbackErr != nil {
			rollbackWarning := fmt.Sprintf("[ERROR] Compaction ledger rollback failed - stale ledger metadata may require a safe full rebuild on retry. (stage=rollback_compaction_ledger action=check ledger storage and retry: %v)", rollbackErr)
			failed.Warnings = append(failed.Warnings, rollbackWarning)
			commit.result = failed
			return commit, fmt.Errorf("compaction checkpoint persistence failed: %w (ledger rollback failed: %v)", err, rollbackErr)
		}
		commit.result = failed
		return commit, fmt.Errorf("compaction checkpoint persistence failed: %w", err)
	}
	commit.result = checkpoint.Result
	commit.transaction = transaction
	commit.persisted = true
	return commit, nil
}

// CommitCompactionCheckpoint durably records compacted provider history before
// callers replace in-memory history. A persistence failure is fail-closed: the
// returned result is not reported as compacted and the caller keeps old state.
func (a *Agent) CommitCompactionCheckpoint(ctx context.Context, messages []llm.Message, res compaction.Result) (compaction.Result, error) {
	releaseCompactionRuntime, err := a.beginCompactionRuntimeUse(ctx)
	if err != nil {
		return res, err
	}
	defer releaseCompactionRuntime()
	return a.commitCompactionCheckpoint(ctx, messages, res)
}

func (a *Agent) commitCompactionCheckpoint(ctx context.Context, messages []llm.Message, res compaction.Result) (compaction.Result, error) {
	commit, err := a.persistCompactionCheckpoint(ctx, messages, res)
	if err != nil {
		return commit.result, err
	}
	if commit.persisted {
		a.compactor.FinalizePendingLedger(&commit.transaction)
	}
	return commit.result, nil
}

// CompactLocalNow forces the local snip/prune reducers using an estimated token
// count. It never invokes the model and is intended for prompt preflight.
func (a *Agent) CompactLocalNow(ctx context.Context, estimatedTokens int) (compaction.Result, error) {
	return a.CompactPipelineNow(ctx, compaction.PipelineRequest{
		Trigger:         "preflight",
		EstimatedTokens: estimatedTokens,
		AllowSummary:    false,
	})
}

func (a *Agent) shouldAttemptCompaction(ctx context.Context, last *llm.Completion, additionalTokens ...int) bool {
	if !a.hasCompactor || a.compactor == nil || last == nil {
		return false
	}
	return a.shouldAttemptCompactionUsage(ctx, a.effectiveCompactionUsage(last.Usage, additionalTokens...))
}

func (a *Agent) shouldAttemptCompactionUsage(ctx context.Context, usage *llm.Usage) bool {
	if !a.hasCompactor || a.compactor == nil {
		return false
	}
	if a.compactionAdmissionObserved != nil {
		a.compactionAdmissionObserved()
	}
	if ctx != nil && ctx.Err() != nil {
		return false
	}
	if a.compactionInFlight.Load() || a.hasPendingCompaction() {
		return false
	}
	if a.compactor.IsOverflow(usage) {
		// Overflow is a hard boundary: the next request cannot be sent without
		// reducing history, so it is never suppressed by the failure cooldown.
		return true
	}
	if a.compactionInCooldown() {
		// A recent failure streak is still cooling down. Retrying the same
		// pipeline every turn only repeats its cost without changing the
		// outcome; overflow above remains the escape hatch.
		return false
	}
	if a.compactor.ShouldCompact(usage) {
		return true
	}
	return a.destroyedToolMessageCount() >= defaultDestroyedToolCompactThreshold &&
		a.compactor.PromptTokens(usage) > 0
}

func (a *Agent) effectiveCompactionUsage(usage *llm.Usage, additionalTokens ...int) *llm.Usage {
	currentHistoryGrowth := 0
	for _, value := range additionalTokens {
		if value > 0 {
			currentHistoryGrowth += value
		}
	}
	return a.effectiveCompactionUsageWithGrowth(usage, currentHistoryGrowth, 0)
}

func (a *Agent) effectiveCompactionUsageWithGrowth(usage *llm.Usage, currentHistoryGrowth, pendingHistoryGrowth int) *llm.Usage {
	if a == nil || a.compactor == nil {
		return usage
	}
	effective := usage
	usedCurrentHistoryEstimate := false
	if usage != nil && (llm.PromptUsageIsProviderValid(usage) ||
		(usage.PromptTokensSource == llm.PromptTokensSourceEstimate && usage.PromptTokens > 0)) {
		effective = llm.NormalizeUsage(usage)
	} else {
		estimated := a.compactor.EstimateMessages(a.Messages())
		effective = llm.WithPromptEstimate(usage, estimated)
		usedCurrentHistoryEstimate = true
	}
	if currentHistoryGrowth < 0 {
		currentHistoryGrowth = 0
	}
	if pendingHistoryGrowth < 0 {
		pendingHistoryGrowth = 0
	}
	additional := currentHistoryGrowth + pendingHistoryGrowth
	if usedCurrentHistoryEstimate {
		additional = pendingHistoryGrowth
	}
	if effective == nil || additional <= 0 {
		return effective
	}
	out := llm.CloneUsage(effective)
	decision := a.compactor.DecisionTokens(out) + additional
	// The growth increment is a local estimate while the base may be an exact
	// provider count, so the sum is a conservative decision value rather than a
	// measurement. Preserve the exact provider numbers in the diagnostic
	// Provider* fields before downgrading the effective source to estimate,
	// otherwise the precise count becomes unobservable for calibration.
	if llm.PromptUsageIsProviderValid(out) {
		if out.ProviderPromptTokens == nil {
			providerPrompt := out.PromptTokens
			out.ProviderPromptTokens = &providerPrompt
		}
		if out.ProviderTotalTokens == nil {
			providerTotal := out.TotalTokens
			out.ProviderTotalTokens = &providerTotal
		}
	}
	out.PromptTokens = decision
	out.CompletionTokens = 0
	out.TotalTokens = decision
	out.PromptTokensValid = false
	out.PromptTokensSource = llm.PromptTokensSourceEstimate
	out.PromptTokensSemantics = llm.PromptTokensSemanticsTotalInputV1
	return out
}

// destroyedToolMessageCount reports how many tool results have already been
// recycled to the ephemeral placeholder. A high count means context is filling
// with zero-information placeholders — a signal to compact them out even before
// the token watermark triggers.
func (a *Agent) destroyedToolMessageCount() int {
	a.mu.Lock()
	defer a.mu.Unlock()
	n := 0
	for _, m := range a.messages {
		if m.Role == llm.RoleTool && m.Destroyed {
			n++
		}
	}
	return n
}

func (a *Agent) compactionTriggerAndWatermark(last *llm.Completion, additionalTokens ...int) (string, string) {
	if a.compactor == nil || last == nil {
		return "usage", "summarize"
	}
	return a.compactionTriggerAndWatermarkForUsage(a.effectiveCompactionUsage(last.Usage, additionalTokens...))
}

func (a *Agent) compactionTriggerAndWatermarkForUsage(usage *llm.Usage) (string, string) {
	if a == nil || a.compactor == nil {
		return "usage", "summarize"
	}
	if a.compactor.IsOverflow(usage) {
		return "overflow", "overflow"
	}
	if watermark := a.compactor.WatermarkForUsage(usage); strings.TrimSpace(watermark) != "" {
		if a.todoCompactionPending.Load() {
			return "todo_checkpoint", watermark
		}
		if a.compactionRetryPending.Load() {
			return "retry_checkpoint", watermark
		}
		return "usage", watermark
	}
	if a.destroyedToolMessageCount() >= defaultDestroyedToolCompactThreshold && a.compactor.PromptTokens(usage) > 0 {
		return "placeholder_pressure", "placeholder_cleanup"
	}
	return "usage", ""
}

func (a *Agent) reconcileCompactionTelemetry(res compaction.Result, before, after []llm.Message, additionalTokens int) compaction.Result {
	if a == nil || a.compactor == nil || !res.Compacted {
		return res
	}
	if additionalTokens < 0 {
		additionalTokens = 0
	}
	res.OriginalTokens = a.compactor.EstimateMessages(before) + additionalTokens
	res.NewTokens = a.compactor.EstimateMessages(after) + additionalTokens
	res.TokenCountSource = compaction.TokenCountSourceEstimate
	return res
}

func (a *Agent) withCompactionTelemetry(res compaction.Result, trigger string, watermark string, usage *llm.Usage) compaction.Result {
	if strings.TrimSpace(trigger) != "" {
		res.Trigger = strings.TrimSpace(trigger)
	}
	if strings.TrimSpace(res.Trigger) == "" {
		res.Trigger = "manual"
	}
	if strings.TrimSpace(res.Watermark) == "" && strings.TrimSpace(watermark) != "" {
		res.Watermark = strings.TrimSpace(watermark)
	}
	if strings.TrimSpace(res.Watermark) == "" {
		res.Watermark = "summarize"
	}
	if res.Compacted && len(res.TiersApplied) == 0 {
		res.TiersApplied = []string{"summarize"}
	}
	if usage != nil {
		res.Usage = cloneUsage(usage)
	}
	if res.NewTokens <= 0 {
		res.NewTokens = res.OriginalTokens
	}
	if (res.OriginalTokens > 0 || res.NewTokens > 0) && strings.TrimSpace(res.TokenCountSource) == "" {
		res.TokenCountSource = compaction.TokenCountSourceEstimate
	}
	return res
}

func (a *Agent) compactionMaxAttempts() int {
	if a.compactor == nil {
		return 1
	}
	retries := a.compactor.Config.CompactionRetries
	if retries <= 0 {
		retries = compaction.DefaultCompactionRetries
	}
	return retries + 1
}

func (a *Agent) compactionRetryDelay(attempt int) time.Duration {
	if a.compactor == nil {
		return 0
	}
	base := a.compactor.Config.CompactionRetryBackoff
	if base <= 0 {
		return 0
	}
	if attempt < 1 {
		attempt = 1
	}
	delay := base
	for i := 1; i < attempt; i++ {
		if delay > (time.Duration(1<<63-1) / 2) {
			return time.Duration(1<<63 - 1)
		}
		delay *= 2
	}
	const maxDelay = 5 * time.Second
	if delay > maxDelay {
		return maxDelay
	}
	return delay
}

func (a *Agent) compactWithRetry(ctx context.Context, messages []llm.Message, req compaction.PipelineRequest) ([]llm.Message, compaction.Result, error) {
	attempts := a.compactionMaxAttempts()
	if attempts <= 0 {
		attempts = 1
	}
	var lastErr error
	var lastRes compaction.Result
	for attempt := 1; attempt <= attempts; attempt++ {
		if err := ctx.Err(); err != nil {
			return messages, lastRes, err
		}
		newMsgs, res, err := a.compactor.CompactPipeline(ctx, a.llm, messages, req)
		if err == nil {
			return newMsgs, res, nil
		}
		lastErr = err
		lastRes = res
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) || ctx.Err() != nil {
			if ctxErr := ctx.Err(); ctxErr != nil {
				return messages, lastRes, ctxErr
			}
			return messages, lastRes, err
		}
		if attempt >= attempts {
			break
		}
		delay := a.compactionRetryDelay(attempt)
		a.warnf("compaction failed (attempt %d/%d): %v", attempt, attempts, err)
		if delay <= 0 {
			continue
		}
		t := time.NewTimer(delay)
		select {
		case <-ctx.Done():
			if !t.Stop() {
				<-t.C
			}
			return messages, lastRes, ctx.Err()
		case <-t.C:
		}
	}
	if lastErr == nil {
		lastErr = errors.New("compaction failed")
	}
	return messages, lastRes, lastErr
}

func (a *Agent) withPreservedSystem(orig []llm.Message, compacted []llm.Message) []llm.Message {
	sys := make([]llm.Message, 0, 1)
	seen := map[string]struct{}{}
	addSystem := func(m llm.Message) {
		if m.Role != llm.RoleSystem {
			return
		}
		sig := systemMessageSignature(m)
		if _, ok := seen[sig]; ok {
			return
		}
		seen[sig] = struct{}{}
		sys = append(sys, m)
	}
	for _, m := range orig {
		addSystem(m)
	}
	for _, m := range compacted {
		addSystem(m)
	}
	if len(sys) == 0 && strings.TrimSpace(a.systemPrompt) != "" {
		addSystem(llm.NewSystemMessage(a.systemPrompt))
	}
	if len(sys) == 0 {
		return compacted
	}
	out := make([]llm.Message, 0, len(sys)+len(compacted))
	out = append(out, sys...)
	for _, m := range compacted {
		if m.Role == llm.RoleSystem {
			continue
		}
		out = append(out, m)
	}
	return out
}

func systemMessageSignature(m llm.Message) string {
	return strings.TrimSpace(m.Name) + "\x1f" + strings.TrimSpace(m.Content.PlainText())
}

func cloneUsage(u *llm.Usage) *llm.Usage {
	return llm.CloneUsage(u)
}

func retryAfterMillis(d time.Duration) int64 {
	if d <= 0 {
		return 0
	}
	ms := int64(d / time.Millisecond)
	if ms <= 0 {
		return 1
	}
	return ms
}

func appendRetryAfterMessage(message string, retryAfter time.Duration) string {
	msg := strings.TrimSpace(message)
	if retryAfter <= 0 {
		return msg
	}
	if msg == "" {
		return fmt.Sprintf("retry after %s", retryAfter)
	}
	return fmt.Sprintf("%s (retry after %s)", msg, retryAfter)
}

func providerErrorKind(status int) string {
	switch status {
	case 429:
		return "rate_limit"
	case 401:
		return "auth"
	case 403:
		return "permission"
	case 408:
		return "timeout"
	case 499:
		return "canceled"
	case 400, 404, 405, 406, 409, 410, 411, 412, 413, 414, 415, 422:
		return "invalid_request"
	}
	if status >= 500 {
		return "provider"
	}
	if status >= 400 {
		return "provider"
	}
	return "provider"
}

func retryableProviderStatus(status int) bool {
	switch status {
	case 408, 409, 425, 429:
		return true
	default:
		return status >= 500 && status <= 599
	}
}

func statusCodeInText(msg string) (int, bool) {
	if msg == "" {
		return 0, false
	}
	for _, code := range []int{400, 401, 403, 408, 409, 422, 425, 429, 500, 501, 502, 503, 504, 529} {
		codeText := fmt.Sprint(code)
		if !containsDelimitedNumber(msg, codeText) {
			continue
		}
		if strings.Contains(msg, "("+codeText+")") ||
			strings.Contains(msg, "["+codeText+"]") ||
			strings.Contains(msg, "status") ||
			strings.Contains(msg, "status_code") ||
			strings.Contains(msg, "statuscode") ||
			strings.Contains(msg, "http") ||
			strings.Contains(msg, "response code") ||
			strings.Contains(msg, "error code") ||
			strings.Contains(msg, "code="+codeText) ||
			strings.Contains(msg, "code "+codeText) {
			return code, true
		}
	}
	return 0, false
}

func containsDelimitedNumber(text, number string) bool {
	if text == "" || number == "" {
		return false
	}
	offset := 0
	for {
		idx := strings.Index(text[offset:], number)
		if idx < 0 {
			return false
		}
		start := offset + idx
		end := start + len(number)
		beforeOK := start == 0 || !isASCIIDigit(text[start-1])
		afterOK := end == len(text) || !isASCIIDigit(text[end])
		if beforeOK && afterOK {
			return true
		}
		offset = start + 1
	}
}

func isASCIIDigit(b byte) bool {
	return b >= '0' && b <= '9'
}

func classifyGenericErrorKind(err error) string {
	if err == nil {
		return "unknown"
	}
	if errors.Is(err, context.Canceled) {
		return "canceled"
	}
	if errors.Is(err, context.DeadlineExceeded) {
		return "timeout"
	}
	var dnsErr *net.DNSError
	if errors.As(err, &dnsErr) {
		if dnsErr.Timeout() {
			return "timeout"
		}
		return "network"
	}
	var netErr net.Error
	if errors.As(err, &netErr) {
		if netErr.Timeout() {
			return "timeout"
		}
		return "network"
	}
	var syntaxErr *json.SyntaxError
	if errors.As(err, &syntaxErr) {
		return "decode"
	}
	var typeErr *json.UnmarshalTypeError
	if errors.As(err, &typeErr) {
		return "decode"
	}
	msg := strings.ToLower(strings.TrimSpace(err.Error()))
	if msg == "" {
		return "unknown"
	}
	if strings.Contains(msg, "rate limit") ||
		strings.Contains(msg, "too many requests") {
		return "rate_limit"
	}
	if status, ok := statusCodeInText(msg); ok {
		return providerErrorKind(status)
	}
	switch {
	case strings.Contains(msg, "context canceled"):
		return "canceled"
	case strings.Contains(msg, "context deadline exceeded"),
		strings.Contains(msg, "deadline exceeded"),
		strings.Contains(msg, "timed out"),
		strings.Contains(msg, "timeout"):
		return "timeout"
	case strings.Contains(msg, "invalid character") ||
		strings.Contains(msg, "cannot unmarshal") ||
		strings.Contains(msg, "unexpected end of json") ||
		(strings.Contains(msg, "decode") && strings.Contains(msg, "json")):
		return "decode"
	case strings.Contains(msg, "no such host") ||
		strings.Contains(msg, "name resolution") ||
		strings.Contains(msg, "connection refused") ||
		strings.Contains(msg, "connection reset") ||
		strings.Contains(msg, "network is unreachable") ||
		strings.Contains(msg, "tls:") ||
		strings.Contains(msg, "x509") ||
		strings.Contains(msg, "proxy") ||
		strings.Contains(msg, "unsupported protocol scheme") ||
		strings.Contains(msg, "missing protocol scheme") ||
		strings.Contains(msg, "dial tcp") ||
		strings.Contains(msg, "eof"):
		return "network"
	case strings.Contains(msg, "bad gateway") ||
		strings.Contains(msg, "gateway timeout") ||
		strings.Contains(msg, "internal server error") ||
		strings.Contains(msg, "server error") ||
		strings.Contains(msg, "service unavailable") ||
		strings.Contains(msg, "temporarily unavailable") ||
		strings.Contains(msg, "overloaded") ||
		strings.Contains(msg, "try again later") ||
		strings.Contains(msg, "please retry"):
		return "provider"
	default:
		return "unknown"
	}
}

func (a *Agent) errEvent(err error) ErrorEvent {
	prov := ""
	if a != nil && a.llm != nil {
		prov = strings.TrimSpace(a.llm.Provider())
	}
	if err == nil {
		return ErrorEvent{Provider: prov, Message: "<nil>", Kind: "unknown"}
	}
	var rl *llm.RateLimitError
	if errors.As(err, &rl) {
		rp := strings.TrimSpace(rl.Provider)
		if rp == "" {
			rp = prov
		}
		return ErrorEvent{
			Provider:     rp,
			StatusCode:   429,
			Message:      appendRetryAfterMessage(rl.Message, rl.RetryAfter),
			RetryAfterMS: retryAfterMillis(rl.RetryAfter),
			Kind:         "rate_limit",
		}
	}
	var pe *llm.ProviderError
	if errors.As(err, &pe) {
		pp := strings.TrimSpace(pe.Provider)
		if pp == "" {
			pp = prov
		}
		return ErrorEvent{
			Provider:     pp,
			StatusCode:   pe.StatusCode,
			Message:      appendRetryAfterMessage(pe.Message, pe.RetryAfter),
			RetryAfterMS: retryAfterMillis(pe.RetryAfter),
			Kind:         providerErrorKind(pe.StatusCode),
		}
	}
	return ErrorEvent{Provider: prov, Message: err.Error(), Kind: classifyGenericErrorKind(err)}
}

// drainSteering non-blockingly reads all pending messages from the steering channel
// and appends them to the conversation history as user messages. This allows users
// to inject new instructions at natural breakpoints (tool-call boundaries) without
// blocking the agent's execution loop.
//
// The function returns immediately if the channel is nil or empty.
// Each received message triggers a SteeringReceivedEvent to notify the CLI layer.
func (a *Agent) drainSteering(ch <-chan SteeringMsg, out *eventOutput) int {
	messages := a.collectSteering(ch, out)
	if len(messages) == 0 {
		return 0
	}
	a.appendMessages(messages)
	return len(messages)
}

// collectSteering drains the pending steering messages and emits their events
// without appending them to history. Callers that are in the middle of an
// assistant tool-call block must first complete the tool_result block and only
// then append the returned messages: a provider rejects a tool_use block whose
// tool results are interleaved with user text.
func (a *Agent) collectSteering(ch <-chan SteeringMsg, out *eventOutput) []llm.Message {
	if ch == nil {
		return nil
	}
	var messages []llm.Message
	for {
		msg, ok := takeNextSteering(ch)
		if !ok {
			return messages
		}
		messages = append(messages, llm.Message{
			Role:    llm.RoleUser,
			Content: llm.TextContent(msg.Content),
		})
		a.emitEvent(out, SteeringReceivedEvent{Content: msg.Content})
	}
}

// appendMessages appends framework-authored messages to history under the
// history lock. It is a no-op for an empty batch.
func (a *Agent) appendMessages(messages []llm.Message) {
	if a == nil || len(messages) == 0 {
		return
	}
	a.mu.Lock()
	a.messages = append(a.messages, messages...)
	a.mu.Unlock()
}

func takeNextSteering(ch <-chan SteeringMsg) (SteeringMsg, bool) {
	if ch == nil {
		return SteeringMsg{}, false
	}
	for {
		select {
		case msg, ok := <-ch:
			if !ok {
				return SteeringMsg{}, false
			}
			if strings.TrimSpace(msg.Content) == "" {
				continue
			}
			return msg, true
		default:
			return SteeringMsg{}, false
		}
	}
}

// wrapInvalidToolArgs constructs arguments for the "invalid" fallback tool,
// including the original tool name and any original arguments.
func wrapInvalidToolArgs(toolName, originalArgs string) string {
	m := map[string]any{"tool": toolName}
	var orig map[string]any
	if json.Unmarshal([]byte(originalArgs), &orig) == nil {
		for k, v := range orig {
			if k != "tool" {
				m[k] = v
			}
		}
	} else if strings.TrimSpace(originalArgs) != "" {
		m["original_args"] = originalArgs
	}
	b, _ := json.Marshal(m)
	return string(b)
}

func unknownToolDiagnostic(toolName string) string {
	summary := "Unknown tool requested"
	if name := strings.TrimSpace(toolName); name != "" {
		summary = fmt.Sprintf("Unknown tool %q requested", name)
	}
	return fmt.Sprintf("[ERROR] %s - Use one of the available tools listed in this session and retry.", summary)
}

// autoInvalidTool returns an internal fallback tool for handling unknown tool calls.
// This tool is NOT exposed to the model in tool definitions.
func autoInvalidTool() tools.Tool {
	return tools.Tool{
		Name: "invalid",
		Handler: func(ctx context.Context, raw json.RawMessage, _ *tools.Container) (llm.Content, error) {
			var m map[string]any
			if json.Unmarshal(raw, &m) == nil {
				if toolName, ok := m["tool"].(string); ok {
					name := strings.TrimSpace(toolName)
					meta := map[string]any{"error_kind": "tool_not_found"}
					if name != "" {
						meta["tool"] = name
					}
					tools.UpsertToolResultMetadata(ctx, meta)
					if name != "" {
						return llm.TextContent(unknownToolDiagnostic(name)), fmt.Errorf("tool not found: %s", name)
					}
				}
			}
			tools.UpsertToolResultMetadata(ctx, map[string]any{"error_kind": "tool_not_found"})
			return llm.TextContent(unknownToolDiagnostic("")),
				fmt.Errorf("tool not found")
		},
	}
}

// --- Tool call continuation helpers ---

// toolCallContinuation tracks partial tool calls across auto-continue boundaries.
type toolCallContinuation struct {
	partialCalls     []llm.ToolCall
	msgIndices       []int
	mergeDiagnostics map[string][]string
	turns            int
	maxTurns         int
	// exhausted records that the turn budget was already spent and the partial
	// tool calls were discarded. reset() deliberately does not clear it: the
	// budget must not silently rearm, or every following truncated response
	// appends another tool_use block that can never receive a tool_result.
	exhausted bool
}

func newToolCallContinuation(maxTurns int) toolCallContinuation {
	if maxTurns <= 0 {
		maxTurns = defaultMaxContinuationTurns
	}
	return toolCallContinuation{maxTurns: maxTurns}
}

func (c *toolCallContinuation) hasPending() bool {
	return len(c.partialCalls) > 0
}

func (c *toolCallContinuation) nextTurn() (turn int, allowed bool) {
	if c == nil {
		return 0, false
	}
	if c.maxTurns <= 0 {
		c.maxTurns = defaultMaxContinuationTurns
	}
	c.turns++
	if c.exhausted {
		return c.turns, false
	}
	return c.turns, c.turns <= c.maxTurns
}

// rearm re-enables the continuation budget after a response that completed
// normally, i.e. one that was not truncated by the provider.
func (c *toolCallContinuation) rearm() {
	if c == nil {
		return
	}
	c.exhausted = false
	c.turns = 0
}

// exhaust marks the continuation budget as spent. It survives reset() so a
// later truncated response cannot start a fresh continuation episode and append
// another tool_use block that can never be paired with a tool_result.
func (c *toolCallContinuation) exhaust() {
	if c == nil {
		return
	}
	c.exhausted = true
}

func (c *toolCallContinuation) reset() {
	if c == nil {
		return
	}
	c.partialCalls = nil
	c.msgIndices = nil
	c.mergeDiagnostics = nil
	c.turns = 0
}

// discardPartialToolCalls abandons an unfinished continuation. Unlike reset it
// also strips the partial tool_use blocks from history: the continuation limit
// path appends a user reminder next, which would leave assistant messages whose
// tool_use blocks can never receive a tool_result and make every subsequent
// provider request invalid.
func (c *toolCallContinuation) discardPartialToolCalls(messages []llm.Message) {
	if c == nil {
		return
	}
	c.clearPartialToolCalls(messages, len(messages))
}

// discardContinuationToolCalls abandons an unfinished tool-call continuation.
// Besides clearing the continuation state it strips the unfinished tool_use
// blocks from history — both the accumulated partials and the assistant message
// that was just appended. The continuation-limit paths append a user reminder
// next, so leaving those tool_use blocks in place would produce assistant
// messages whose tool calls can never receive a tool_result, making every
// subsequent provider request permanently invalid.
//
// The budget is marked exhausted rather than reset: otherwise the very next
// truncated response would start a fresh continuation episode and append
// another unpairable tool_use block, permanently malforming the history again.
func (a *Agent) discardContinuationToolCalls(cont *toolCallContinuation, currentIndex int) {
	a.mu.Lock()
	cont.discardPartialToolCalls(a.messages)
	if currentIndex >= 0 && currentIndex < len(a.messages) && a.messages[currentIndex].Role == llm.RoleAssistant {
		a.messages[currentIndex].ToolCalls = nil
		a.messages[currentIndex].Content = llm.WithoutProviderState(a.messages[currentIndex].Content)
	}
	a.mu.Unlock()
	cont.reset()
	cont.exhaust()
}

func (c *toolCallContinuation) addPartial(msgIndex int, calls []llm.ToolCall) {
	if len(c.partialCalls) == 0 {
		c.partialCalls = cloneToolCalls(calls)
	} else {
		// Accumulate: merge by ID
		for _, tc := range calls {
			found := false
			for i, p := range c.partialCalls {
				if sameStableToolCallID(p.ID, tc.ID) {
					merged := mergeToolArgsWithDiagnostics(p.Function.Arguments, tc.Function.Arguments)
					c.partialCalls[i].Function.Arguments = merged.arguments
					c.recordMergeDiagnostics(tc.ID, merged.diagnostics)
					found = true
					break
				}
			}
			if !found {
				c.partialCalls = append(c.partialCalls, tc)
			}
		}
	}
	c.msgIndices = append(c.msgIndices, msgIndex)
}

func (c *toolCallContinuation) setAccumulated(calls []llm.ToolCall, msgIndex int) {
	c.partialCalls = cloneToolCalls(calls)
	c.msgIndices = append(c.msgIndices, msgIndex)
}

func (c *toolCallContinuation) mergeToolCalls(current []llm.ToolCall) []llm.ToolCall {
	result := make([]llm.ToolCall, 0, len(current))
	for _, tc := range current {
		merged := tc
		for _, p := range c.partialCalls {
			if sameStableToolCallID(p.ID, tc.ID) {
				mergedArgs := mergeToolArgsWithDiagnostics(p.Function.Arguments, tc.Function.Arguments)
				merged.Function.Arguments = mergedArgs.arguments
				c.recordMergeDiagnostics(tc.ID, mergedArgs.diagnostics)
				if merged.Function.Name == "" {
					merged.Function.Name = p.Function.Name
				}
				break
			}
		}
		result = append(result, merged)
	}
	return result
}

func (c *toolCallContinuation) clearPartialToolCalls(messages []llm.Message, currentIndex int) {
	if len(c.partialCalls) == 0 {
		return
	}
	for i := 0; i+1 < currentIndex && i+1 < len(messages); i++ {
		if messages[i].Role != llm.RoleAssistant || len(messages[i].ToolCalls) == 0 {
			continue
		}
		// The SDK reminder owns the preceding unfinished block. IDs cannot
		// identify the episode: providers may rotate them between fragments,
		// while synthetic IDs are reused by later completed responses.
		if messages[i+1].Role != llm.RoleUser || messages[i+1].Name != messageorigin.Name(messageorigin.KindToolCallContinuation) {
			continue
		}
		messages[i].ToolCalls = nil
		messages[i].Content = llm.WithoutProviderState(messages[i].Content)
	}
	c.reset()
}

func (c *toolCallContinuation) recordMergeDiagnostics(callID string, diagnostics []string) {
	callID = strings.TrimSpace(callID)
	if callID == "" || len(diagnostics) == 0 {
		return
	}
	if c.mergeDiagnostics == nil {
		c.mergeDiagnostics = map[string][]string{}
	}
	c.mergeDiagnostics[callID] = appendUniqueStrings(c.mergeDiagnostics[callID], diagnostics...)
}

func (c *toolCallContinuation) mergeDiagnosticsForCalls(calls []llm.ToolCall) map[string][]string {
	if len(calls) == 0 || len(c.mergeDiagnostics) == 0 {
		return nil
	}
	out := map[string][]string{}
	for _, tc := range calls {
		id := strings.TrimSpace(tc.ID)
		if id == "" {
			continue
		}
		if diagnostics := c.mergeDiagnostics[id]; len(diagnostics) > 0 {
			out[id] = append([]string(nil), diagnostics...)
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

type toolArgsMergeResult struct {
	arguments   string
	diagnostics []string
}

// mergeToolArgs merges two tool argument strings.
//
// Priority order:
//  1. Stitch overlapping fragments when partial JSON chunks can form valid JSON.
//  2. Deep-merge decoded JSON values (objects recursively, arrays by index).
//  3. Fall back to overlap-based concatenation for non-JSON fragments.
func mergeToolArgs(old, new string) string {
	return mergeToolArgsWithDiagnostics(old, new).arguments
}

func mergeToolArgsWithDiagnostics(old, new string) toolArgsMergeResult {
	if strings.TrimSpace(old) == "" {
		return toolArgsMergeResult{arguments: new}
	}
	if strings.TrimSpace(new) == "" {
		return toolArgsMergeResult{arguments: old}
	}

	if stitched, ok := stitchToolArgFragments(old, new); ok {
		return toolArgsMergeResult{arguments: stitched}
	}

	var oldValue, newValue any
	if json.Unmarshal([]byte(old), &oldValue) == nil && json.Unmarshal([]byte(new), &newValue) == nil {
		mergedValue, diagnostics := deepMergeJSONValue(oldValue, newValue, "$")
		if marshaled, err := json.Marshal(mergedValue); err == nil {
			return toolArgsMergeResult{arguments: string(marshaled), diagnostics: diagnostics}
		}
		return toolArgsMergeResult{
			arguments:   mergeArgsByOverlap(old, new),
			diagnostics: append(diagnostics, "failed to serialize deep-merged JSON arguments; used fragment fallback"),
		}
	}

	return toolArgsMergeResult{arguments: mergeArgsByOverlap(old, new)}
}

func stitchToolArgFragments(old, new string) (string, bool) {
	stitched := mergeArgsByOverlap(old, new)
	if !json.Valid([]byte(stitched)) {
		return "", false
	}
	oldValid := json.Valid([]byte(strings.TrimSpace(old)))
	newValid := json.Valid([]byte(strings.TrimSpace(new)))
	if oldValid && newValid {
		return "", false
	}
	return stitched, true
}

func deepMergeJSONValue(oldValue, newValue any, path string) (any, []string) {
	oldObject, oldIsObject := oldValue.(map[string]any)
	newObject, newIsObject := newValue.(map[string]any)
	if oldIsObject && newIsObject {
		merged := make(map[string]any, len(oldObject))
		for k, v := range oldObject {
			merged[k] = v
		}
		diagnostics := make([]string, 0)
		for k, nextValue := range newObject {
			childPath := joinJSONPath(path, k)
			if current, ok := merged[k]; ok {
				mergedValue, childDiagnostics := deepMergeJSONValue(current, nextValue, childPath)
				merged[k] = mergedValue
				diagnostics = append(diagnostics, childDiagnostics...)
				continue
			}
			merged[k] = nextValue
		}
		return merged, diagnostics
	}
	if oldIsObject != newIsObject {
		return newValue, []string{fmt.Sprintf("%s changed shape from %s to %s", path, jsonValueKind(oldValue), jsonValueKind(newValue))}
	}

	oldArray, oldIsArray := oldValue.([]any)
	newArray, newIsArray := newValue.([]any)
	if oldIsArray && newIsArray {
		return mergeJSONArrayByIndex(oldArray, newArray, path)
	}
	if oldIsArray != newIsArray {
		return newValue, []string{fmt.Sprintf("%s changed shape from %s to %s", path, jsonValueKind(oldValue), jsonValueKind(newValue))}
	}

	return newValue, nil
}

func mergeJSONArrayByIndex(oldArray, newArray []any, path string) ([]any, []string) {
	maxLen := len(oldArray)
	if len(newArray) > maxLen {
		maxLen = len(newArray)
	}
	merged := make([]any, 0, maxLen)
	diagnostics := make([]string, 0)
	if len(oldArray) != len(newArray) {
		diagnostics = append(diagnostics,
			fmt.Sprintf("%s array length mismatch old=%d new=%d; preserved unmatched elements by index", path, len(oldArray), len(newArray)),
		)
	}
	for i := 0; i < maxLen; i++ {
		childPath := fmt.Sprintf("%s[%d]", path, i)
		switch {
		case i >= len(oldArray):
			merged = append(merged, newArray[i])
		case i >= len(newArray):
			merged = append(merged, oldArray[i])
		default:
			mergedValue, childDiagnostics := deepMergeJSONValue(oldArray[i], newArray[i], childPath)
			merged = append(merged, mergedValue)
			diagnostics = append(diagnostics, childDiagnostics...)
		}
	}
	return merged, diagnostics
}

func jsonValueKind(v any) string {
	switch v.(type) {
	case map[string]any:
		return "object"
	case []any:
		return "array"
	case string:
		return "string"
	case float64:
		return "number"
	case bool:
		return "boolean"
	case nil:
		return "null"
	default:
		return fmt.Sprintf("%T", v)
	}
}

func joinJSONPath(parent, key string) string {
	if parent == "" || parent == "$" {
		return "$." + key
	}
	return parent + "." + key
}

func mergeArgsByOverlap(old, new string) string {
	maxOverlap := len(old)
	if len(new) < maxOverlap {
		maxOverlap = len(new)
	}
	for overlap := maxOverlap; overlap > 0; overlap-- {
		if old[len(old)-overlap:] == new[:overlap] {
			return old + new[overlap:]
		}
	}
	return old + new
}

func sameStableToolCallID(oldID, newID string) bool {
	oldID = strings.TrimSpace(oldID)
	newID = strings.TrimSpace(newID)
	return oldID != "" && oldID == newID
}

func appendUniqueStrings(dst []string, values ...string) []string {
	for _, value := range values {
		value = strings.TrimSpace(value)
		if value == "" {
			continue
		}
		already := false
		for _, existing := range dst {
			if existing == value {
				already = true
				break
			}
		}
		if !already {
			dst = append(dst, value)
		}
	}
	return dst
}

func appendToolArgMergeDiagnostics(meta map[string]any, diagnostics []string) map[string]any {
	if len(diagnostics) == 0 {
		return meta
	}
	if meta == nil {
		meta = map[string]any{}
	}
	var existing []string
	if raw, ok := meta["tool_arg_merge_conflicts"]; ok {
		switch v := raw.(type) {
		case string:
			existing = append(existing, v)
		case []string:
			existing = append(existing, v...)
		case []any:
			for _, item := range v {
				if s, ok := item.(string); ok {
					existing = append(existing, s)
				}
			}
		}
	}
	meta["tool_arg_merge_conflicts"] = appendUniqueStrings(existing, diagnostics...)
	return meta
}

func allToolArgsValid(calls []llm.ToolCall) bool {
	for _, tc := range calls {
		args := strings.TrimSpace(tc.Function.Arguments)
		if args == "" || args == "{}" {
			continue
		}
		if !json.Valid([]byte(args)) {
			return false
		}
	}
	return true
}

func cloneToolCalls(calls []llm.ToolCall) []llm.ToolCall {
	if calls == nil {
		return nil
	}
	out := make([]llm.ToolCall, len(calls))
	copy(out, calls)
	return out
}

func sameToolCalls(a, b []llm.ToolCall) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i].ID != b[i].ID || a[i].Function.Name != b[i].Function.Name || a[i].Function.Arguments != b[i].Function.Arguments {
			return false
		}
	}
	return true
}
