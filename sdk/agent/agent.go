package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net"
	"os"
	"runtime/debug"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unicode/utf8"

	"github.com/timwhitez/agent-sdk-golang/sdk/agent/compaction"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

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
	// LoopGuardStrikeThreshold controls how many repeated-signature strikes are allowed
	// before the run is aborted with a doom_loop error.
	// Values <= 0 use a safe default when RepeatToolSignatureThreshold is enabled.
	LoopGuardStrikeThreshold int
	// LoopGuardUserMessage is injected into conversation history when a repeated-signature
	// strike is detected so the model can correct course. Empty uses a safe default.
	LoopGuardUserMessage string
	// MaxToolResultBytes bounds tool result text stored in history and emitted in ToolResultEvent.
	// Values <= 0 use a safe default.
	MaxToolResultBytes int
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
	llm                llm.ChatModel
	systemPrompt       string
	maxIterations      int
	invokeRetryMax     int
	invokeRetryBackoff time.Duration
	repeatSigThreshold int
	repeatSigWindow    int
	loopGuardStrikeMax int
	loopGuardUserMsg   string
	maxToolResultBytes int
	toolResultDumpTTL  time.Duration
	eventBufferSize    int
	eventSendTimeout   time.Duration
	eventDropLogEvery  uint64
	streamIdleTimeout  time.Duration
	streamIdleMaxRecov int
	toolChoice         llm.ToolChoice
	requireDone        bool
	warningf           func(format string, args ...any)
	hasCompactor       bool

	tools             []tools.Tool
	toolMap           map[string]tools.Tool
	toolMapNormalized map[string]tools.Tool
	deps              *tools.Container

	compactor *compaction.Service

	todoCompactionPending atomic.Bool
	compactionInFlight    atomic.Bool

	pendingCompactionMu sync.Mutex
	pendingCompaction   *pendingCompaction

	mu              sync.Mutex
	messages        []llm.Message
	lastPromptCount int
	// Ephemeral cleanup state avoids full-history scans on every loop.
	ephemeralByTool   map[string][]int
	ephemeralScanFrom int

	toolResultDumpsMu sync.Mutex
	toolResultDumps   map[string]toolResultDumpLifecycleEntry
	toolResultDumpDir string
	toolResultDumpID  string
	toolResultDumpIdx string

	eventDropCount atomic.Uint64
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
	requireDoneReminderText        = "Task completion must use the done tool. If the task is complete, call done with a concise completion message. Do not end with text-only completion claims."
	defaultLoopGuardUserMsg        = "You are repeating the same tool call with identical arguments. Stop repeating, reuse prior results, adjust arguments, or call done if the task is complete."
	earlyStopReminderText          = "You already used tools in this run. Before stopping, verify the task is complete and call done with a concise completion message."
	streamIdleRecoveryText         = "The previous response stream stalled before completion. Continue from the current conversation state. Do not repeat completed tool calls unless needed. If you were mid-analysis or mid-sentence, continue exactly where you left off. If enough information is already available, complete the task."
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

	toolMap := map[string]tools.Tool{}
	for _, t := range cfg.Tools {
		if t.Name == "" {
			return nil, fmt.Errorf("agent: tool missing name")
		}
		toolMap[t.Name] = t
	}

	compSvc := compaction.NewService(cfg.Compaction)
	hasCompactor := compSvc != nil && compSvc.Config.Enabled
	if !hasCompactor {
		compSvc = nil
	}

	ag := &Agent{
		llm:                cfg.LLM,
		systemPrompt:       cfg.SystemPrompt,
		maxIterations:      cfg.MaxIterations,
		invokeRetryMax:     cfg.InvokeRetryMaxAttempts,
		invokeRetryBackoff: cfg.InvokeRetryBackoff,
		repeatSigThreshold: cfg.RepeatToolSignatureThreshold,
		repeatSigWindow:    cfg.RepeatToolSignatureWindow,
		loopGuardStrikeMax: cfg.LoopGuardStrikeThreshold,
		loopGuardUserMsg:   cfg.LoopGuardUserMessage,
		maxToolResultBytes: cfg.MaxToolResultBytes,
		toolResultDumpTTL:  cfg.ToolResultDumpTTL,
		eventBufferSize:    cfg.EventBufferSize,
		eventSendTimeout:   cfg.EventSendTimeout,
		eventDropLogEvery:  uint64(cfg.EventDropLogEvery),
		streamIdleTimeout:  cfg.StreamIdleTimeout,
		streamIdleMaxRecov: cfg.StreamIdleMaxRecoveries,
		toolChoice:         cfg.ToolChoice,
		requireDone:        cfg.RequireDoneTool,
		warningf:           cfg.Warningf,
		hasCompactor:       hasCompactor,
		tools:              append([]tools.Tool(nil), cfg.Tools...),
		toolMap:            toolMap,
		toolMapNormalized:  buildNormalizedToolMap(toolMap, cfg.Tools),
		deps:               cfg.Deps,
		compactor:          compSvc,
		toolResultDumps:    make(map[string]toolResultDumpLifecycleEntry),
		ephemeralByTool:    make(map[string][]int),
	}
	if len(cfg.InitialMessages) > 0 {
		ag.messages = append([]llm.Message(nil), cfg.InitialMessages...)
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

// UpdateCompactionConfig replaces the compaction service used for subsequent turns.
// Callers should prefer updating compaction between turns when the agent is idle.
func (a *Agent) UpdateCompactionConfig(cfg *compaction.Config) {
	if a == nil {
		return
	}
	compSvc := compaction.NewService(cfg)
	hasCompactor := compSvc != nil && compSvc.Config.Enabled
	if !hasCompactor {
		compSvc = nil
		a.todoCompactionPending.Store(false)
		a.pendingCompactionMu.Lock()
		a.pendingCompaction = nil
		a.pendingCompactionMu.Unlock()
	}
	a.compactor = compSvc
	a.hasCompactor = hasCompactor
}

func (a *Agent) Messages() []llm.Message {
	a.mu.Lock()
	defer a.mu.Unlock()
	cpy := make([]llm.Message, len(a.messages))
	copy(cpy, a.messages)
	return cpy
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
	a.messages = append([]llm.Message(nil), messages...)
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

// QueryStreamWithSteering is like QueryStream but accepts an optional steering channel.
// When steeringCh is non-nil, the agent checks for new user messages at natural breakpoints
// (before each LLM invocation and after each tool execution). Any received steering messages
// are appended to the conversation history as user messages, so the next LLM call will
// see them and can adjust its plan accordingly.
//
// The steering channel is caller-owned. The agent only reads from it and never closes it.
func (a *Agent) QueryStreamWithSteering(ctx context.Context, input llm.Content, steeringCh <-chan SteeringMsg) <-chan Event {
	bufferSize := defaultEventBufferSize
	if a != nil && a.eventBufferSize > 0 {
		bufferSize = a.eventBufferSize
	}
	out := make(chan Event, bufferSize)
	go func() {
		defer close(out)
		a.cleanupToolResultDumps(toolResultDumpNow(), false)

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
		seenToolCallHistory := false
		lastResponseID := ""
		pendingTextContinuation := ""
		pendingRequireDoneFinalText := ""
		pendingRequireDoneFinalResponseID := ""
		streamIdleRecoveries := 0
		streamIdleRecoveryTotal := 0
		cont := newToolCallContinuation(defaultMaxContinuationTurns)
		repeatGuard := newRepeatedToolSignatureGuard(a.repeatSigThreshold, a.repeatSigWindow)
		loopGuardStrikes := 0
		hasDoneTool := a.hasToolNamed("done")
		emitFinal := func(content, responseID string) {
			a.emitEvent(out, FinalResponseEvent{
				Content:         content,
				ResponseID:      responseID,
				StallRecoveries: streamIdleRecoveryTotal,
			})
		}
		emitErr := func(e ErrorEvent) {
			e.StallRecoveries = streamIdleRecoveryTotal
			a.emitEvent(out, e)
		}

		// maxIterations < 0 means unlimited: the loop is then bounded only by
		// tool-loop guards, idle detection, and context cancellation.
		unlimitedIter := a.maxIterations < 0
		for iter := 0; unlimitedIter || iter < a.maxIterations; iter++ {
			a.applyPendingCompaction(out)

			// *** Boundary-aware steering: check for new user messages before each LLM call ***
			a.drainSteering(steeringCh, out)

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

			comp, streamedText, err := a.invokeCompletionWithRetryAndSteering(ctx, llm.InvokeRequest{
				Messages:   messages,
				Tools:      toolDefs,
				ToolChoice: a.toolChoice,
			}, out, steeringCh)
			if err != nil {
				// Check for steering interrupt - handle specially
				var steerErr *llm.SteeringInterruptError
				if errors.As(err, &steerErr) {
					// Save partial assistant message if any text was streamed.
					if comp != nil && !comp.Content.IsEmpty() {
						a.mu.Lock()
						a.messages = append(a.messages, llm.Message{
							Role:    llm.RoleAssistant,
							Content: comp.Content,
						})
						a.mu.Unlock()
					}
					if msg := strings.TrimSpace(steerErr.Message); msg != "" {
						a.mu.Lock()
						a.messages = append(a.messages, llm.NewUserMessage(msg))
						a.mu.Unlock()
					}
					// Emit the steering received event and continue loop.
					a.emitEvent(out, SteeringReceivedEvent{Content: steerErr.Message})
					continue
				}

				var idleErr *llm.StreamIdleTimeoutError
				if errors.As(err, &idleErr) {
					maxRecov := agentStreamIdleMaxRecoveries
					if a != nil && a.streamIdleMaxRecov > 0 {
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
						a.mu.Lock()
						a.messages = append(a.messages, llm.NewUserMessage(streamIdleRecoveryText))
						a.mu.Unlock()
						log.Printf(
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
				emitErr(a.errEvent(err))
				return
			}
			streamIdleRecoveries = 0
			responseID := strings.TrimSpace(comp.ResponseID)
			if responseID != "" {
				lastResponseID = responseID
			}
			comp.ToolCalls = ensureSyntheticToolCallIDs(comp.ToolCalls)

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
				a.emitEvent(out, UsageEvent{Usage: *comp.Usage, ResponseID: responseID})
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
			a.messages = append(a.messages, llm.Message{Role: llm.RoleAssistant, Content: comp.Content, ToolCalls: comp.ToolCalls})
			msgIndex := len(a.messages) - 1
			a.mu.Unlock()

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
					cont.reset()
					a.mu.Lock()
					a.messages = append(a.messages, llm.Message{
						Role:    llm.RoleUser,
						Content: llm.TextContent("Your tool-call arguments were repeatedly truncated. Split the work into smaller tool calls and continue."),
					})
					a.mu.Unlock()
					continue
				}
				cont.addPartial(msgIndex, comp.ToolCalls)
				a.emitEvent(out, WarnEvent{
					Message: fmt.Sprintf("continuing truncated tool-call arguments (%d/%d)", turn, cont.maxTurns),
					Kind:    "continuation",
				})
				a.mu.Lock()
				a.messages = append(a.messages, llm.Message{
					Role:    llm.RoleUser,
					Content: llm.TextContent("Your response was truncated. Please continue exactly where you left off."),
				})
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
						cont.reset()
						a.mu.Lock()
						a.messages = append(a.messages, llm.Message{
							Role:    llm.RoleUser,
							Content: llm.TextContent("Tool-call arguments are still invalid after continuation. Split the work into smaller tool calls and continue."),
						})
						a.mu.Unlock()
						continue
					}
					// Still invalid JSON — keep accumulating.
					cont.setAccumulated(merged, msgIndex)
					a.emitEvent(out, WarnEvent{
						Message: fmt.Sprintf("tool-call merge remained invalid; requesting continuation (%d/%d)", turn, cont.maxTurns),
						Kind:    "continuation",
					})
					a.mu.Lock()
					a.messages = append(a.messages, llm.Message{
						Role:    llm.RoleUser,
						Content: llm.TextContent("Your response was truncated. Please continue exactly where you left off."),
					})
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
					if a.hasCompactor {
						a.checkAndCompact(ctx, comp, out)
					}
					a.mu.Lock()
					a.messages = append(a.messages, llm.Message{
						Role:    llm.RoleUser,
						Content: llm.TextContent("Your response was truncated. Please continue exactly where you left off."),
					})
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
						a.mu.Lock()
						a.messages = append(a.messages, llm.Message{
							Role:    llm.RoleUser,
							Content: llm.TextContent(earlyStopReminderText),
						})
						a.mu.Unlock()
						continue
					}
					// compaction check
					if a.hasCompactor {
						a.checkAndCompact(ctx, comp, out)
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
							a.checkAndCompact(ctx, comp, out)
						}
						clearPendingTextContinuation()
						a.emitEvent(out, FinalResponseEvent{Content: combinedText, ResponseID: responseID})
						return
					}
					// Tools were used earlier; enforce done-tool completion.
					if txt := strings.TrimSpace(combinedText); txt != "" && strings.TrimSpace(pendingRequireDoneFinalText) == "" {
						pendingRequireDoneFinalText = txt
						pendingRequireDoneFinalResponseID = responseID
					}
					requireDoneReminders++
					if !requireDoneReminderLogged {
						log.Printf("warning: RequireDoneTool is true but model stopped with text-only after tool usage; prompting done-tool reminder")
						requireDoneReminderLogged = true
					}
					// Safety valve: cap consecutive reminders to prevent runaway loops.
					if requireDoneReminders > defaultRequireDoneMaxReminders {
						a.emitEvent(out, WarnEvent{
							Message: "require-done safety valve: model produced text-only responses without calling done tool; auto-terminating to prevent loop",
							Kind:    "require_done_safety",
						})
						if a.hasCompactor {
							a.checkAndCompact(ctx, comp, out)
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
						emitFinal(finalContent, finalResponseID)
						return
					}
					if a.hasCompactor {
						a.checkAndCompact(ctx, comp, out)
					}
					a.mu.Lock()
					a.messages = append(a.messages, llm.Message{
						Role:    llm.RoleUser,
						Content: llm.TextContent(requireDoneReminderText),
					})
					a.mu.Unlock()
					continue
				}
				continue
			}

			// Execute tool calls with alias resolution and unknown-tool fallback.
			loopGuardTriggered := false
			for idx, tc := range comp.ToolCalls {
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
				if !strings.EqualFold(strings.TrimSpace(resolvedName), "done") {
					pendingRequireDoneFinalText = ""
					pendingRequireDoneFinalResponseID = ""
				}
				if mergeWarnings := continuationMergeDiagnostics[tc.ID]; len(mergeWarnings) > 0 {
					norm.Meta = appendToolArgMergeDiagnostics(norm.Meta, mergeWarnings)
					for _, warning := range mergeWarnings {
						log.Printf("warning: tool-call argument merge conflict for call %q: %s", tc.ID, warning)
					}
				}
				if repeatGuard != nil {
					signature := normalizeToolSignature(resolvedName, norm.Normalized, execArgs)
					if seen, blocked := repeatGuard.observe(signature); blocked {
						loopGuardTriggered = true
						loopGuardStrikes++
						a.appendLoopGuardSkippedToolResults(comp.ToolCalls[idx:], resolvedName)
						reminder := strings.TrimSpace(a.loopGuardUserMsg)
						if reminder != "" {
							a.mu.Lock()
							a.messages = append(a.messages, llm.Message{
								Role:    llm.RoleUser,
								Content: llm.TextContent(reminder),
							})
							a.mu.Unlock()
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
						if a.loopGuardStrikeMax > 0 && loopGuardStrikes >= a.loopGuardStrikeMax {
							// Repeat protection is exhausted. Rather than aborting
							// the whole run (which kills legitimate work — e.g. a
							// long research turn that re-reads a file after context
							// compaction evicted the earlier result), retreat:
							// disable the guard and let subsequent tool calls
							// execute. The run is then bounded only by
							// iteration/idle/compaction/cancel, matching Codex's
							// loop, which has no repeated-call abort.
							a.emitEvent(out, WarnEvent{
								Message: fmt.Sprintf(
									"repeated tool-call loop protection exhausted after %d strike(s); disabling repeat guard and allowing tool execution to proceed",
									loopGuardStrikes,
								),
								Kind: "loop_guard",
							})
							repeatGuard = nil
						}
						break
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

				start := time.Now()
				ctxTool := tools.WithToolCallID(ctx, tc.ID)
				ctxTool = tools.WithToolResultMetadata(ctxTool)
				content, toolErr := a.executeToolSafely(ctxTool, tool, execArgs)
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
				if errors.As(toolErr, &tce) {
					isError = false
					status = "completed"
					content = llm.TextContent("Task completed: " + tce.Message)
					content, meta = a.applyToolResultTruncation(content, meta)
					// append tool message and finish
					a.mu.Lock()
					a.messages = append(a.messages, llm.Message{Role: llm.RoleTool, ToolCallID: tc.ID, ToolName: resolvedName, Content: content, IsError: false, Ephemeral: ephemeral})
					a.mu.Unlock()
					a.emitEvent(out, ToolResultEvent{Tool: resolvedName, Result: content.PlainText(), ToolCallID: tc.ID, IsError: false, Metadata: meta})
					a.emitEvent(out, StepCompleteEvent{StepID: tc.ID, Status: status, DurationMS: time.Since(start).Milliseconds()})
					finalContent := strings.TrimSpace(tce.Message)
					finalResponseID := responseID
					if preserved := strings.TrimSpace(pendingRequireDoneFinalText); preserved != "" {
						finalContent = preserved
						if preservedResponseID := strings.TrimSpace(pendingRequireDoneFinalResponseID); preservedResponseID != "" {
							finalResponseID = preservedResponseID
						}
					}
					emitFinal(finalContent, finalResponseID)
					return
				}

				content, meta = a.applyToolResultTruncation(content, meta)

				// append tool message
				a.mu.Lock()
				a.messages = append(a.messages, llm.Message{Role: llm.RoleTool, ToolCallID: tc.ID, ToolName: resolvedName, Content: content, IsError: isError, Ephemeral: ephemeral})
				a.mu.Unlock()

				a.emitEvent(out, ToolResultEvent{Tool: resolvedName, Result: content.PlainText(), ToolCallID: tc.ID, IsError: isError, Metadata: meta})
				a.emitEvent(out, StepCompleteEvent{StepID: tc.ID, Status: status, DurationMS: time.Since(start).Milliseconds()})

				// *** Boundary-aware steering: check for new user messages after each tool execution ***
				a.drainSteering(steeringCh, out)
			}
			if loopGuardTriggered {
				continue
			}

			if a.hasCompactor {
				a.checkAndCompact(ctx, comp, out)
			}
		}

		// Max iterations reached — emit both error and final events.
		// Unreachable when maxIterations < 0 (unlimited).
		msg := fmt.Sprintf("Max iterations reached (%d)", a.maxIterations)
		emitErr(ErrorEvent{Provider: a.llm.Provider(), Message: msg, Kind: "max_iterations"})
		emitFinal(fmt.Sprintf("[Max iterations reached] %d", a.maxIterations), lastResponseID)
	}()
	return out
}

func (a *Agent) applyToolResultTruncation(content llm.Content, meta map[string]any) (llm.Content, map[string]any) {
	content, meta, dumpPath := truncateToolResultContent(content, meta, a.maxToolResultBytes)
	if dumpPath == "" {
		return content, meta
	}
	now := toolResultDumpNow()
	lifecycle := a.registerToolResultDump(dumpPath, now)
	a.cleanupToolResultDumps(now, false)
	if meta == nil {
		meta = map[string]any{}
	}
	meta["result_output_ttl_ms"] = a.toolResultDumpTTL.Milliseconds()
	meta["result_output_created_at"] = lifecycle.CreatedAt.UTC().Format(time.RFC3339)
	meta["result_output_expires_at"] = lifecycle.ExpiresAt.UTC().Format(time.RFC3339)
	meta["result_output_expiry_policy"] = toolResultDumpExpiryPolicy
	return content, meta
}

func truncateToolResultContent(content llm.Content, meta map[string]any, maxBytes int) (llm.Content, map[string]any, string) {
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
	if err != nil {
		log.Printf("warning: failed to persist full tool result for truncation: %v", err)
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

func (a *toolCallAccumulator) finalize() []llm.ToolCall {
	out := []llm.ToolCall{}
	for i, it := range a.items {
		name := strings.TrimSpace(it.name.String())
		args := strings.TrimSpace(it.args.String())
		if name == "" {
			continue
		}
		id := strings.TrimSpace(it.id)
		if id == "" {
			id = fmt.Sprintf("%s%d", syntheticToolCallIDPrefix, i)
		}
		out = append(out, llm.ToolCall{ID: id, Type: "function", Function: llm.FunctionCall{Name: name, Arguments: args}})
	}
	return out
}

type repeatedToolSignatureGuard struct {
	threshold int
	window    int
	recent    []string
	counts    map[string]int
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
	return len(comp.ToolCalls) > 0
}

type streamMetadataBuffer struct {
	events     []llm.StreamEvent
	usage      *llm.Usage
	responseID string
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
func (a *Agent) invokeCompletion(ctx context.Context, req llm.InvokeRequest, out chan Event) (*llm.Completion, bool, error) {
	return a.invokeCompletionWithSteering(ctx, req, out, nil)
}

// invokeCompletionWithSteering extends invokeCompletion with real-time steering support.
// When steeringCh is non-nil, the stream can be interrupted mid-generation if the user
// sends a steering message. The function returns a SteeringInterruptError in that case,
// allowing the caller to immediately incorporate the steering message into the conversation.
func (a *Agent) invokeCompletionWithSteering(ctx context.Context, req llm.InvokeRequest, out chan Event, steeringCh <-chan SteeringMsg) (*llm.Completion, bool, error) {
	if a == nil || a.llm == nil {
		return nil, false, fmt.Errorf("agent: nil llm")
	}
	if sm, ok := a.llm.(llm.StreamingChatModel); ok {
		invokeCtx, cancelStream := context.WithCancel(ctx)
		defer cancelStream()
		ch, err := sm.InvokeStream(invokeCtx, req)
		if err != nil {
			return nil, false, err
		}
		var text strings.Builder
		var thinking strings.Builder
		acc := &toolCallAccumulator{}
		var usage *llm.Usage
		stopReason := ""
		responseID := ""
		streamedText := false
		emittedVisible := false
		metadata := &streamMetadataBuffer{}
		partialCompletion := func() *llm.Completion {
			content := llm.TextContent(text.String())
			thinkingText := strings.TrimSpace(thinking.String())
			toolCalls := acc.finalize()
			visible := !content.IsEmpty() || thinkingText != "" || len(toolCalls) > 0
			completionUsage := usage
			if completionUsage == nil && !visible {
				completionUsage = metadata.usage
			}
			completionResponseID := responseID
			if strings.TrimSpace(completionResponseID) == "" && !visible {
				completionResponseID = metadata.responseID
			}
			return &llm.Completion{
				Content:    content,
				Thinking:   thinkingText,
				ToolCalls:  toolCalls,
				Usage:      completionUsage,
				StopReason: stopReason,
				ResponseID: completionResponseID,
			}
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
				thinking.WriteString(e.Delta)
				a.emitEvent(out, ThinkingDeltaEvent{Delta: e.Delta})
			case llm.StreamToolCallDeltaEvent:
				acc.apply(e)
			case llm.StreamUsageEvent:
				u := e.Usage
				usage = &u
			case llm.StreamResponseEvent:
				if id := strings.TrimSpace(e.ResponseID); id != "" {
					responseID = id
				}
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
						return partialCompletion(), streamedText, err
					}
					return partialCompletion(), streamedText, nil
				}
				resetIdleTimer()
				if !emittedVisible && metadata.add(ev) {
					continue
				}
				if isVisibleProviderStreamEvent(ev) {
					if err := metadata.flush(processStreamEvent); err != nil {
						return partialCompletion(), streamedText, err
					}
					emittedVisible = true
				}
				if err := processStreamEvent(ev); err != nil {
					return partialCompletion(), streamedText, err
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
					cancelStream()
					drainStreamAsync(ch)
					return partialCompletion(), streamedText, &llm.SteeringInterruptError{Message: msg.Content}
				}

			case <-idleC:
				cancelStream()
				drainStreamAsync(ch)
				return partialCompletion(), streamedText, &llm.StreamIdleTimeoutError{Duration: idleTimeout}
			}
		}
	}
	comp, err := a.llm.Invoke(ctx, req)
	return comp, false, err
}

func (a *Agent) invokeCompletionWithRetry(ctx context.Context, req llm.InvokeRequest, out chan Event) (*llm.Completion, bool, error) {
	return a.invokeCompletionWithRetryAndSteering(ctx, req, out, nil)
}

// invokeCompletionWithRetryAndSteering extends invokeCompletionWithRetry with steering support.
// When a steering interrupt occurs, it immediately returns the partial completion along with
// the SteeringInterruptError, allowing the agent loop to process the steering message.
func (a *Agent) invokeCompletionWithRetryAndSteering(ctx context.Context, req llm.InvokeRequest, out chan Event, steeringCh <-chan SteeringMsg) (*llm.Completion, bool, error) {
	maxAttempts := defaultInvokeRetryMax
	if a != nil && a.invokeRetryMax > 0 {
		maxAttempts = a.invokeRetryMax
	}
	if maxAttempts <= 1 {
		return a.invokeCompletionWithSteering(ctx, req, out, steeringCh)
	}

	var lastComp *llm.Completion
	var lastStreamed bool
	var lastErr error
	for attempt := 1; attempt <= maxAttempts; attempt++ {
		comp, streamedText, err := a.invokeCompletionWithSteering(ctx, req, out, steeringCh)
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
			log.Printf("agent invoke transient failure (attempt %d/%d): %v; retrying in %s", attempt, maxAttempts, err, delay)
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
		log.Printf("agent invoke transient failure (attempt %d/%d): %v; retrying", attempt, maxAttempts, err)
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

func (a *Agent) appendLoopGuardSkippedToolResults(calls []llm.ToolCall, currentResolvedName string) {
	if len(calls) == 0 {
		return
	}
	content := llm.TextContent("[ERROR] Tool call skipped by loop guard - Repeated identical tool call blocked before execution. Reuse previous results, change arguments, or call done if the task is complete.")
	msgs := make([]llm.Message, 0, len(calls))
	for i, tc := range calls {
		id := strings.TrimSpace(tc.ID)
		if id == "" {
			continue
		}
		name := strings.TrimSpace(tc.Function.Name)
		if i == 0 {
			if resolved := strings.TrimSpace(currentResolvedName); resolved != "" {
				name = resolved
			}
		}
		if name == "" {
			name = "unknown"
		}
		msgs = append(msgs, llm.NewToolMessage(id, name, content, true))
	}
	if len(msgs) == 0 {
		return
	}
	a.mu.Lock()
	a.messages = append(a.messages, msgs...)
	a.mu.Unlock()
}

func (a *Agent) executeToolSafely(ctx context.Context, tool tools.Tool, raw string) (content llm.Content, err error) {
	defer func() {
		if recovered := recover(); recovered != nil {
			panicMsg := fmt.Sprintf("tool %q panicked: %v", tool.Name, recovered)
			log.Printf("error: recovered panic from tool %q: %v\n%s", tool.Name, recovered, debug.Stack())
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

func (a *Agent) emitEvent(out chan Event, ev Event) bool {
	if out == nil {
		return false
	}
	select {
	case out <- ev:
		return true
	default:
	}

	timeout := defaultEventSendTimeout
	if a != nil && a.eventSendTimeout > 0 {
		timeout = a.eventSendTimeout
	}
	if isTerminalAgentEvent(ev) {
		if a.tryEnqueueTerminalEvent(out, ev) {
			return true
		}
		if timeout < 250*time.Millisecond {
			timeout = 250 * time.Millisecond
		}
	}
	if timeout <= 0 {
		a.logDroppedEvent(ev, "channel_full")
		return false
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
	select {
	case out <- ev:
		return true
	case <-timer.C:
		a.logDroppedEvent(ev, fmt.Sprintf("send_timeout_%s", timeout))
		return false
	}
}

func (a *Agent) tryEnqueueTerminalEvent(out chan Event, ev Event) bool {
	select {
	case buffered := <-out:
		if isTerminalAgentEvent(buffered) && terminalEventPriority(buffered) > terminalEventPriority(ev) {
			out <- buffered
			a.logDroppedEvent(ev, "terminal_priority_loss")
			return false
		}
		a.logDroppedEvent(buffered, "evicted_for_terminal")
		out <- ev
		return true
	default:
		return false
	}
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
	if dropped == 1 || dropped%logEvery == 0 {
		log.Printf("warning: dropping agent event %T due to backpressure (%s); dropped_total=%d", ev, reason, dropped)
	}
}

func (a *Agent) emitAutoContinue(out chan Event, reason string, responseID string) {
	a.emitEvent(out, AutoContinueEvent{Reason: reason, ResponseID: strings.TrimSpace(responseID)})
}

func (a *Agent) destroyEphemeralMessages() {
	a.mu.Lock()
	defer a.mu.Unlock()

	if a.ephemeralByTool == nil {
		a.ephemeralByTool = make(map[string][]int)
	}
	if a.ephemeralScanFrom < 0 || a.ephemeralScanFrom > len(a.messages) {
		a.resetEphemeralTrackingLocked()
	}

	for i := a.ephemeralScanFrom; i < len(a.messages); i++ {
		m := a.messages[i]
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
		a.ephemeralByTool[toolName] = append(a.ephemeralByTool[toolName], i)
	}
	a.ephemeralScanFrom = len(a.messages)

	for toolName, idxs := range a.ephemeralByTool {
		keep := 1
		if t, ok := a.toolMap[toolName]; ok {
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
			m.Content = llm.TextContent("<removed to save context>")
			a.messages[i] = m
		}
		if len(idxs) == 0 {
			delete(a.ephemeralByTool, toolName)
			continue
		}
		a.ephemeralByTool[toolName] = idxs
	}
}

func (a *Agent) resetEphemeralTrackingLocked() {
	a.ephemeralScanFrom = 0
	if a.ephemeralByTool == nil {
		a.ephemeralByTool = make(map[string][]int)
		return
	}
	for toolName := range a.ephemeralByTool {
		delete(a.ephemeralByTool, toolName)
	}
}

// NotifyTodoCompletion marks that compaction should be considered on the next check,
// even if the token threshold has not been reached yet.
func (a *Agent) NotifyTodoCompletion() {
	a.todoCompactionPending.Store(true)
}

func (a *Agent) checkAndCompact(ctx context.Context, last *llm.Completion, out chan Event) {
	if !a.hasCompactor || a.compactor == nil || last == nil {
		return
	}
	if ctx != nil && ctx.Err() != nil {
		return
	}
	a.applyPendingCompaction(out)
	if a.compactor.IsOverflow(last.Usage) {
		_ = a.compactSyncOverflow(ctx, last, out)
		return
	}
	if !a.shouldAttemptCompaction(ctx, last) {
		return
	}
	if !a.compactionInFlight.CompareAndSwap(false, true) {
		return
	}
	a.mu.Lock()
	messages := make([]llm.Message, len(a.messages))
	copy(messages, a.messages)
	a.mu.Unlock()
	snapshotLen := len(messages)
	triggerUsage := cloneUsage(last.Usage)
	trigger, watermark := a.compactionTriggerAndWatermark(last)
	go a.runCompactionAsync(ctx, messages, snapshotLen, triggerUsage, trigger, watermark)
}

func (a *Agent) runCompactionAsync(ctx context.Context, snapshot []llm.Message, snapshotLen int, triggerUsage *llm.Usage, trigger string, watermark string) {
	defer a.compactionInFlight.Store(false)

	compactCtx, cancelCompact := asyncCompactionContext(ctx)
	defer cancelCompact()
	compactUsage := triggerUsage
	if strings.TrimSpace(trigger) == "todo" {
		compactUsage = nil
	}
	newMsgs, res, err := a.compactWithRetry(compactCtx, snapshot, compactUsage, watermark)
	if err != nil {
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			return
		}
		a.warnf("compaction failed after %d attempt(s): %v", a.compactionMaxAttempts(), err)
		a.todoCompactionPending.Store(true)
		return
	}
	a.todoCompactionPending.Store(false)
	newMsgs = a.withPreservedSystem(snapshot, newMsgs)

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

func (a *Agent) compactSyncOverflow(ctx context.Context, last *llm.Completion, out chan Event) error {
	if !a.hasCompactor || a.compactor == nil || last == nil {
		return nil
	}
	if ctx != nil && ctx.Err() != nil {
		return ctx.Err()
	}
	a.applyPendingCompaction(out)
	if !a.compactionInFlight.CompareAndSwap(false, true) {
		if err := a.waitForCompactionIdle(ctx); err != nil {
			return err
		}
		a.applyPendingCompaction(out)
		if !a.compactionInFlight.CompareAndSwap(false, true) {
			return nil
		}
	}
	defer a.compactionInFlight.Store(false)

	a.mu.Lock()
	messages := make([]llm.Message, len(a.messages))
	copy(messages, a.messages)
	a.mu.Unlock()

	snapshotLen := len(messages)
	triggerUsage := cloneUsage(last.Usage)
	trigger, watermark := a.compactionTriggerAndWatermark(last)
	newMsgs, res, err := a.compactOverflowWithRetry(ctx, messages, triggerUsage)
	if err != nil {
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			return err
		}
		a.warnf("compaction failed after %d attempt(s): %v", a.compactionMaxAttempts(), err)
		a.todoCompactionPending.Store(true)
		return err
	}
	if strings.TrimSpace(res.Watermark) == "summarize" {
		a.todoCompactionPending.Store(false)
	} else {
		a.todoCompactionPending.Store(true)
	}
	newMsgs = a.withPreservedSystem(messages, newMsgs)

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

func (a *Agent) compactOverflowWithRetry(ctx context.Context, messages []llm.Message, usage *llm.Usage) ([]llm.Message, compaction.Result, error) {
	attempts := a.compactionMaxAttempts()
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
		localMsgs, localRes, summaryMsgs, summaryRes, err := a.compactOverflowOnce(ctx, messages, usage)
		if localRes.Compacted {
			fallbackMsgs = localMsgs
			fallbackRes = localRes
		}
		if err == nil {
			return summaryMsgs, summaryRes, nil
		}
		lastErr = err
		lastRes = summaryRes
		if !lastRes.Compacted {
			lastRes = localRes
		}
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
		a.todoCompactionPending.Store(true)
		return fallbackMsgs, fallbackRes, nil
	}
	if lastErr == nil {
		lastErr = errors.New("overflow summary compaction failed")
	}
	return messages, lastRes, lastErr
}

func (a *Agent) compactOverflowOnce(ctx context.Context, messages []llm.Message, usage *llm.Usage) ([]llm.Message, compaction.Result, []llm.Message, compaction.Result, error) {
	estimated := 0
	if a.compactor != nil {
		estimated = a.compactor.TotalTokens(usage)
	}
	var localMsgs []llm.Message
	var localRes compaction.Result
	var err error
	if estimated > 0 {
		localMsgs, localRes, err = a.compactor.CompactLocalEstimated(ctx, messages, estimated)
	} else {
		localMsgs, localRes, err = a.compactor.CompactLocal(ctx, messages, usage)
	}
	if err != nil {
		return messages, localRes, messages, compaction.Result{}, err
	}
	summaryInput := messages
	if localRes.Compacted {
		summaryInput = localMsgs
	}
	summaryMsgs, summaryRes, err := a.compactor.Compact(ctx, a.llm, summaryInput)
	if localRes.Compacted {
		summaryRes = mergeAgentLocalSummaryResult(localRes, summaryRes)
	}
	return localMsgs, localRes, summaryMsgs, summaryRes, err
}

func mergeAgentLocalSummaryResult(localRes compaction.Result, summaryRes compaction.Result) compaction.Result {
	if !localRes.Compacted {
		return summaryRes
	}
	if summaryRes.OriginalTokens <= 0 {
		summaryRes.OriginalTokens = localRes.OriginalTokens
	}
	if strings.TrimSpace(summaryRes.LedgerPath) == "" {
		summaryRes.LedgerPath = strings.TrimSpace(localRes.LedgerPath)
	}
	summaryRes.Warnings = append(localRes.Warnings, summaryRes.Warnings...)
	summaryRes.TiersApplied = appendDistinctTiers(localRes.TiersApplied, summaryRes.TiersApplied)
	return summaryRes
}

func appendDistinctTiers(first []string, rest []string) []string {
	out := make([]string, 0, len(first)+len(rest))
	seen := map[string]struct{}{}
	add := func(tiers []string) {
		for _, tier := range tiers {
			tier = strings.TrimSpace(tier)
			if tier == "" {
				continue
			}
			if _, ok := seen[tier]; ok {
				continue
			}
			seen[tier] = struct{}{}
			out = append(out, tier)
		}
	}
	add(first)
	add(rest)
	return out
}

func (a *Agent) waitForCompactionIdle(ctx context.Context) error {
	for a.compactionInFlight.Load() {
		if ctx != nil {
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(25 * time.Millisecond):
			}
			continue
		}
		time.Sleep(25 * time.Millisecond)
	}
	return nil
}

func (a *Agent) hasPendingCompaction() bool {
	a.pendingCompactionMu.Lock()
	defer a.pendingCompactionMu.Unlock()
	return a.pendingCompaction != nil
}

func (a *Agent) applyPendingCompaction(out chan Event) {
	if !a.hasCompactor {
		return
	}
	a.pendingCompactionMu.Lock()
	pending := a.pendingCompaction
	if pending != nil {
		a.pendingCompaction = nil
	}
	a.pendingCompactionMu.Unlock()
	if pending == nil {
		return
	}

	a.mu.Lock()
	currentLen := len(a.messages)
	tailCap := 0
	if currentLen > pending.snapshotLen {
		tailCap = currentLen - pending.snapshotLen
	}
	merged := make([]llm.Message, 0, len(pending.messages)+tailCap)
	merged = append(merged, pending.messages...)
	if currentLen < pending.snapshotLen {
		a.warnf("compaction apply skipped: history shrank (%d < %d); scheduling retry", currentLen, pending.snapshotLen)
		a.mu.Unlock()
		a.todoCompactionPending.Store(true)
		return
	}
	if currentLen > pending.snapshotLen {
		merged = append(merged, a.messages[pending.snapshotLen:]...)
	}
	a.messages = merged
	a.resetEphemeralTrackingLocked()
	a.mu.Unlock()

	a.emitEvent(out, CompactionEvent{Result: pending.result, TriggerUsage: pending.triggerUsage})
}

// CompactNow forces a compaction run regardless of current token usage.
func (a *Agent) CompactNow(ctx context.Context) (compaction.Result, error) {
	if !a.hasCompactor || a.compactor == nil {
		return compaction.Result{Compacted: false}, nil
	}
	a.applyPendingCompaction(nil)
	if !a.compactionInFlight.CompareAndSwap(false, true) {
		return compaction.Result{Compacted: false}, fmt.Errorf("compaction already in progress")
	}
	defer a.compactionInFlight.Store(false)

	a.mu.Lock()
	orig := make([]llm.Message, len(a.messages))
	copy(orig, a.messages)
	a.mu.Unlock()

	newMsgs, res, err := a.compactWithRetry(ctx, orig, nil, "summarize")
	if err != nil {
		return res, err
	}
	res = a.withCompactionTelemetry(res, "manual", "summarize", res.Usage)
	a.todoCompactionPending.Store(false)
	newMsgs = a.withPreservedSystem(orig, newMsgs)
	a.mu.Lock()
	a.messages = newMsgs
	a.resetEphemeralTrackingLocked()
	a.mu.Unlock()
	return res, nil
}

// CompactLocalNow forces the local snip/prune reducers using an estimated token
// count. It never invokes the model and is intended for prompt preflight.
func (a *Agent) CompactLocalNow(ctx context.Context, estimatedTokens int) (compaction.Result, error) {
	if !a.hasCompactor || a.compactor == nil {
		return compaction.Result{Compacted: false}, nil
	}
	a.applyPendingCompaction(nil)
	if !a.compactionInFlight.CompareAndSwap(false, true) {
		return compaction.Result{Compacted: false}, fmt.Errorf("compaction already in progress")
	}
	defer a.compactionInFlight.Store(false)

	a.mu.Lock()
	orig := make([]llm.Message, len(a.messages))
	copy(orig, a.messages)
	a.mu.Unlock()

	newMsgs, res, err := a.compactor.CompactLocalEstimated(ctx, orig, estimatedTokens)
	if err != nil {
		return res, err
	}
	res = a.withCompactionTelemetry(res, "preflight", res.Watermark, res.Usage)
	a.todoCompactionPending.Store(false)
	newMsgs = a.withPreservedSystem(orig, newMsgs)
	a.mu.Lock()
	a.messages = newMsgs
	a.resetEphemeralTrackingLocked()
	a.mu.Unlock()
	return res, nil
}

func (a *Agent) shouldAttemptCompaction(ctx context.Context, last *llm.Completion) bool {
	if !a.hasCompactor || a.compactor == nil || last == nil {
		return false
	}
	if ctx != nil && ctx.Err() != nil {
		return false
	}
	if a.compactionInFlight.Load() || a.hasPendingCompaction() {
		return false
	}
	if a.compactor.IsOverflow(last.Usage) {
		return true
	}
	if a.compactor.ShouldCompact(last.Usage) {
		return true
	}
	if !a.todoCompactionPending.Load() {
		return false
	}
	return a.compactor.PromptTokens(last.Usage) > 0
}

func (a *Agent) compactionTriggerAndWatermark(last *llm.Completion) (string, string) {
	if a.compactor == nil || last == nil {
		return "usage", "summarize"
	}
	if a.compactor.IsOverflow(last.Usage) {
		return "overflow", "overflow"
	}
	if watermark := a.compactor.WatermarkForUsage(last.Usage); strings.TrimSpace(watermark) != "" {
		return "usage", watermark
	}
	if a.todoCompactionPending.Load() && a.compactor.PromptTokens(last.Usage) > 0 {
		return "todo", "summarize"
	}
	return "usage", "summarize"
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
	if len(res.TiersApplied) == 0 {
		res.TiersApplied = []string{"summarize"}
	}
	if usage != nil {
		res.Usage = cloneUsage(usage)
	}
	if usage != nil && strings.TrimSpace(res.Trigger) != "manual" {
		if total := a.compactor.TotalTokens(usage); total > 0 {
			res.OriginalTokens = total
		}
	}
	if res.OriginalTokens <= 0 && usage != nil {
		if total := a.compactor.TotalTokens(usage); total > 0 {
			res.OriginalTokens = total
		}
	}
	if res.NewTokens <= 0 {
		res.NewTokens = res.OriginalTokens
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

func (a *Agent) compactWithRetry(ctx context.Context, messages []llm.Message, usage *llm.Usage, watermark string) ([]llm.Message, compaction.Result, error) {
	attempts := a.compactionMaxAttempts()
	if attempts <= 0 {
		attempts = 1
	}
	requestedWatermark := strings.TrimSpace(watermark)
	var lastErr error
	var lastRes compaction.Result
	for attempt := 1; attempt <= attempts; attempt++ {
		if err := ctx.Err(); err != nil {
			return messages, lastRes, err
		}
		newMsgs, res, err := a.compactor.CompactAuto(ctx, a.llm, messages, usage, requestedWatermark)
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
	if u == nil {
		return nil
	}
	cpy := *u
	if u.PromptCachedTokens != nil {
		v := *u.PromptCachedTokens
		cpy.PromptCachedTokens = &v
	}
	if u.PromptCacheCreationTokens != nil {
		v := *u.PromptCacheCreationTokens
		cpy.PromptCacheCreationTokens = &v
	}
	if u.PromptImageTokens != nil {
		v := *u.PromptImageTokens
		cpy.PromptImageTokens = &v
	}
	return &cpy
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
	case 401, 403, 408, 409, 425, 429:
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
func (a *Agent) drainSteering(ch <-chan SteeringMsg, out chan Event) {
	if ch == nil {
		return
	}
	for {
		select {
		case msg, ok := <-ch:
			if !ok {
				// Channel closed, stop draining
				return
			}
			if strings.TrimSpace(msg.Content) == "" {
				continue
			}
			a.mu.Lock()
			a.messages = append(a.messages, llm.Message{
				Role:    llm.RoleUser,
				Content: llm.TextContent(msg.Content),
			})
			a.mu.Unlock()
			a.emitEvent(out, SteeringReceivedEvent{Content: msg.Content})
		default:
			// Channel empty, return immediately (non-blocking)
			return
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
	return c.turns, c.turns <= c.maxTurns
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
	ids := make(map[string]bool)
	for _, tc := range c.partialCalls {
		if id := strings.TrimSpace(tc.ID); id != "" {
			ids[id] = true
		}
	}
	if len(ids) == 0 {
		c.reset()
		return
	}
	for i := 0; i < currentIndex && i < len(messages); i++ {
		if messages[i].Role != llm.RoleAssistant {
			continue
		}
		hasPartial := false
		for _, tc := range messages[i].ToolCalls {
			if ids[strings.TrimSpace(tc.ID)] {
				hasPartial = true
				break
			}
		}
		if hasPartial {
			messages[i].ToolCalls = nil
		}
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
