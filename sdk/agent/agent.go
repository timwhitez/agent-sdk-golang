package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/agent/compaction"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

type Config struct {
	LLM          llm.ChatModel
	Tools        []tools.Tool
	SystemPrompt string

	// InitialMessages restores a previous conversation history.
	// If provided, the agent will not auto-insert SystemPrompt on first query unless you include it here.
	InitialMessages []llm.Message

	MaxIterations int
	ToolChoice    llm.ToolChoice

	Compaction *compaction.Config

	RequireDoneTool bool

	Deps *tools.Container
}

type Agent struct {
	llm           llm.ChatModel
	systemPrompt  string
	maxIterations int
	toolChoice    llm.ToolChoice
	requireDone   bool

	tools   []tools.Tool
	toolMap map[string]tools.Tool
	deps    *tools.Container

	compactor *compaction.Service

	mu       sync.Mutex
	messages []llm.Message
}

// SteeringMsg represents a user message injected mid-turn for real-time steering.
// This enables the "Real-time steering" feature where users can send feedback
// while the agent is working, and the agent incorporates it immediately.
type SteeringMsg struct {
	Content string
}

func New(cfg Config) (*Agent, error) {
	if cfg.LLM == nil {
		return nil, fmt.Errorf("agent: LLM is required")
	}
	if cfg.MaxIterations <= 0 {
		cfg.MaxIterations = 200
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
	if cfg.InitialMessages == nil {
		cfg.InitialMessages = nil
	}

	ag := &Agent{
		llm:           cfg.LLM,
		systemPrompt:  cfg.SystemPrompt,
		maxIterations: cfg.MaxIterations,
		toolChoice:    cfg.ToolChoice,
		requireDone:   cfg.RequireDoneTool,
		tools:         append([]tools.Tool(nil), cfg.Tools...),
		toolMap:       toolMap,
		deps:          cfg.Deps,
		compactor:     compSvc,
	}
	if len(cfg.InitialMessages) > 0 {
		ag.messages = append([]llm.Message(nil), cfg.InitialMessages...)
	}
	return ag, nil
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
	defer a.mu.Unlock()
	a.messages = nil
}

// ReplaceHistory replaces the current conversation history.
// Callers should include the system prompt message if they want it preserved.
func (a *Agent) ReplaceHistory(messages []llm.Message) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.messages = append([]llm.Message(nil), messages...)
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
// This implements "boundary-aware queuing" and "real-time steering":
//   - Boundary-aware: messages are processed at tool-call boundaries, not just at turn end
//   - Real-time steering: users can redirect the agent mid-turn without waiting for completion
func (a *Agent) QueryStreamWithSteering(ctx context.Context, input llm.Content, steeringCh <-chan SteeringMsg) <-chan Event {
	out := make(chan Event, 32)
	go func() {
		defer close(out)

		a.mu.Lock()
		if len(a.messages) == 0 && a.systemPrompt != "" {
			// For Anthropic, system can be cached; keep a.Cache=false by default.
			a.messages = append(a.messages, llm.NewSystemMessage(a.systemPrompt))
		}
		a.messages = append(a.messages, llm.Message{Role: llm.RoleUser, Content: input})
		a.mu.Unlock()

		incompleteTodosPrompted := false

		for iter := 0; iter < a.maxIterations; iter++ {
			// *** Boundary-aware steering: check for new user messages before each LLM call ***
			a.drainSteering(steeringCh, out)

			// Remove old ephemeral messages before the next LLM call.
			a.destroyEphemeralMessages()

			messages := a.Messages()
			toolDefs := make([]llm.ToolDefinition, 0, len(a.tools))
			for _, t := range a.tools {
				toolDefs = append(toolDefs, t.Definition())
			}

			comp, streamedText, err := a.invokeCompletion(ctx, llm.InvokeRequest{
				Messages:   messages,
				Tools:      toolDefs,
				ToolChoice: a.toolChoice,
			}, out)
			if err != nil {
				out <- a.errEvent(err)
				return
			}
			if comp.Usage != nil {
				out <- UsageEvent{Usage: *comp.Usage}
			}

			if comp.Thinking != "" {
				out <- ThinkingEvent{Content: comp.Thinking}
			}
			if !streamedText {
				if txt := comp.PlainText(); txt != "" {
					out <- TextEvent{Content: txt}
				}
			}

			// Append assistant message.
			a.mu.Lock()
			a.messages = append(a.messages, llm.Message{Role: llm.RoleAssistant, Content: comp.Content, ToolCalls: comp.ToolCalls})
			a.mu.Unlock()

			// Stopping condition.
			if !comp.HasToolCalls() {
				// Auto-continue: if the response was truncated due to max_tokens,
				// send a continuation prompt and loop again.
				if comp.StopReason == "max_tokens" {
					a.mu.Lock()
					a.messages = append(a.messages, llm.Message{
						Role:    llm.RoleUser,
						Content: llm.TextContent("Your response was truncated. Please continue exactly where you left off."),
					})
					a.mu.Unlock()
					out <- TextDeltaEvent{Delta: "\n[auto-continue]\n"}
					continue
				}
				if !a.requireDone {
					// Hook placeholder: incomplete todo prompting (future), matching Python API.
					if !incompleteTodosPrompted {
						// no prompt for now
						incompleteTodosPrompted = true
					}
					// compaction check
					a.checkAndCompact(ctx, comp, out)
					out <- FinalResponseEvent{Content: comp.PlainText()}
					return
				}
				// require done tool: continue looping
				continue
			}

			// Execute tool calls.
			step := 0
			for _, tc := range comp.ToolCalls {
				step++
				out <- StepStartEvent{StepID: tc.ID, Title: tc.Function.Name, StepNumber: step}

				argsMap := map[string]any{}
				if err := json.Unmarshal([]byte(tc.Function.Arguments), &argsMap); err != nil {
					argsMap = map[string]any{"__raw": tc.Function.Arguments}
				}
				out <- ToolCallEvent{Tool: tc.Function.Name, Args: argsMap, ToolCallID: tc.ID, DisplayName: tc.Function.Name}

				start := time.Now()
				tool := a.toolMap[tc.Function.Name]
				ctxTool := tools.WithToolCallID(ctx, tc.ID)
				ctxTool = tools.WithToolResultMetadata(ctxTool)
				content, toolErr := tool.Execute(ctxTool, tc.Function.Arguments, a.deps)
				isError := toolErr != nil
				status := "completed"
				if isError {
					status = "error"
				}
				meta := tools.TakeToolResultMetadataSnapshot(ctxTool)

				// If tool is configured ephemeral, mark tool message accordingly.
				ephemeral := tool.EphemeralKeep > 0

				// Tool completion special-case.
				var tce *tools.TaskCompleteError
				if errors.As(toolErr, &tce) {
					isError = false
					status = "completed"
					content = llm.TextContent("Task completed: " + tce.Message)
					// append tool message and finish
					a.mu.Lock()
					a.messages = append(a.messages, llm.Message{Role: llm.RoleTool, ToolCallID: tc.ID, ToolName: tc.Function.Name, Content: content, IsError: false, Ephemeral: ephemeral})
					a.mu.Unlock()
					out <- ToolResultEvent{Tool: tc.Function.Name, Result: content.PlainText(), ToolCallID: tc.ID, IsError: false, Metadata: meta}
					out <- StepCompleteEvent{StepID: tc.ID, Status: status, DurationMS: time.Since(start).Milliseconds()}
					out <- FinalResponseEvent{Content: tce.Message}
					return
				}

				// append tool message
				a.mu.Lock()
				a.messages = append(a.messages, llm.Message{Role: llm.RoleTool, ToolCallID: tc.ID, ToolName: tc.Function.Name, Content: content, IsError: isError, Ephemeral: ephemeral})
				a.mu.Unlock()

				out <- ToolResultEvent{Tool: tc.Function.Name, Result: content.PlainText(), ToolCallID: tc.ID, IsError: isError, Metadata: meta}
				out <- StepCompleteEvent{StepID: tc.ID, Status: status, DurationMS: time.Since(start).Milliseconds()}

				// *** Boundary-aware steering: check for new user messages after each tool execution ***
				a.drainSteering(steeringCh, out)
			}

			a.checkAndCompact(ctx, comp, out)
		}

		out <- FinalResponseEvent{Content: fmt.Sprintf("[Max iterations reached] %d", a.maxIterations)}
	}()
	return out
}

type toolCallBuilder struct {
	id   string
	name strings.Builder
	args strings.Builder
}

type toolCallAccumulator struct {
	items []toolCallBuilder
}

func (a *toolCallAccumulator) ensure(index int) *toolCallBuilder {
	if index < 0 {
		index = 0
	}
	for len(a.items) <= index {
		a.items = append(a.items, toolCallBuilder{})
	}
	return &a.items[index]
}

func (a *toolCallAccumulator) apply(d llm.StreamToolCallDeltaEvent) {
	it := a.ensure(d.Index)
	if strings.TrimSpace(d.ID) != "" && strings.TrimSpace(it.id) == "" {
		it.id = d.ID
	}
	if strings.TrimSpace(d.NameDelta) != "" {
		it.name.WriteString(d.NameDelta)
	}
	if strings.TrimSpace(d.ArgumentsDelta) != "" {
		it.args.WriteString(d.ArgumentsDelta)
	}
}

func (a *toolCallAccumulator) finalize() []llm.ToolCall {
	out := []llm.ToolCall{}
	for i := range a.items {
		it := &a.items[i]
		name := strings.TrimSpace(it.name.String())
		args := strings.TrimSpace(it.args.String())
		if name == "" {
			continue
		}
		if args == "" {
			args = "{}"
		}
		id := strings.TrimSpace(it.id)
		if id == "" {
			id = fmt.Sprintf("call_%d", i)
		}
		out = append(out, llm.ToolCall{ID: id, Type: "function", Function: llm.FunctionCall{Name: name, Arguments: args}})
	}
	return out
}

// invokeCompletion calls the provider using streaming when available.
// It returns the completion, whether text was streamed, and error.
func (a *Agent) invokeCompletion(ctx context.Context, req llm.InvokeRequest, out chan<- Event) (*llm.Completion, bool, error) {
	if a == nil || a.llm == nil {
		return nil, false, fmt.Errorf("agent: nil llm")
	}
	if sm, ok := a.llm.(llm.StreamingChatModel); ok {
		ch, err := sm.InvokeStream(ctx, req)
		if err != nil {
			return nil, false, err
		}
		var text strings.Builder
		var thinking strings.Builder
		acc := &toolCallAccumulator{}
		var usage *llm.Usage
		stopReason := ""
		streamedText := false
		for ev := range ch {
			switch e := ev.(type) {
			case llm.StreamTextDeltaEvent:
				if strings.TrimSpace(e.Delta) != "" || e.Delta == "\n" {
					text.WriteString(e.Delta)
					streamedText = true
					if out != nil {
						out <- TextDeltaEvent{Delta: e.Delta}
					}
				} else {
					// preserve whitespace as-is
					text.WriteString(e.Delta)
					if e.Delta != "" {
						streamedText = true
						if out != nil {
							out <- TextDeltaEvent{Delta: e.Delta}
						}
					}
				}
			case llm.StreamThinkingDeltaEvent:
				thinking.WriteString(e.Delta)
			case llm.StreamToolCallDeltaEvent:
				acc.apply(e)
			case llm.StreamUsageEvent:
				u := e.Usage
				usage = &u
			case llm.StreamErrorEvent:
				if e.Err != nil {
					return nil, streamedText, e.Err
				}
				return nil, streamedText, fmt.Errorf("stream error")
			case llm.StreamDoneEvent:
				stopReason = e.StopReason
			default:
				// ignore unknown
			}
		}
		return &llm.Completion{Content: llm.TextContent(text.String()), Thinking: strings.TrimSpace(thinking.String()), ToolCalls: acc.finalize(), Usage: usage, StopReason: stopReason}, streamedText, nil
	}
	comp, err := a.llm.Invoke(ctx, req)
	return comp, false, err
}

func (a *Agent) destroyEphemeralMessages() {
	a.mu.Lock()
	defer a.mu.Unlock()

	byTool := map[string][]int{} // tool_name -> indices
	for i, m := range a.messages {
		if m.Role != llm.RoleTool {
			continue
		}
		if !m.Ephemeral || m.Destroyed {
			continue
		}
		byTool[m.ToolName] = append(byTool[m.ToolName], i)
	}
	for toolName, idxs := range byTool {
		keep := 1
		if t, ok := a.toolMap[toolName]; ok {
			if t.EphemeralKeep > 0 {
				keep = t.EphemeralKeep
			}
		}
		if keep <= 0 {
			keep = 1
		}
		if len(idxs) <= keep {
			continue
		}
		toDestroy := idxs[:len(idxs)-keep]
		for _, i := range toDestroy {
			m := a.messages[i]
			m.Destroyed = true
			m.Content = llm.TextContent("<removed to save context>")
			a.messages[i] = m
		}
	}
}

func (a *Agent) checkAndCompact(ctx context.Context, last *llm.Completion, out chan<- Event) {
	if a.compactor == nil || last == nil {
		return
	}
	if !a.compactor.ShouldCompact(last.Usage) {
		return
	}
	a.mu.Lock()
	messages := make([]llm.Message, len(a.messages))
	copy(messages, a.messages)
	a.mu.Unlock()

	origMsgs := messages
	newMsgs, res, err := a.compactor.Compact(ctx, a.llm, messages)
	if err != nil {
		return
	}
	newMsgs = a.withPreservedSystem(origMsgs, newMsgs)
	a.mu.Lock()
	a.messages = newMsgs
	a.mu.Unlock()
	if out != nil {
		out <- CompactionEvent{Result: res, TriggerUsage: last.Usage}
	}
}

// CompactNow forces a compaction run regardless of current token usage.
func (a *Agent) CompactNow(ctx context.Context) (compaction.Result, error) {
	if a.compactor == nil {
		return compaction.Result{Compacted: false}, nil
	}
	a.mu.Lock()
	orig := make([]llm.Message, len(a.messages))
	copy(orig, a.messages)
	a.mu.Unlock()

	newMsgs, res, err := a.compactor.Compact(ctx, a.llm, orig)
	if err != nil {
		return res, err
	}
	newMsgs = a.withPreservedSystem(orig, newMsgs)
	a.mu.Lock()
	a.messages = newMsgs
	a.mu.Unlock()
	return res, nil
}

func (a *Agent) withPreservedSystem(orig []llm.Message, compacted []llm.Message) []llm.Message {
	if len(compacted) > 0 && compacted[0].Role == llm.RoleSystem {
		return compacted
	}
	sys := make([]llm.Message, 0, 1)
	for _, m := range orig {
		if m.Role == llm.RoleSystem {
			sys = append(sys, m)
		}
	}
	if len(sys) == 0 && strings.TrimSpace(a.systemPrompt) != "" {
		sys = append(sys, llm.NewSystemMessage(a.systemPrompt))
	}
	if len(sys) == 0 {
		return compacted
	}
	out := make([]llm.Message, 0, len(sys)+len(compacted))
	out = append(out, sys...)
	out = append(out, compacted...)
	return out
}

func (a *Agent) errEvent(err error) ErrorEvent {
	if err == nil {
		return ErrorEvent{Provider: a.llm.Provider(), Message: "<nil>", Kind: "unknown"}
	}
	var rl *llm.RateLimitError
	if errors.As(err, &rl) {
		return ErrorEvent{Provider: rl.Provider, StatusCode: 429, Message: rl.Message, Kind: "rate_limit"}
	}
	var pe *llm.ProviderError
	if errors.As(err, &pe) {
		return ErrorEvent{Provider: pe.Provider, StatusCode: pe.StatusCode, Message: pe.Message, Kind: "provider"}
	}
	// Best-effort: preserve model provider when possible.
	prov := ""
	if a.llm != nil {
		prov = a.llm.Provider()
	}
	return ErrorEvent{Provider: prov, Message: err.Error(), Kind: "unknown"}
}

// drainSteering non-blockingly reads all pending messages from the steering channel
// and appends them to the conversation history as user messages. This allows users
// to inject new instructions at natural breakpoints (tool-call boundaries) without
// blocking the agent's execution loop.
//
// The function returns immediately if the channel is nil or empty.
// Each received message triggers a SteeringReceivedEvent to notify the CLI layer.
func (a *Agent) drainSteering(ch <-chan SteeringMsg, out chan<- Event) {
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
			if out != nil {
				out <- SteeringReceivedEvent{Content: msg.Content}
			}
		default:
			// Channel empty, return immediately (non-blocking)
			return
		}
	}
}
