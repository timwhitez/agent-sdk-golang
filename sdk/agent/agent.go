package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
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

	return &Agent{
		llm:           cfg.LLM,
		systemPrompt:  cfg.SystemPrompt,
		maxIterations: cfg.MaxIterations,
		toolChoice:    cfg.ToolChoice,
		requireDone:   cfg.RequireDoneTool,
		tools:         append([]tools.Tool(nil), cfg.Tools...),
		toolMap:       toolMap,
		deps:          cfg.Deps,
		compactor:     compSvc,
	}, nil
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

func (a *Agent) Query(ctx context.Context, text string) (string, error) {
	ch := a.QueryStream(ctx, llm.TextContent(text))
	final := ""
	for ev := range ch {
		if f, ok := ev.(FinalResponseEvent); ok {
			final = f.Content
		}
	}
	return final, nil
}

func (a *Agent) QueryStream(ctx context.Context, input llm.Content) <-chan Event {
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
			// Remove old ephemeral messages before the next LLM call.
			a.destroyEphemeralMessages()

			messages := a.Messages()
			toolDefs := make([]llm.ToolDefinition, 0, len(a.tools))
			for _, t := range a.tools {
				toolDefs = append(toolDefs, t.Definition())
			}

			comp, err := a.llm.Invoke(ctx, llm.InvokeRequest{
				Messages:   messages,
				Tools:      toolDefs,
				ToolChoice: a.toolChoice,
			})
			if err != nil {
				out <- FinalResponseEvent{Content: fmt.Sprintf("LLM error: %v", err)}
				return
			}

			if comp.Thinking != "" {
				out <- ThinkingEvent{Content: comp.Thinking}
			}
			if txt := comp.PlainText(); txt != "" {
				out <- TextEvent{Content: txt}
			}

			// Append assistant message.
			a.mu.Lock()
			a.messages = append(a.messages, llm.Message{Role: llm.RoleAssistant, Content: comp.Content, ToolCalls: comp.ToolCalls})
			a.mu.Unlock()

			// Stopping condition.
			if !comp.HasToolCalls() {
				if !a.requireDone {
					// Hook placeholder: incomplete todo prompting (future), matching Python API.
					if !incompleteTodosPrompted {
						// no prompt for now
						incompleteTodosPrompted = true
					}
					// compaction check
					a.checkAndCompact(ctx, comp)
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
				_ = json.Unmarshal([]byte(tc.Function.Arguments), &argsMap)
				out <- ToolCallEvent{Tool: tc.Function.Name, Args: argsMap, ToolCallID: tc.ID, DisplayName: tc.Function.Name}

				start := time.Now()
				tool := a.toolMap[tc.Function.Name]
				content, toolErr := tool.Execute(ctx, tc.Function.Arguments, a.deps)
				isError := toolErr != nil
				status := "completed"
				if isError {
					status = "error"
				}

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
					out <- ToolResultEvent{Tool: tc.Function.Name, Result: content.PlainText(), ToolCallID: tc.ID, IsError: false}
					out <- StepCompleteEvent{StepID: tc.ID, Status: status, DurationMS: time.Since(start).Milliseconds()}
					out <- FinalResponseEvent{Content: tce.Message}
					return
				}

				// append tool message
				a.mu.Lock()
				a.messages = append(a.messages, llm.Message{Role: llm.RoleTool, ToolCallID: tc.ID, ToolName: tc.Function.Name, Content: content, IsError: isError, Ephemeral: ephemeral})
				a.mu.Unlock()

				out <- ToolResultEvent{Tool: tc.Function.Name, Result: content.PlainText(), ToolCallID: tc.ID, IsError: isError}
				out <- StepCompleteEvent{StepID: tc.ID, Status: status, DurationMS: time.Since(start).Milliseconds()}
			}

			a.checkAndCompact(ctx, comp)
		}

		out <- FinalResponseEvent{Content: fmt.Sprintf("[Max iterations reached] %d", a.maxIterations)}
	}()
	return out
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

func (a *Agent) checkAndCompact(ctx context.Context, last *llm.Completion) {
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

	newMsgs, _, err := a.compactor.Compact(ctx, a.llm, messages)
	if err != nil {
		return
	}
	a.mu.Lock()
	a.messages = newMsgs
	a.mu.Unlock()
}
