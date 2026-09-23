// Command streaming shows how to consume real HTTP SSE output from the three
// built-in protocol clients, either one model call at a time (InvokeStream) or
// through the Agent tool loop (QueryStream).
//
// Configuration comes from flags and the environment; nothing is hard-coded:
//
//	STREAM_MODEL     model name (required)
//	ANTHROPIC_API_KEY / OPENAI_API_KEY   key for the selected protocol (required)
//	STREAM_BASE_URL  optional API base URL (gateways, local fixtures)
//
//	go run ./examples/streaming -protocol responses -mode invoke -prompt "Say hi"
//	go run ./examples/streaming -protocol anthropic -mode agent  -prompt "Say hi"
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"strings"

	"github.com/timwhitez/agent-sdk-golang/sdk/agent"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm/anthropic"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm/openai"
)

// The three built-in clients implement real HTTP SSE streaming.
var (
	_ llm.StreamingChatModel = (*anthropic.Client)(nil)
	_ llm.StreamingChatModel = (*openai.ChatClient)(nil)
	_ llm.StreamingChatModel = (*openai.ResponsesClient)(nil)
)

type config struct {
	Protocol string // anthropic | chat | responses
	Model    string
	APIKey   string
	BaseURL  string
}

// newModel builds the selected built-in client. Missing configuration fails
// here, before any network call.
func newModel(cfg config) (llm.StreamingChatModel, error) {
	if strings.TrimSpace(cfg.Model) == "" {
		return nil, errors.New("missing model: set STREAM_MODEL")
	}
	if strings.TrimSpace(cfg.APIKey) == "" {
		return nil, fmt.Errorf("missing API key for protocol %q", cfg.Protocol)
	}
	switch cfg.Protocol {
	case "anthropic":
		return &anthropic.Client{BaseURL: cfg.BaseURL, APIKey: cfg.APIKey, ModelName: cfg.Model}, nil
	case "chat":
		return &openai.ChatClient{BaseURL: cfg.BaseURL, APIKey: cfg.APIKey, ModelName: cfg.Model}, nil
	case "responses":
		return &openai.ResponsesClient{BaseURL: cfg.BaseURL, APIKey: cfg.APIKey, ModelName: cfg.Model}, nil
	default:
		return nil, fmt.Errorf("unknown protocol %q (want anthropic, chat or responses)", cfg.Protocol)
	}
}

// errIncompleteStream reports a stream that closed without a terminal event:
// the text received so far is not a complete response.
var errIncompleteStream = errors.New("stream closed before its terminal event; the output is incomplete")

// streamInvoke performs one model call and writes each visible text delta as
// it arrives. Only deltas are written; tool-call argument deltas are partial
// JSON and are never executed here (use the Agent for tool loops). Thinking,
// signatures and opaque provider state are not printed.
func streamInvoke(ctx context.Context, model llm.StreamingChatModel, prompt string, w, diag io.Writer) error {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel() // an early return also stops the request and closes the body
	events, err := model.InvokeStream(ctx, llm.InvokeRequest{Messages: []llm.Message{llm.NewUserMessage(prompt)}})
	if err != nil {
		return err
	}
	done := false
	stopReason := ""
	for event := range events {
		switch e := event.(type) {
		case llm.StreamTextDeltaEvent:
			if _, err := io.WriteString(w, e.Delta); err != nil {
				return err
			}
		case llm.StreamErrorEvent:
			return e.AsError()
		case llm.StreamDoneEvent:
			// A normal provider terminal. StopReason is the provider's own
			// value and may still say the output was cut short (for example a
			// length limit), so it is reported rather than interpreted.
			done = true
			stopReason = e.StopReason
		}
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if !done {
		return errIncompleteStream
	}
	if _, err := io.WriteString(w, "\n"); err != nil {
		return err
	}
	if stopReason != "" {
		fmt.Fprintf(diag, "[stop_reason=%s]\n", stopReason)
	}
	return nil
}

// errPartialResponse reports a FinalResponseEvent the Agent marked partial:
// a bounded fallback answer, not a normally completed task.
var errPartialResponse = errors.New("agent returned a partial response")

// consumeAgentEvents prints the Agent's text deltas once each. The final
// answer is printed only when no delta was streamed (a non-streaming model),
// so text is never duplicated. Errors, partial answers, critically dropped
// events and a stream without a final response are all failures.
// finalMarker introduces an authoritative final answer that differs from the
// text the last model turn streamed.
const finalMarker = "[final answer]\n"

func consumeAgentEvents(events <-chan agent.Event, w, diag io.Writer) error {
	// turn is the text streamed by the current model turn. A tool call or tool
	// result ends the turn: text streamed before a tool is progress, not the
	// answer. The final answer is printed unless this turn already showed
	// exactly that text; an empty turn, a lost delta or a final answer that
	// comes from a tool (such as done) all print it.
	var turn strings.Builder
	printed, atLineStart := false, true
	write := func(text string) error {
		if text == "" {
			return nil
		}
		if _, err := io.WriteString(w, text); err != nil {
			return err
		}
		printed, atLineStart = true, strings.HasSuffix(text, "\n")
		return nil
	}
	endLine := func() error {
		if atLineStart {
			return nil
		}
		return write("\n")
	}
	for event := range events {
		switch e := event.(type) {
		case agent.TextDeltaEvent:
			turn.WriteString(e.Delta)
			if err := write(e.Delta); err != nil {
				return err
			}
		case agent.ToolCallEvent, agent.ToolResultEvent:
			turn.Reset()
			if err := endLine(); err != nil {
				return err
			}
		case agent.ErrorEvent:
			return fmt.Errorf("agent error (%s): %s", e.Kind, e.Message)
		case agent.FinalResponseEvent:
			shown := strings.TrimSpace(turn.String()) == strings.TrimSpace(e.Content)
			if err := endLine(); err != nil {
				return err
			}
			if !shown && e.Content != "" {
				if printed {
					if err := write(finalMarker); err != nil {
						return err
					}
				}
				if err := write(e.Content); err != nil {
					return err
				}
				if err := endLine(); err != nil {
					return err
				}
			}
			if !printed {
				if err := write("\n"); err != nil {
					return err
				}
			}
			if e.DroppedEvents > 0 {
				fmt.Fprintf(diag, "[%d events were dropped; %d critical]\n", e.DroppedEvents, e.DroppedCriticalEvents)
			}
			if e.Status == "partial" {
				return fmt.Errorf("%w (reason: %s)", errPartialResponse, e.Reason)
			}
			if e.DroppedCriticalEvents > 0 {
				return errors.New("critical events were dropped; the delivered stream is inconsistent with history")
			}
			return nil
		}
	}
	return errIncompleteStream
}

func streamAgent(ctx context.Context, model llm.ChatModel, prompt string, w, diag io.Writer) error {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	a, err := agent.New(agent.Config{LLM: model, SystemPrompt: "You are a helpful assistant."})
	if err != nil {
		return err
	}
	err = consumeAgentEvents(a.QueryStream(ctx, llm.TextContent(prompt)), w, diag)
	if ctxErr := ctx.Err(); err != nil && ctxErr != nil {
		return ctxErr
	}
	return err
}

func main() {
	protocol := flag.String("protocol", "chat", "anthropic | chat | responses")
	mode := flag.String("mode", "invoke", "invoke (one model call) | agent (tool loop)")
	prompt := flag.String("prompt", "Say hello in one short sentence.", "user prompt")
	flag.Parse()

	key := os.Getenv("OPENAI_API_KEY")
	if *protocol == "anthropic" {
		key = os.Getenv("ANTHROPIC_API_KEY")
	}
	model, err := newModel(config{Protocol: *protocol, Model: os.Getenv("STREAM_MODEL"), APIKey: key, BaseURL: os.Getenv("STREAM_BASE_URL")})
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(2)
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	switch *mode {
	case "invoke":
		err = streamInvoke(ctx, model, *prompt, os.Stdout, os.Stderr)
	case "agent":
		err = streamAgent(ctx, model, *prompt, os.Stdout, os.Stderr)
	default:
		err = fmt.Errorf("unknown mode %q", *mode)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "\nerror:", err)
		os.Exit(1)
	}
}
