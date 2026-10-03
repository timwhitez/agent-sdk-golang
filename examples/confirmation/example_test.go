// Package confirmation_test demonstrates host-supplied dependency injection
// with a host-owned confirmation policy and a local model fixture.
package confirmation_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/agent"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools/sandbox"
)

const fixtureCommand = "echo confirmed>confirmation-marker.txt"

// mockConfirmer is a host-owned decision for this fixed local example only.
// It is not a policy for arbitrary model-generated commands.
type mockConfirmer struct {
	root    string
	allow   bool
	calls   int
	failure error
}

func (c *mockConfirmer) Confirm(_ context.Context, action, detail string) (bool, error) {
	c.calls++
	if action != "bash" {
		return false, fmt.Errorf("unexpected action %q", action)
	}
	var meta struct {
		Command string `json:"command"`
		Workdir string `json:"workdir"`
	}
	if err := json.Unmarshal([]byte(detail), &meta); err != nil {
		return false, err
	}
	if meta.Command != fixtureCommand || meta.Workdir != c.root {
		return false, errors.New("unexpected confirmation request")
	}
	return c.allow, c.failure
}

// fakeModel never connects to a provider. It observes the actual bash result and
// forwards its text through the real done tool, like the reported host flow.
type fakeModel struct {
	calls  int
	result *llm.Message
}

func (*fakeModel) Provider() string { return "local-fixture" }
func (*fakeModel) Model() string    { return "confirmation-fixture" }
func (m *fakeModel) Invoke(_ context.Context, req llm.InvokeRequest) (*llm.Completion, error) {
	m.calls++
	if m.calls == 1 {
		args, err := json.Marshal(map[string]string{"command": fixtureCommand})
		if err != nil {
			return nil, err
		}
		return &llm.Completion{ToolCalls: []llm.ToolCall{{ID: "bash-1", Type: "function", Function: llm.FunctionCall{Name: "bash", Arguments: string(args)}}}}, nil
	}
	if m.calls != 2 {
		return nil, errors.New("unexpected third model invocation")
	}
	for _, message := range req.Messages {
		if message.Role == llm.RoleTool && message.ToolCallID == "bash-1" {
			copy := message
			m.result = &copy
		}
	}
	if m.result == nil {
		return nil, errors.New("missing actual bash tool result")
	}
	message := "command completed"
	if m.result.IsError {
		message = m.result.PlainText()
	}
	args, err := json.Marshal(map[string]string{"message": message})
	if err != nil {
		return nil, err
	}
	return &llm.Completion{ToolCalls: []llm.ToolCall{{ID: "done-1", Type: "function", Function: llm.FunctionCall{Name: "done", Arguments: string(args)}}}}, nil
}

// Example shows both dependencies being injected before an Agent uses sandbox
// tools. The confirmer explicitly approves this fixed command in its private
// temporary directory. No provider, credentials, or interactive input is used.
func Example() {
	root, err := os.MkdirTemp("", "sdk-confirmation-example-*")
	if err != nil {
		panic(err)
	}
	defer os.RemoveAll(root)
	root, err = filepath.EvalSymlinks(root)
	if err != nil {
		panic(err)
	}
	box, err := sandbox.New(root)
	if err != nil {
		panic(err)
	}

	policy := &mockConfirmer{root: root, allow: true}
	deps := tools.NewContainer()
	tools.Provide(deps, sandbox.Key, func(context.Context) (*sandbox.Sandbox, error) {
		return box, nil
	})
	tools.Provide(deps, sandbox.ConfirmKey, func(context.Context) (sandbox.Confirmer, error) {
		return policy, nil
	})
	a, err := agent.New(agent.Config{
		LLM: &fakeModel{}, Tools: sandbox.Tools(), Deps: deps,
		MaxIterations: 3, RequireDoneTool: true,
		Warningf: func(string, ...any) {},
	})
	if err != nil {
		panic(err)
	}
	answer, err := a.Query(context.Background(), "Run the fixed local fixture command, then report its result with done.")
	if err != nil {
		panic(err)
	}
	marker, err := os.ReadFile(filepath.Join(root, "confirmation-marker.txt"))
	if err != nil {
		panic(err)
	}
	fmt.Println(answer)
	fmt.Println(strings.TrimSpace(string(marker)))
	// Output:
	// command completed
	// confirmed
}

func TestAgentConfirmationBoundary(t *testing.T) {
	for _, tc := range []struct {
		name      string
		register  bool
		allow     bool
		failure   error
		errorText string
	}{
		{name: "approve", register: true, allow: true},
		{name: "deny", register: true, errorText: "bash request denied: user denied request"},
		{name: "missing", errorText: sandbox.ErrMissingConfirmer.Error()},
		{name: "confirmation_error", register: true, failure: errors.New("fixture confirmation unavailable"), errorText: "fixture confirmation unavailable"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			root, err := filepath.EvalSymlinks(t.TempDir())
			if err != nil {
				t.Fatal(err)
			}
			box, err := sandbox.New(root)
			if err != nil {
				t.Fatal(err)
			}
			deps := tools.NewContainer()
			tools.Provide(deps, sandbox.Key, func(context.Context) (*sandbox.Sandbox, error) { return box, nil })
			confirmer := &mockConfirmer{root: root, allow: tc.allow, failure: tc.failure}
			if tc.register {
				tools.Provide(deps, sandbox.ConfirmKey, func(context.Context) (sandbox.Confirmer, error) { return confirmer, nil })
			}
			model := &fakeModel{}
			a, err := agent.New(agent.Config{LLM: model, Tools: sandbox.Tools(), Deps: deps, MaxIterations: 3, RequireDoneTool: true, Warningf: func(string, ...any) {}})
			if err != nil {
				t.Fatal(err)
			}
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			var results []agent.ToolResultEvent
			var finals []agent.FinalResponseEvent
			for event := range a.QueryStream(ctx, llm.TextContent("Run the fixed local fixture command, then report its result with done.")) {
				switch e := event.(type) {
				case agent.ToolResultEvent:
					if e.ToolCallID == "bash-1" {
						results = append(results, e)
					}
				case agent.FinalResponseEvent:
					finals = append(finals, e)
				case agent.ErrorEvent:
					t.Fatalf("agent error: %+v", e)
				}
			}
			if model.calls != 2 || model.result == nil || len(results) != 1 || len(finals) != 1 {
				t.Fatalf("calls=%d result=%v events=%d finals=%d", model.calls, model.result, len(results), len(finals))
			}
			wantConfirms := 0
			if tc.register {
				wantConfirms = 1
			}
			if confirmer.calls != wantConfirms {
				t.Fatalf("confirmer calls=%d want=%d", confirmer.calls, wantConfirms)
			}
			succeeded := tc.allow && tc.failure == nil
			marker, readErr := os.ReadFile(filepath.Join(root, "confirmation-marker.txt"))
			if succeeded {
				if readErr != nil || strings.TrimSpace(string(marker)) != "confirmed" {
					t.Fatalf("missing execution evidence: %q %v", marker, readErr)
				}
			} else if !errors.Is(readErr, os.ErrNotExist) {
				t.Fatalf("unapproved command effect: %q %v", marker, readErr)
			}
			if tc.name == "deny" && results[0].Metadata["error_kind"] != "denied" {
				t.Fatalf("missing denial metadata: %+v", results[0].Metadata)
			}
			if model.result.IsError == succeeded || results[0].IsError == succeeded {
				t.Fatalf("wrong tool outcome: history=%+v event=%+v", model.result, results[0])
			}
			if tc.errorText != "" && (!strings.Contains(model.result.PlainText(), tc.errorText) || !strings.Contains(finals[0].Content, tc.errorText)) {
				t.Fatalf("missing failure evidence in history/final: %q %q", model.result.PlainText(), finals[0].Content)
			}
			count := 0
			for _, message := range a.Messages() {
				if message.Role == llm.RoleTool && message.ToolCallID == "bash-1" {
					count++
				}
			}
			if count != 1 {
				t.Fatalf("terminal history results=%d", count)
			}
		})
	}
}
