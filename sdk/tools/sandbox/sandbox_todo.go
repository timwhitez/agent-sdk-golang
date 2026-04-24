package sandbox

import (
	"context"
	"fmt"
	"strings"

	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

// todoWriteArgs holds the arguments for the todo_write tool.
type todoWriteArgs struct {
	Todos []TodoItem `json:"todos"`
}

// todoReadToolNamed returns a tool that reads the current todo list.
// The name parameter allows both the preferred "todoread" name and the legacy "todo_read" alias.
func todoReadToolNamed(name string) tools.Tool {
	name = strings.TrimSpace(name)
	if name == "" {
		name = "todo_read"
	}
	desc := "Read current todo list"
	if name == "todoread" {
		desc = "Read current todo list (alias: todo_read)"
	}
	return tools.Func[struct{}](name, desc, func(ctx context.Context, _ struct{}, deps *tools.Container) (any, error) {
		s, err := tools.Get(deps, ctx, Key)
		if err != nil {
			return "", err
		}
		conf := getConfirmer(deps, ctx)
		meta := attachToolCallMeta(ctx, map[string]any{
			"category": "state_read",
			"summary":  fmt.Sprintf("%s", name),
			"raw":      "read todos",
		})
		ok, err := conf.Confirm(ctx, name, buildConfirmDetail(meta))
		if err != nil {
			msg := formatErrorDiagnosticFromErr("todo read confirmation failed", err, "Retry after confirmation policy is available.")
			return msg, err
		}
		if !ok {
			denied, denyErr := denyToolResult(ctx, name, "user denied request")
			return denied.PlainText(), denyErr
		}
		todos := s.TodosSnapshot()
		if len(todos) == 0 {
			return "Todo list is empty", nil
		}
		lines := []string{}
		for i, t := range todos {
			status := map[string]string{"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}[t.Status]
			if status == "" {
				status = "[ ]"
			}
			lines = append(lines, fmt.Sprintf("%d. %s %s", i+1, status, t.Content))
		}
		return strings.Join(lines, "\n"), nil
	})
}

// todoWriteToolNamed returns a tool that updates the todo list.
// The name parameter allows both the preferred "todowrite" name and the legacy "todo_write" alias.
func todoWriteToolNamed(name string) tools.Tool {
	name = strings.TrimSpace(name)
	if name == "" {
		name = "todo_write"
	}
	desc := "Update the todo list"
	if name == "todowrite" {
		desc = "Update the todo list (alias: todo_write)"
	}
	return tools.Func[todoWriteArgs](name, desc, func(ctx context.Context, a todoWriteArgs, deps *tools.Container) (any, error) {
		s, err := tools.Get(deps, ctx, Key)
		if err != nil {
			return "", err
		}
		conf := getConfirmer(deps, ctx)
		meta := attachToolCallMeta(ctx, map[string]any{
			"category": "state_write",
			"summary":  fmt.Sprintf("%s (%d items)", name, len(a.Todos)),
			"raw":      fmt.Sprintf("%d items", len(a.Todos)),
		})
		ok, err := conf.Confirm(ctx, name, buildConfirmDetail(meta))
		if err != nil {
			msg := formatErrorDiagnosticFromErr("todo write confirmation failed", err, "Retry after confirmation policy is available.")
			return msg, err
		}
		if !ok {
			denied, denyErr := denyToolResult(ctx, name, "user denied request")
			return denied.PlainText(), denyErr
		}
		s.ReplaceTodos(a.Todos)
		stats := map[string]int{"pending": 0, "in_progress": 0, "completed": 0}
		for _, t := range a.Todos {
			stats[t.Status]++
		}
		return fmt.Sprintf("Updated todos: %d pending, %d in progress, %d completed", stats["pending"], stats["in_progress"], stats["completed"]), nil
	})
}

// doneArgs holds the arguments for the done tool.
type doneArgs struct {
	Message string `json:"message"`
}

// doneTool returns a tool that signals task completion.
func doneTool() tools.Tool {
	return tools.Func[doneArgs]("done", "Signal that the task is complete", func(ctx context.Context, a doneArgs, _ *tools.Container) (any, error) {
		return "", tools.TaskComplete(a.Message)
	}).WithEphemeralKeep(0)
}
