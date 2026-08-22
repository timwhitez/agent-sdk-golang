package compaction

import (
	"context"
	"fmt"
	"strings"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

const (
	checkpointMaxItems          = 32
	checkpointMaxTasks          = 24
	checkpointMaxTaskTodos      = 16
	checkpointMaxEvidence       = 32
	checkpointMaxPendingTools   = 24
	checkpointMaxErrors         = 16
	checkpointMaxValidations    = 24
	checkpointMaxClaims         = 16
	checkpointFieldTokenBudget  = 100
	checkpointDetailTokenBudget = 160
)

const (
	CheckpointStatusVerified   = "VERIFIED"
	CheckpointStatusUnverified = "UNVERIFIED"
	CheckpointStatusUnknown    = "UNKNOWN"
)

// CheckpointProvider is the portable host boundary for compaction checkpoint
// material. Implementations may read host stores or inspect the supplied
// history, but the SDK itself never imports host packages or executes
// repository commands.
type CheckpointProvider func(context.Context, []llm.Message) (CheckpointContext, error)

type CheckpointValue struct {
	Value  string
	Status string
	Source string
}

type CheckpointItem struct {
	ID     string
	Text   string
	Status string
	Source string
}

type CheckpointTask struct {
	ID        string
	Subject   string
	Status    string
	Owner     string
	BlockedBy []string
	Todos     []CheckpointItem
	Source    string
}

type CheckpointGoal struct {
	Objective   string
	Status      string
	TokenBudget int64
	TokensUsed  int64
	Source      string
}

type CheckpointWorkspace struct {
	Status     string
	CWD        string
	Repository string
	Branch     string
	DirtyPaths []string
	Source     string
	Error      string
}

type CheckpointEvidence struct {
	Kind       string
	Status     string
	Target     string
	Command    string
	Detail     string
	Source     string
	ToolCallID string
}

type CheckpointToolCall struct {
	ID        string
	Tool      string
	Target    string
	Arguments string
	Status    string
	Source    string
}

type CheckpointError struct {
	Kind       string
	Message    string
	Source     string
	ToolCallID string
}

type CheckpointValidation struct {
	Command string
	Status  string
	Source  string
}

type CheckpointClaim struct {
	Text   string
	Status string
	Source string
}

// CheckpointContext is intentionally data-only. Hosts fill it from their
// authoritative stores and verified tool/filesystem evidence; SDK rendering
// applies a second entry/token bound before it reaches the summary model.
type CheckpointContext struct {
	Status           string
	Objective        CheckpointValue
	SessionTodos     []CheckpointItem
	Tasks            []CheckpointTask
	Plan             []CheckpointItem
	Goal             *CheckpointGoal
	Workspace        CheckpointWorkspace
	Evidence         []CheckpointEvidence
	PendingToolCalls []CheckpointToolCall
	RecentErrors     []CheckpointError
	Validations      []CheckpointValidation
	UnverifiedClaims []CheckpointClaim
	Warnings         []string
}

func renderCheckpointContext(snapshot CheckpointContext, maxTokens int, estimate tokenEstimator) string {
	estimate = normalizedTokenEstimator(estimate)
	if maxTokens <= 0 {
		maxTokens = DefaultCheckpointMaxTokens
	}
	status := strings.ToUpper(strings.TrimSpace(snapshot.Status))
	if status == "" {
		status = CheckpointStatusVerified
	}
	var b strings.Builder
	b.WriteString("## Host Checkpoint Context\n")
	b.WriteString("Status: ")
	b.WriteString(status)
	b.WriteByte('\n')
	writeCheckpointValue(&b, "Objective", snapshot.Objective, estimate)
	writeWorkspace(&b, snapshot.Workspace, estimate)
	writeItems(&b, "Session Todo", snapshot.SessionTodos, checkpointMaxItems, estimate)
	writeTasks(&b, snapshot.Tasks, estimate)
	writeItems(&b, "Plan", snapshot.Plan, checkpointMaxItems, estimate)
	if snapshot.Goal != nil {
		goal := snapshot.Goal
		fmt.Fprintf(&b, "Goal: status=%s objective=%s token_budget=%d tokens_used=%d source=%s\n",
			checkpointOneLine(goal.Status, checkpointFieldTokenBudget, estimate),
			checkpointOneLine(goal.Objective, checkpointDetailTokenBudget, estimate),
			goal.TokenBudget,
			goal.TokensUsed,
			checkpointSource(goal.Source),
		)
	}
	writeEvidence(&b, snapshot.Evidence, estimate)
	writePendingTools(&b, snapshot.PendingToolCalls, estimate)
	writeErrors(&b, snapshot.RecentErrors, estimate)
	writeValidations(&b, snapshot.Validations, estimate)
	writeClaims(&b, snapshot.UnverifiedClaims, estimate)
	for i, warning := range snapshot.Warnings {
		if i >= checkpointMaxErrors {
			break
		}
		fmt.Fprintf(&b, "Snapshot Warning: %s\n", checkpointOneLine(warning, checkpointDetailTokenBudget, estimate))
	}
	rendered := strings.TrimSpace(b.String())
	if estimate(rendered) <= maxTokens {
		return rendered
	}
	return truncateTextToTokenBudget(rendered, maxTokens, estimate)
}

func writeCheckpointValue(b *strings.Builder, label string, value CheckpointValue, estimate tokenEstimator) {
	if strings.TrimSpace(value.Value) == "" && strings.TrimSpace(value.Status) == "" {
		return
	}
	status := strings.ToUpper(strings.TrimSpace(value.Status))
	if status == "" {
		status = CheckpointStatusVerified
	}
	fmt.Fprintf(b, "%s: status=%s value=%s source=%s\n", label, status, checkpointOneLine(value.Value, checkpointDetailTokenBudget, estimate), checkpointSource(value.Source))
}

func writeWorkspace(b *strings.Builder, workspace CheckpointWorkspace, estimate tokenEstimator) {
	if strings.TrimSpace(workspace.Status) == "" && strings.TrimSpace(workspace.CWD) == "" && strings.TrimSpace(workspace.Repository) == "" && strings.TrimSpace(workspace.Error) == "" {
		return
	}
	status := strings.ToUpper(strings.TrimSpace(workspace.Status))
	if status == "" {
		status = CheckpointStatusVerified
	}
	fmt.Fprintf(b, "Workspace: status=%s cwd=%s repo=%s branch=%s source=%s",
		status,
		checkpointOneLine(workspace.CWD, checkpointFieldTokenBudget, estimate),
		checkpointOneLine(workspace.Repository, checkpointFieldTokenBudget, estimate),
		checkpointOneLine(workspace.Branch, checkpointFieldTokenBudget, estimate),
		checkpointSource(workspace.Source),
	)
	if strings.TrimSpace(workspace.Error) != "" {
		fmt.Fprintf(b, " error=%s", checkpointOneLine(workspace.Error, checkpointDetailTokenBudget, estimate))
	}
	b.WriteByte('\n')
	for i, path := range workspace.DirtyPaths {
		if i >= checkpointMaxItems {
			break
		}
		fmt.Fprintf(b, "- Dirty Path: %s\n", checkpointOneLine(path, checkpointFieldTokenBudget, estimate))
	}
}

func writeItems(b *strings.Builder, label string, items []CheckpointItem, limit int, estimate tokenEstimator) {
	for i, item := range items {
		if i >= limit {
			break
		}
		fmt.Fprintf(b, "- %s: id=%s status=%s text=%s source=%s\n",
			label,
			checkpointOneLine(item.ID, checkpointFieldTokenBudget, estimate),
			checkpointOneLine(item.Status, checkpointFieldTokenBudget, estimate),
			checkpointOneLine(item.Text, checkpointDetailTokenBudget, estimate),
			checkpointSource(item.Source),
		)
	}
}

func writeTasks(b *strings.Builder, tasks []CheckpointTask, estimate tokenEstimator) {
	for i, task := range tasks {
		if i >= checkpointMaxTasks {
			break
		}
		fmt.Fprintf(b, "- Task: id=%s status=%s subject=%s owner=%s blocked_by=%s source=%s\n",
			checkpointOneLine(task.ID, checkpointFieldTokenBudget, estimate),
			checkpointOneLine(task.Status, checkpointFieldTokenBudget, estimate),
			checkpointOneLine(task.Subject, checkpointDetailTokenBudget, estimate),
			checkpointOneLine(task.Owner, checkpointFieldTokenBudget, estimate),
			checkpointOneLine(strings.Join(task.BlockedBy, ","), checkpointFieldTokenBudget, estimate),
			checkpointSource(task.Source),
		)
		writeItems(b, "Task Todo "+task.ID, task.Todos, checkpointMaxTaskTodos, estimate)
	}
}

func writeEvidence(b *strings.Builder, evidence []CheckpointEvidence, estimate tokenEstimator) {
	for i, item := range evidence {
		if i >= checkpointMaxEvidence {
			break
		}
		fmt.Fprintf(b, "- Verified Evidence: kind=%s status=%s target=%s command=%s detail=%s source=%s tool_call_id=%s\n",
			checkpointOneLine(item.Kind, checkpointFieldTokenBudget, estimate),
			checkpointOneLine(item.Status, checkpointFieldTokenBudget, estimate),
			checkpointOneLine(item.Target, checkpointFieldTokenBudget, estimate),
			checkpointOneLine(item.Command, checkpointDetailTokenBudget, estimate),
			checkpointOneLine(item.Detail, checkpointDetailTokenBudget, estimate),
			checkpointSource(item.Source),
			checkpointOneLine(item.ToolCallID, checkpointFieldTokenBudget, estimate),
		)
	}
}

func writePendingTools(b *strings.Builder, calls []CheckpointToolCall, estimate tokenEstimator) {
	for i, call := range calls {
		if i >= checkpointMaxPendingTools {
			break
		}
		fmt.Fprintf(b, "- Pending Tool Call: id=%s tool=%s target=%s status=%s args=%s source=%s\n",
			checkpointOneLine(call.ID, checkpointFieldTokenBudget, estimate),
			checkpointOneLine(call.Tool, checkpointFieldTokenBudget, estimate),
			checkpointOneLine(call.Target, checkpointFieldTokenBudget, estimate),
			checkpointOneLine(call.Status, checkpointFieldTokenBudget, estimate),
			checkpointOneLine(call.Arguments, checkpointDetailTokenBudget, estimate),
			checkpointSource(call.Source),
		)
	}
}

func writeErrors(b *strings.Builder, errors []CheckpointError, estimate tokenEstimator) {
	for i, item := range errors {
		if i >= checkpointMaxErrors {
			break
		}
		fmt.Fprintf(b, "- Recent Error: kind=%s message=%s source=%s tool_call_id=%s\n",
			checkpointOneLine(item.Kind, checkpointFieldTokenBudget, estimate),
			checkpointOneLine(item.Message, checkpointDetailTokenBudget, estimate),
			checkpointSource(item.Source),
			checkpointOneLine(item.ToolCallID, checkpointFieldTokenBudget, estimate),
		)
	}
}

func writeValidations(b *strings.Builder, validations []CheckpointValidation, estimate tokenEstimator) {
	for i, item := range validations {
		if i >= checkpointMaxValidations {
			break
		}
		fmt.Fprintf(b, "- Validation: status=%s command=%s source=%s\n",
			checkpointOneLine(item.Status, checkpointFieldTokenBudget, estimate),
			checkpointOneLine(item.Command, checkpointDetailTokenBudget, estimate),
			checkpointSource(item.Source),
		)
	}
}

func writeClaims(b *strings.Builder, claims []CheckpointClaim, estimate tokenEstimator) {
	for i, item := range claims {
		if i >= checkpointMaxClaims {
			break
		}
		status := strings.ToUpper(strings.TrimSpace(item.Status))
		if status == "" {
			status = CheckpointStatusUnverified
		}
		fmt.Fprintf(b, "- %s assistant claim: %s source=%s\n", status, checkpointOneLine(item.Text, checkpointDetailTokenBudget, estimate), checkpointSource(item.Source))
	}
}

func checkpointOneLine(value string, budget int, estimate tokenEstimator) string {
	value = strings.Join(strings.Fields(strings.TrimSpace(value)), " ")
	if value == "" {
		return "-"
	}
	return truncateTextToTokenBudget(value, budget, estimate)
}

func checkpointSource(source string) string {
	source = strings.Join(strings.Fields(strings.TrimSpace(source)), " ")
	if source == "" {
		return "host"
	}
	return source
}
