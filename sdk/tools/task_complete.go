package tools

import "fmt"

// TaskCompleteError signals explicit task completion (the "done tool" pattern).
// Tool handlers can return this error to stop the agent loop.
type TaskCompleteError struct{ Message string }

func (e *TaskCompleteError) Error() string { return fmt.Sprintf("task complete: %s", e.Message) }

func TaskComplete(message string) error { return &TaskCompleteError{Message: message} }
