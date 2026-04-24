package tools

import (
	"context"
	"fmt"
	"strings"
)

type invalidArgs struct {
	Tool  string `json:"tool"`
	Error string `json:"error"`
}

// InvalidTool returns a lightweight internal tool for reporting invalid tool calls.
func InvalidTool() Tool {
	t := Func[invalidArgs]("invalid", "Report an invalid tool call or schema mismatch.", func(_ context.Context, a invalidArgs, _ *Container) (any, error) {
		tool := strings.TrimSpace(a.Tool)
		errText := strings.TrimSpace(a.Error)
		if errText == "" {
			errText = "invalid tool call"
		}
		title := "invalid"
		if tool != "" {
			title = fmt.Sprintf("invalid: %s", tool)
		}
		return map[string]any{
			"title": title,
			"metadata": map[string]any{
				"invalid": true,
				"tool":    tool,
				"error":   errText,
			},
			"output": errText,
		}, nil
	})
	t.Hidden = true
	return t
}
