package sandbox

import (
	"context"
	"fmt"
	"strings"

	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

// externalDirectoryArgs holds the arguments for the external_directory tool.
type externalDirectoryArgs struct {
	Path string `json:"path"`
}

// externalDirectoryTool returns a tool that allows access to a path outside the sandbox root.
// The tool requires confirmation before adding an external path to the allowlist.
func externalDirectoryTool() tools.Tool {
	return tools.Func[externalDirectoryArgs]("external_directory", "Allow access to a path outside the sandbox root", func(ctx context.Context, a externalDirectoryArgs, deps *tools.Container) (any, error) {
		s, err := tools.Get(deps, ctx, Key)
		if err != nil {
			return "", err
		}
		raw := strings.TrimSpace(a.Path)
		if raw == "" {
			return "", fmt.Errorf("missing path")
		}
		normalized, err := s.normalizeExternalRoot(raw)
		if err != nil {
			return "", err
		}
		if isWithinRoot(normalized, s.RootDir) {
			return fmt.Sprintf("Path is already inside the sandbox root: %s", normalized), nil
		}
		if s.isAllowedExternalRoot(normalized) {
			return fmt.Sprintf("External path already allowed: %s", normalized), nil
		}

		conf := getConfirmer(deps, ctx)
		meta := attachToolCallMeta(ctx, map[string]any{
			"category":  "external_directory",
			"summary":   fmt.Sprintf("Allow external path: %s", normalized),
			"file_path": normalized,
			"path":      normalized,
			"raw":       normalized,
		})
		ok, err := conf.Confirm(ctx, "external_directory", buildConfirmDetail(meta))
		if err != nil {
			return "", err
		}
		if !ok {
			denied, denyErr := denyToolResult(ctx, "external_directory", "user denied request")
			return denied.PlainText(), denyErr
		}

		finalPath, added, err := s.AllowExternalDirectory(normalized)
		if err != nil {
			return "", err
		}
		if !added {
			return fmt.Sprintf("External path already allowed: %s", finalPath), nil
		}
		return fmt.Sprintf("Allowed external path: %s", finalPath), nil
	})
}
