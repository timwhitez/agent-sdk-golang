package sandbox

import (
	"context"
	"errors"
	"fmt"
	"os/exec"
	"runtime"
	"strings"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools/execenv"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools/execrunner"
)

// bashArgs holds the arguments for the bash tool.
type bashArgs struct {
	Command string `json:"command"`
	Timeout int    `json:"timeout,omitempty"` // seconds
}

// bashTool returns a tool that executes shell commands.
func bashTool() tools.Tool {
	return toolWithArgs[bashArgs]("bash", "Execute a shell command and return output", func(ctx context.Context, a bashArgs, deps *tools.Container) (llm.Content, error) {
		s, err := tools.Get(deps, ctx, Key)
		if err != nil {
			return llm.TextContent(""), err
		}
		conf := getConfirmer(deps, ctx)
		cmd0 := strings.TrimSpace(a.Command)
		timeout, timeoutDuration, err := checkedSandboxTimeout(a.Timeout, 30)
		if err != nil {
			msg := formatErrorDiagnosticFromErr("Invalid bash timeout", err, fmt.Sprintf("Use a timeout from 1 to %d seconds and retry.", maxSandboxTimeoutSeconds))
			return llm.TextContent(msg), err
		}
		meta := attachToolCallMeta(ctx, map[string]any{
			"category": "exec",
			"summary":  truncateForMeta(truncateOneLine(cmd0, 240), 400),
			"command":  cmd0,
			"raw":      cmd0,
		})
		workdirAccessPath, execDir, err := s.resolveCommandWorkingDirAccessPath()
		if err != nil {
			var secErr *SecurityError
			if errors.As(err, &secErr) {
				msg := formatErrorDiagnosticFromErr("Security error", err, "Use a working directory inside the sandbox root and retry.")
				return llm.TextContent(msg), err
			}
			msg := formatErrorDiagnosticFromErr("Unable to use current working directory", err, "Check sandbox working directory configuration and retry.")
			return llm.TextContent(msg), err
		}
		meta["workdir"] = execDir
		ok, err := conf.Confirm(ctx, "bash", buildConfirmDetail(meta))
		if err != nil {
			msg := formatErrorDiagnosticFromErr("bash confirmation failed", err, "Retry after confirmation policy is available.")
			return llm.TextContent(msg), err
		}
		if !ok {
			return denyToolResult(ctx, "bash", "user denied request")
		}
		resolvedExecDir, err := s.RevalidateAccessPath(workdirAccessPath)
		if err != nil {
			var secErr *SecurityError
			if errors.As(err, &secErr) {
				msg := formatErrorDiagnosticFromErr("Security error", err, "Use a working directory inside the sandbox root and retry.")
				return llm.TextContent(msg), err
			}
			msg := formatErrorDiagnosticFromErr("Unable to use current working directory", err, "Check sandbox working directory configuration and retry.")
			return llm.TextContent(msg), err
		}
		execDir = strings.TrimSpace(resolvedExecDir)
		shell, shellArg := defaultShell()
		runRes, err := execrunner.Run(ctx, execrunner.Options{
			Program:        shell,
			Args:           []string{shellArg, cmd0},
			Dir:            execDir,
			Env:            execenv.EnvFromDeps(ctx, deps),
			Timeout:        timeoutDuration,
			MaxOutputBytes: execrunner.DefaultMaxOutputBytes,
			ArtifactPrefix: "sdk-bash-output-*.log",
		})

		metaOut := map[string]any{
			"exit_code":     runRes.ExitCode,
			"timed_out":     runRes.TimedOut,
			"timeout_s":     timeout,
			"output_bytes":  runRes.OutputBytes,
			"output_capped": runRes.OutputTruncated,
		}
		if runRes.OutputTruncated {
			metaOut["output_bytes_limit"] = execrunner.DefaultMaxOutputBytes
			if strings.TrimSpace(runRes.OutputPath) != "" {
				metaOut["output_path"] = strings.TrimSpace(runRes.OutputPath)
			}
			if strings.TrimSpace(runRes.OutputArtifactErr) != "" {
				metaOut["output_artifact_error"] = strings.TrimSpace(runRes.OutputArtifactErr)
			}
		}
		tools.UpsertToolResultMetadata(ctx, metaOut)

		res := strings.TrimSpace(runRes.Output)
		if runRes.OutputTruncated {
			notice := fmt.Sprintf("... (output truncated after %d bytes", execrunner.DefaultMaxOutputBytes)
			if strings.TrimSpace(runRes.OutputPath) != "" {
				notice += fmt.Sprintf("; full output saved to %s", strings.TrimSpace(runRes.OutputPath))
			}
			if strings.TrimSpace(runRes.OutputArtifactErr) != "" {
				notice += fmt.Sprintf("; failed to persist full output: %s", strings.TrimSpace(runRes.OutputArtifactErr))
			}
			notice += ")"
			if res == "" {
				res = notice
			} else {
				res = strings.TrimSpace(res + "\n" + notice)
			}
		}
		if res == "" {
			res = "(no output)"
		}
		if runRes.TimedOut {
			return llm.TextContent(fmt.Sprintf("Command timed out after %ds.\n%s", timeout, res)), context.DeadlineExceeded
		}
		if err != nil {
			return llm.TextContent(res), err
		}
		return llm.TextContent(res), nil
	})
}

// defaultShell returns the appropriate shell executable and argument for the current platform.
func defaultShell() (exe, arg string) {
	if runtime.GOOS == "windows" {
		return "cmd", "/C"
	}
	if _, err := exec.LookPath("bash"); err == nil {
		return "bash", "-lc"
	}
	return "sh", "-lc"
}
