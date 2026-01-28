package sandbox

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"sync"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

type SecurityError struct{ Message string }

func (e *SecurityError) Error() string { return e.Message }

type TodoItem struct {
	Content string `json:"content"`
	Status  string `json:"status"` // pending|in_progress|completed
}

type Sandbox struct {
	RootDir    string
	WorkingDir string

	mu    sync.Mutex
	Todos []TodoItem
}

func New(root string) (*Sandbox, error) {
	abs, err := filepath.Abs(root)
	if err != nil {
		return nil, err
	}
	abs = filepath.Clean(abs)
	return &Sandbox{RootDir: abs, WorkingDir: abs}, nil
}

func (s *Sandbox) Resolve(path string) (string, error) {
	if path == "" {
		return "", &SecurityError{Message: "empty path"}
	}
	var abs string
	if filepath.IsAbs(path) {
		abs = filepath.Clean(path)
	} else {
		abs = filepath.Clean(filepath.Join(s.WorkingDir, path))
	}
	abs, err := filepath.Abs(abs)
	if err != nil {
		return "", err
	}
	root := s.RootDir
	if abs == root {
		return abs, nil
	}
	sep := string(os.PathSeparator)
	if !strings.HasPrefix(abs, root+sep) {
		return "", &SecurityError{Message: fmt.Sprintf("path escapes sandbox: %q -> %q", path, abs)}
	}
	return abs, nil
}

var Key = tools.Dep[*Sandbox]("sandbox")

// Confirmer is used by interactive clients to gate potentially dangerous actions.
// Tools should call Confirm() before executing writes/commands.
type Confirmer interface {
	Confirm(ctx context.Context, action string, detail string) (bool, error)
}

var ConfirmKey = tools.Dep[Confirmer]("confirm")

type allowAll struct{}

func (allowAll) Confirm(ctx context.Context, action string, detail string) (bool, error) { return true, nil }

func getConfirmer(deps *tools.Container, ctx context.Context) Confirmer {
	c, err := tools.Get(deps, ctx, ConfirmKey)
	if err != nil {
		return allowAll{}
	}
	return c
}

// Tools returns a Claude Code-style toolset bound to the sandbox dependency.
func Tools() []tools.Tool {
	return []tools.Tool{
		bashTool(),
		readTool(),
		writeTool(),
		editTool(),
		globTool(),
		grepTool(),
		todoReadTool(),
		todoWriteTool(),
		doneTool(),
	}
}

type bashArgs struct {
	Command string `json:"command"`
	Timeout int    `json:"timeout,omitempty"` // seconds
}

func bashTool() tools.Tool {
	return tools.Func[bashArgs]("bash", "Execute a shell command and return output", func(ctx context.Context, a bashArgs, deps *tools.Container) (any, error) {
		s, err := tools.Get(deps, ctx, Key)
		if err != nil {
			return "", err
		}
		conf := getConfirmer(deps, ctx)
		ok, err := conf.Confirm(ctx, "bash", a.Command)
		if err != nil {
			return "Error: " + err.Error(), nil
		}
		if !ok {
			return "Denied", nil
		}
		timeout := a.Timeout
		if timeout <= 0 {
			timeout = 30
		}
		shell, shellArg := defaultShell()
		cctx, cancel := context.WithTimeout(ctx, time.Duration(timeout)*time.Second)
		defer cancel()
		cmd := exec.CommandContext(cctx, shell, shellArg, a.Command)
		cmd.Dir = s.WorkingDir
		out, err := cmd.CombinedOutput()
		if errors.Is(cctx.Err(), context.DeadlineExceeded) {
			return fmt.Sprintf("Command timed out after %ds", timeout), nil
		}
		res := strings.TrimSpace(string(out))
		if res == "" {
			res = "(no output)"
		}
		if err != nil {
			return res, nil
		}
		return res, nil
	})
}

func defaultShell() (exe, arg string) {
	if runtime.GOOS == "windows" {
		return "cmd", "/C"
	}
	if _, err := exec.LookPath("bash"); err == nil {
		return "bash", "-lc"
	}
	return "sh", "-lc"
}

type readArgs struct {
	FilePath string `json:"file_path"`
}

func readTool() tools.Tool {
	return tools.Func[readArgs]("read", "Read contents of a file", func(ctx context.Context, a readArgs, deps *tools.Container) (any, error) {
		s, err := tools.Get(deps, ctx, Key)
		if err != nil {
			return "", err
		}
		p, err := s.Resolve(a.FilePath)
		if err != nil {
			return "Security error: " + err.Error(), nil
		}
		st, err := os.Stat(p)
		if err != nil {
			if os.IsNotExist(err) {
				return "File not found: " + a.FilePath, nil
			}
			return "Error: " + err.Error(), nil
		}
		if st.IsDir() {
			return "Path is a directory: " + a.FilePath, nil
		}
		b, err := os.ReadFile(p)
		if err != nil {
			return "Error reading file: " + err.Error(), nil
		}
		lines := splitLines(string(b))
		out := make([]string, 0, len(lines))
		for i, line := range lines {
			out = append(out, fmt.Sprintf("%4d  %s", i+1, line))
		}
		return strings.Join(out, "\n"), nil
	})
}

func splitLines(s string) []string {
	s = strings.ReplaceAll(s, "\r\n", "\n")
	s = strings.ReplaceAll(s, "\r", "\n")
	// keep last empty line? match Python splitlines() (drops trailing empty)
	parts := strings.Split(s, "\n")
	if len(parts) > 0 && parts[len(parts)-1] == "" {
		parts = parts[:len(parts)-1]
	}
	return parts
}

type writeArgs struct {
	FilePath string `json:"file_path"`
	Content  string `json:"content"`
}

func writeTool() tools.Tool {
	return tools.Func[writeArgs]("write", "Write content to a file", func(ctx context.Context, a writeArgs, deps *tools.Container) (any, error) {
		s, err := tools.Get(deps, ctx, Key)
		if err != nil {
			return "", err
		}
		conf := getConfirmer(deps, ctx)
		ok, err := conf.Confirm(ctx, "write", fmt.Sprintf("%s (%d bytes)", a.FilePath, len(a.Content)))
		if err != nil {
			return "Error: " + err.Error(), nil
		}
		if !ok {
			return "Denied", nil
		}
		p, err := s.Resolve(a.FilePath)
		if err != nil {
			return "Security error: " + err.Error(), nil
		}
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			return "Error writing file: " + err.Error(), nil
		}
		if err := os.WriteFile(p, []byte(a.Content), 0o644); err != nil {
			return "Error writing file: " + err.Error(), nil
		}
		return fmt.Sprintf("Wrote %d bytes to %s", len(a.Content), a.FilePath), nil
	})
}

type editArgs struct {
	FilePath  string `json:"file_path"`
	OldString string `json:"old_string"`
	NewString string `json:"new_string"`
}

func editTool() tools.Tool {
	return tools.Func[editArgs]("edit", "Replace text in a file", func(ctx context.Context, a editArgs, deps *tools.Container) (any, error) {
		s, err := tools.Get(deps, ctx, Key)
		if err != nil {
			return "", err
		}
		conf := getConfirmer(deps, ctx)
		ok, err := conf.Confirm(ctx, "edit", a.FilePath)
		if err != nil {
			return "Error: " + err.Error(), nil
		}
		if !ok {
			return "Denied", nil
		}
		p, err := s.Resolve(a.FilePath)
		if err != nil {
			return "Security error: " + err.Error(), nil
		}
		b, err := os.ReadFile(p)
		if err != nil {
			if os.IsNotExist(err) {
				return "File not found: " + a.FilePath, nil
			}
			return "Error editing file: " + err.Error(), nil
		}
		content := string(b)
		if !strings.Contains(content, a.OldString) {
			return "String not found in " + a.FilePath, nil
		}
		count := strings.Count(content, a.OldString)
		newContent := strings.ReplaceAll(content, a.OldString, a.NewString)
		if err := os.WriteFile(p, []byte(newContent), 0o644); err != nil {
			return "Error editing file: " + err.Error(), nil
		}
		return fmt.Sprintf("Replaced %d occurrence(s) in %s", count, a.FilePath), nil
	})
}

type globArgs struct {
	Pattern string `json:"pattern"`
	Path    string `json:"path,omitempty"`
}

func globTool() tools.Tool {
	return tools.Func[globArgs]("glob", "Find files matching a glob pattern", func(ctx context.Context, a globArgs, deps *tools.Container) (any, error) {
		s, err := tools.Get(deps, ctx, Key)
		if err != nil {
			return "", err
		}
		base := s.WorkingDir
		if strings.TrimSpace(a.Path) != "" {
			p, err := s.Resolve(a.Path)
			if err != nil {
				return "Security error: " + err.Error(), nil
			}
			base = p
		}
		matches, err := filepath.Glob(filepath.Join(base, a.Pattern))
		if err != nil {
			return "Error: " + err.Error(), nil
		}
		files := []string{}
		for _, m := range matches {
			st, err := os.Stat(m)
			if err != nil || st.IsDir() {
				continue
			}
			rel, _ := filepath.Rel(s.RootDir, m)
			files = append(files, filepath.ToSlash(rel))
			if len(files) >= 50 {
				break
			}
		}
		if len(files) == 0 {
			return "No files match pattern: " + a.Pattern, nil
		}
		return fmt.Sprintf("Found %d file(s):\n%s", len(files), strings.Join(files, "\n")), nil
	})
}

type grepArgs struct {
	Pattern string `json:"pattern"`
	Path    string `json:"path,omitempty"`
}

func grepTool() tools.Tool {
	return tools.Func[grepArgs]("grep", "Search file contents with regex", func(ctx context.Context, a grepArgs, deps *tools.Container) (any, error) {
		s, err := tools.Get(deps, ctx, Key)
		if err != nil {
			return "", err
		}
		base := s.WorkingDir
		if strings.TrimSpace(a.Path) != "" {
			p, err := s.Resolve(a.Path)
			if err != nil {
				return "Security error: " + err.Error(), nil
			}
			base = p
		}
		re, err := regexp.Compile(a.Pattern)
		if err != nil {
			return "Invalid regex: " + err.Error(), nil
		}
		results := []string{}
		_ = filepath.WalkDir(base, func(path string, d os.DirEntry, err error) error {
			if err != nil || d.IsDir() {
				return nil
			}
			f, err := os.Open(path)
			if err != nil {
				return nil
			}
			defer f.Close()
			buf := make([]byte, 8000)
			n, _ := f.Read(buf)
			if bytes.IndexByte(buf[:n], 0) >= 0 {
				return nil // skip binary
			}
			if _, err := f.Seek(0, io.SeekStart); err != nil {
				return nil
			}
			scanner := bufio.NewScanner(f)
			lineNo := 0
			for scanner.Scan() {
				lineNo++
				line := scanner.Text()
				if re.MatchString(line) {
					rel, _ := filepath.Rel(s.RootDir, path)
					results = append(results, fmt.Sprintf("%s:%d: %s", filepath.ToSlash(rel), lineNo, truncate(line, 100)))
					if len(results) >= 50 {
						return errors.New("_stop")
					}
				}
			}
			return nil
		})
		if len(results) == 0 {
			return "No matches for: " + a.Pattern, nil
		}
		return strings.Join(results, "\n"), nil
	})
}

func truncate(s string, max int) string {
	if len(s) <= max {
		return s
	}
	if max <= 3 {
		return s[:max]
	}
	return s[:max-3] + "..."
}

type todoWriteArgs struct {
	Todos []TodoItem `json:"todos"`
}

func todoReadTool() tools.Tool {
	return tools.Func[struct{}]("todo_read", "Read current todo list", func(ctx context.Context, _ struct{}, deps *tools.Container) (any, error) {
		s, err := tools.Get(deps, ctx, Key)
		if err != nil {
			return "", err
		}
		s.mu.Lock()
		defer s.mu.Unlock()
		if len(s.Todos) == 0 {
			return "Todo list is empty", nil
		}
		lines := []string{}
		for i, t := range s.Todos {
			status := map[string]string{"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}[t.Status]
			if status == "" {
				status = "[ ]"
			}
			lines = append(lines, fmt.Sprintf("%d. %s %s", i+1, status, t.Content))
		}
		return strings.Join(lines, "\n"), nil
	})
}

func todoWriteTool() tools.Tool {
	return tools.Func[todoWriteArgs]("todo_write", "Update the todo list", func(ctx context.Context, a todoWriteArgs, deps *tools.Container) (any, error) {
		s, err := tools.Get(deps, ctx, Key)
		if err != nil {
			return "", err
		}
		s.mu.Lock()
		s.Todos = append([]TodoItem(nil), a.Todos...)
		s.mu.Unlock()
		stats := map[string]int{"pending": 0, "in_progress": 0, "completed": 0}
		for _, t := range a.Todos {
			stats[t.Status]++
		}
		return fmt.Sprintf("Updated todos: %d pending, %d in progress, %d completed", stats["pending"], stats["in_progress"], stats["completed"]), nil
	})
}

type doneArgs struct {
	Message string `json:"message"`
}

func doneTool() tools.Tool {
	return tools.Func[doneArgs]("done", "Signal that the task is complete", func(ctx context.Context, a doneArgs, _ *tools.Container) (any, error) {
		return "", tools.TaskComplete(a.Message)
	}).WithEphemeralKeep(0)
}

// Multimodal results can be added later using llm.Content blocks.
func ToolResultText(text string) llm.Content { return llm.TextContent(text) }
