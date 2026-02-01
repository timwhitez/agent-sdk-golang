package sandbox

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/bmatcuk/doublestar/v4"

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

	allowedRoots []string
}

func (s *Sandbox) TodosSnapshot() []TodoItem {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]TodoItem, len(s.Todos))
	copy(out, s.Todos)
	return out
}

func (s *Sandbox) ReplaceTodos(todos []TodoItem) {
	if s == nil {
		return
	}
	s.mu.Lock()
	s.Todos = append([]TodoItem(nil), todos...)
	s.mu.Unlock()
}

// AllowedRootsSnapshot returns the currently allowed external roots.
func (s *Sandbox) AllowedRootsSnapshot() []string {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]string, len(s.allowedRoots))
	copy(out, s.allowedRoots)
	return out
}

// ReplaceAllowedRoots replaces the external allowed roots with a cleaned list.
// The roots are assumed to already be normalized (absolute, canonical).
func (s *Sandbox) ReplaceAllowedRoots(roots []string) {
	if s == nil {
		return
	}
	cleaned := make([]string, 0, len(roots))
	seen := map[string]bool{}
	base := strings.TrimSpace(s.RootDir)
	if base == "" {
		base = strings.TrimSpace(s.WorkingDir)
	}
	for _, r := range roots {
		r = strings.TrimSpace(r)
		if r == "" {
			continue
		}
		if !filepath.IsAbs(r) {
			if base != "" {
				r = filepath.Join(base, r)
			}
		}
		r = filepath.Clean(r)
		if r == "" {
			continue
		}
		if seen[r] {
			continue
		}
		seen[r] = true
		cleaned = append(cleaned, r)
	}
	s.mu.Lock()
	s.allowedRoots = cleaned
	s.mu.Unlock()
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
	if s.isAllowedPath(abs) {
		return abs, nil
	}
	return "", &SecurityError{Message: fmt.Sprintf("path escapes sandbox: %q -> %q", path, abs)}
}

func isWithinRoot(path, root string) bool {
	path = filepath.Clean(path)
	root = filepath.Clean(root)
	if path == "" || root == "" {
		return false
	}
	if path == root {
		return true
	}
	sep := string(os.PathSeparator)
	if root == sep {
		return strings.HasPrefix(path, root)
	}
	return strings.HasPrefix(path, root+sep)
}

func (s *Sandbox) isAllowedPath(abs string) bool {
	if s == nil {
		return false
	}
	if isWithinRoot(abs, s.RootDir) {
		return true
	}
	s.mu.Lock()
	allowed := append([]string(nil), s.allowedRoots...)
	s.mu.Unlock()
	for _, root := range allowed {
		if isWithinRoot(abs, root) {
			return true
		}
	}
	return false
}

func (s *Sandbox) isAllowedExternalRoot(abs string) bool {
	if s == nil {
		return false
	}
	s.mu.Lock()
	allowed := append([]string(nil), s.allowedRoots...)
	s.mu.Unlock()
	for _, root := range allowed {
		if isWithinRoot(abs, root) {
			return true
		}
	}
	return false
}

func (s *Sandbox) normalizeExternalRoot(path string) (string, error) {
	if s == nil {
		return "", fmt.Errorf("nil sandbox")
	}
	p := strings.TrimSpace(path)
	if p == "" {
		return "", fmt.Errorf("empty path")
	}
	var abs string
	if filepath.IsAbs(p) {
		abs = filepath.Clean(p)
	} else {
		base := s.WorkingDir
		if strings.TrimSpace(base) == "" {
			base = s.RootDir
		}
		abs = filepath.Clean(filepath.Join(base, p))
	}
	abs, err := filepath.Abs(abs)
	if err != nil {
		return "", err
	}
	resolved, err := filepath.EvalSymlinks(abs)
	if err != nil {
		if os.IsNotExist(err) {
			return "", fmt.Errorf("path does not exist: %s", abs)
		}
		return "", err
	}
	resolved = filepath.Clean(resolved)
	st, err := os.Stat(resolved)
	if err != nil {
		if os.IsNotExist(err) {
			return "", fmt.Errorf("path does not exist: %s", resolved)
		}
		return "", err
	}
	if !st.IsDir() {
		return "", fmt.Errorf("path is not a directory: %s", resolved)
	}
	return resolved, nil
}

// AllowExternalDirectory adds an external directory to the sandbox allowlist.
// It returns the normalized path, whether it was newly added, and any error.
func (s *Sandbox) AllowExternalDirectory(path string) (string, bool, error) {
	normalized, err := s.normalizeExternalRoot(path)
	if err != nil {
		return "", false, err
	}
	if isWithinRoot(normalized, s.RootDir) {
		return normalized, false, nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, root := range s.allowedRoots {
		if isWithinRoot(normalized, root) {
			return normalized, false, nil
		}
	}
	s.allowedRoots = append(s.allowedRoots, normalized)
	return normalized, true, nil
}

var Key = tools.Dep[*Sandbox]("sandbox")

// Confirmer is used by interactive clients to gate potentially dangerous actions.
// Tools should call Confirm() before executing writes/commands.
type Confirmer interface {
	Confirm(ctx context.Context, action string, detail string) (bool, error)
}

var ConfirmKey = tools.Dep[Confirmer]("confirm")

type allowAll struct{}

func (allowAll) Confirm(ctx context.Context, action string, detail string) (bool, error) {
	return true, nil
}

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
		bashTool().WithEphemeralKeep(1),
		lsTool().WithEphemeralKeep(1),
		readTool().WithEphemeralKeep(1),
		webfetchTool().WithEphemeralKeep(1),
		writeTool().WithEphemeralKeep(1),
		editTool().WithEphemeralKeep(1),
		multieditTool().WithEphemeralKeep(1),
		applyPatchTool().WithEphemeralKeep(1),
		globTool().WithEphemeralKeep(1),
		externalDirectoryTool().WithEphemeralKeep(1),
		grepTool().WithEphemeralKeep(1),
		// Preferred (opencode-compatible) names.
		todoReadToolNamed("todoread"),
		todoWriteToolNamed("todowrite"),
		// Backward-compatible aliases.
		todoReadToolNamed("todo_read"),
		todoWriteToolNamed("todo_write"),
		doneTool(),
	}
}

func attachToolCallMeta(ctx context.Context, meta map[string]any) map[string]any {
	if meta == nil {
		meta = map[string]any{}
	}
	if id := tools.ToolCallID(ctx); id != "" {
		meta["tool_call_id"] = id
	}
	return meta
}

type webfetchArgs struct {
	URL      string            `json:"url"`
	Method   string            `json:"method,omitempty"`  // GET|HEAD (default GET)
	Headers  map[string]string `json:"headers,omitempty"` // best-effort
	Timeout  int               `json:"timeout,omitempty"` // seconds
	MaxBytes int               `json:"max_bytes,omitempty"`
}

func webfetchTool() tools.Tool {
	return tools.Func[webfetchArgs]("webfetch", "Fetch a URL over HTTP(S) and return the response body (best-effort)", func(ctx context.Context, a webfetchArgs, deps *tools.Container) (any, error) {
		conf := getConfirmer(deps, ctx)
		rawURL := strings.TrimSpace(a.URL)
		if rawURL == "" {
			return "Error: missing url", nil
		}
		u, err := url.Parse(rawURL)
		if err != nil {
			return "Error: invalid url", nil
		}
		scheme := strings.ToLower(strings.TrimSpace(u.Scheme))
		if scheme != "http" && scheme != "https" {
			return "Error: only http/https is supported", nil
		}
		method := strings.ToUpper(strings.TrimSpace(a.Method))
		if method == "" {
			method = http.MethodGet
		}
		if method != http.MethodGet && method != http.MethodHead {
			return "Error: only GET/HEAD is supported", nil
		}
		timeout := a.Timeout
		if timeout <= 0 {
			timeout = 30
		}
		maxBytes := a.MaxBytes
		if maxBytes <= 0 {
			maxBytes = 1024 * 1024
		}
		if maxBytes > 5*1024*1024 {
			maxBytes = 5 * 1024 * 1024
		}

		meta := attachToolCallMeta(ctx, map[string]any{
			"category": "network",
			"summary":  fmt.Sprintf("%s %s", method, rawURL),
			"url":      rawURL,
			"raw":      fmt.Sprintf("%s %s (timeout=%ds, max_bytes=%d)", method, rawURL, timeout, maxBytes),
		})
		ok, err := conf.Confirm(ctx, "webfetch", buildConfirmDetail(meta))
		if err != nil {
			return "Error: " + err.Error(), err
		}
		if !ok {
			return "Denied", nil
		}

		hc := &http.Client{Timeout: time.Duration(timeout) * time.Second}
		req, err := http.NewRequestWithContext(ctx, method, rawURL, nil)
		if err != nil {
			return "Error: " + err.Error(), nil
		}
		for k, v := range a.Headers {
			kk := strings.TrimSpace(k)
			vv := strings.TrimSpace(v)
			if kk != "" && vv != "" {
				req.Header.Set(kk, vv)
			}
		}
		resp, err := hc.Do(req)
		if err != nil {
			return "Error: " + err.Error(), nil
		}
		defer func() { _ = resp.Body.Close() }()

		var body []byte
		if method != http.MethodHead {
			body, _ = io.ReadAll(io.LimitReader(resp.Body, int64(maxBytes)))
		}
		text := strings.TrimSpace(string(body))
		if text == "" {
			text = "(no body)"
		}
		return fmt.Sprintf("%s\n\n%s", resp.Status, text), nil
	})
}

type lsArgs struct {
	Path   string   `json:"path,omitempty"`
	Ignore []string `json:"ignore,omitempty"`
}

func lsTool() tools.Tool {
	return tools.Func[lsArgs]("ls", "List files and directories in a given path", func(ctx context.Context, a lsArgs, deps *tools.Container) (any, error) {
		s, err := tools.Get(deps, ctx, Key)
		if err != nil {
			return "", err
		}
		p := strings.TrimSpace(a.Path)
		if p == "" {
			p = "."
		}
		abs, err := s.Resolve(p)
		if err != nil {
			return "Security error: " + err.Error(), nil
		}
		st, err := os.Stat(abs)
		if err != nil {
			return "Error: " + err.Error(), nil
		}
		if !st.IsDir() {
			return "Path is not a directory: " + p, nil
		}
		ents, err := os.ReadDir(abs)
		if err != nil {
			return "Error: " + err.Error(), nil
		}
		ignore := make([]string, 0, len(a.Ignore))
		for _, ig := range a.Ignore {
			ig = strings.TrimSpace(ig)
			if ig != "" {
				ignore = append(ignore, ig)
			}
		}
		matchIgnore := func(name string) bool {
			if len(ignore) == 0 {
				return false
			}
			for _, pat := range ignore {
				ok, _ := doublestar.Match(pat, name)
				if ok {
					return true
				}
			}
			return false
		}
		items := []string{}
		for _, e := range ents {
			name := e.Name()
			if matchIgnore(name) {
				continue
			}
			if e.IsDir() {
				name += "/"
			}
			items = append(items, name)
			if len(items) >= 200 {
				break
			}
		}
		sort.Strings(items)
		if len(items) == 0 {
			return "(empty)", nil
		}
		return strings.Join(items, "\n"), nil
	})
}

func toolWithArgs[Args any](name, description string, fn func(ctx context.Context, args Args, deps *tools.Container) (llm.Content, error)) tools.Tool {
	schema := tools.SchemaFor[Args]()
	return tools.Tool{
		Name:        name,
		Description: description,
		Schema:      schema,
		Handler: func(ctx context.Context, raw json.RawMessage, deps *tools.Container) (llm.Content, error) {
			var a Args
			dec := json.NewDecoder(bytes.NewReader(raw))
			dec.DisallowUnknownFields()
			if err := dec.Decode(&a); err != nil {
				return llm.TextContent(fmt.Sprintf("Error parsing arguments: %v", err)), err
			}
			return fn(ctx, a, deps)
		},
	}
}

type bashArgs struct {
	Command string `json:"command"`
	Timeout int    `json:"timeout,omitempty"` // seconds
}

func bashTool() tools.Tool {
	return toolWithArgs[bashArgs]("bash", "Execute a shell command and return output", func(ctx context.Context, a bashArgs, deps *tools.Container) (llm.Content, error) {
		s, err := tools.Get(deps, ctx, Key)
		if err != nil {
			return llm.TextContent(""), err
		}
		conf := getConfirmer(deps, ctx)
		cmd0 := strings.TrimSpace(a.Command)
		meta := attachToolCallMeta(ctx, map[string]any{
			"category": "exec",
			"summary":  truncateForMeta(truncateOneLine(cmd0, 240), 400),
			"command":  cmd0,
			"raw":      cmd0,
		})
		ok, err := conf.Confirm(ctx, "bash", buildConfirmDetail(meta))
		if err != nil {
			return llm.TextContent("Error: " + err.Error()), err
		}
		if !ok {
			// User denied: do not surface as tool error in the UI.
			return llm.TextContent("Denied"), nil
		}
		timeout := a.Timeout
		if timeout <= 0 {
			timeout = 30
		}
		shell, shellArg := defaultShell()
		cctx, cancel := context.WithTimeout(ctx, time.Duration(timeout)*time.Second)
		defer cancel()
		cmd := exec.CommandContext(cctx, shell, shellArg, cmd0)
		cmd.Dir = s.WorkingDir
		out, err := cmd.CombinedOutput()
		if errors.Is(cctx.Err(), context.DeadlineExceeded) {
			return llm.TextContent(fmt.Sprintf("Command timed out after %ds", timeout)), context.DeadlineExceeded
		}
		res := strings.TrimSpace(string(out))
		if res == "" {
			res = "(no output)"
		}
		if err != nil {
			return llm.TextContent(res), err
		}
		return llm.TextContent(res), nil
	})
}

func truncateOneLine(s string, max int) string {
	s = strings.ReplaceAll(s, "\r\n", "\n")
	s = strings.ReplaceAll(s, "\r", "\n")
	s = strings.ReplaceAll(s, "\n", " ")
	s = strings.Join(strings.Fields(s), " ")
	return truncateForMeta(s, max)
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
	Offset   int    `json:"offset,omitempty"` // 1-based line offset
	Limit    int    `json:"limit,omitempty"`  // number of lines
}

func readTool() tools.Tool {
	return toolWithArgs[readArgs]("read", "Read contents of a file", func(ctx context.Context, a readArgs, deps *tools.Container) (llm.Content, error) {
		s, err := tools.Get(deps, ctx, Key)
		if err != nil {
			return llm.TextContent(""), err
		}
		p, err := s.Resolve(a.FilePath)
		if err != nil {
			return llm.TextContent("Security error: " + err.Error()), err
		}
		st, err := os.Stat(p)
		if err != nil {
			if os.IsNotExist(err) {
				return llm.TextContent("File not found: " + a.FilePath), err
			}
			return llm.TextContent("Error: " + err.Error()), err
		}
		if st.IsDir() {
			return llm.TextContent("Path is a directory: " + a.FilePath), fmt.Errorf("is a directory")
		}
		b, err := os.ReadFile(p)
		if err != nil {
			return llm.TextContent("Error reading file: " + err.Error()), err
		}
		lines := splitLines(string(b))
		offset := a.Offset
		if offset <= 0 {
			offset = 1
		}
		limit := a.Limit
		if limit <= 0 {
			limit = 2000
		}
		start := offset - 1
		if start < 0 {
			start = 0
		}
		if start > len(lines) {
			start = len(lines)
		}
		end := start + limit
		if end > len(lines) {
			end = len(lines)
		}
		out := make([]string, 0, end-start)
		for i := start; i < end; i++ {
			out = append(out, fmt.Sprintf("%4d  %s", i+1, lines[i]))
		}
		if len(out) == 0 {
			return llm.TextContent("(no content)"), nil
		}
		return llm.TextContent(strings.Join(out, "\n")), nil
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

const (
	maxConfirmDiffChars = 120_000
	maxConfirmRawChars  = 40_000
	maxConfirmDiffLines = 2_000
)

func truncateForMeta(s string, max int) string {
	s = strings.TrimSpace(s)
	if max <= 0 || len(s) <= max {
		return s
	}
	if max <= 3 {
		return s[:max]
	}
	return s[:max-3] + "..."
}

func buildConfirmDetail(meta map[string]any) string {
	if meta == nil {
		return ""
	}
	if v, ok := meta["diff"].(string); ok {
		meta["diff"] = truncateForMeta(v, maxConfirmDiffChars)
	}
	if v, ok := meta["raw"].(string); ok {
		meta["raw"] = truncateForMeta(v, maxConfirmRawChars)
	}
	b, err := json.Marshal(meta)
	if err != nil {
		// Fallback: keep confirm usable even if meta is malformed.
		if s, ok := meta["summary"].(string); ok {
			return s
		}
		return "(confirm)"
	}
	return string(b)
}

// fullReplaceDiff returns a unified-diff-like preview by treating the change as a full-file replacement.
// It's meant for human preview in interactive clients (not for applying).
func fullReplaceDiff(filePath, oldContent, newContent string) string {
	oldLines := splitLines(oldContent)
	newLines := splitLines(newContent)
	oldN := len(oldLines)
	newN := len(newLines)

	var b strings.Builder
	b.WriteString("--- a/")
	b.WriteString(strings.TrimSpace(filePath))
	b.WriteString("\n+++ b/")
	b.WriteString(strings.TrimSpace(filePath))
	b.WriteString("\n")

	// Hunk header.
	if oldN == 0 && newN == 0 {
		b.WriteString("@@ -0,0 +0,0 @@\n")
		return b.String()
	}
	oldStart := 1
	newStart := 1
	if oldN == 0 {
		oldStart = 0
	}
	if newN == 0 {
		newStart = 0
	}
	b.WriteString(fmt.Sprintf("@@ -%d,%d +%d,%d @@\n", oldStart, oldN, newStart, newN))

	// Emit lines (bounded).
	lineBudget := maxConfirmDiffLines
	for _, l := range oldLines {
		if lineBudget <= 0 {
			b.WriteString("... (diff truncated)\n")
			return b.String()
		}
		b.WriteString("-")
		b.WriteString(l)
		b.WriteString("\n")
		lineBudget--
	}
	for _, l := range newLines {
		if lineBudget <= 0 {
			b.WriteString("... (diff truncated)\n")
			return b.String()
		}
		b.WriteString("+")
		b.WriteString(l)
		b.WriteString("\n")
		lineBudget--
	}
	return b.String()
}

type writeArgs struct {
	FilePath string `json:"file_path"`
	Content  string `json:"content"`
}

func writeTool() tools.Tool {
	return toolWithArgs[writeArgs]("write", "Write content to a file", func(ctx context.Context, a writeArgs, deps *tools.Container) (llm.Content, error) {
		s, err := tools.Get(deps, ctx, Key)
		if err != nil {
			return llm.TextContent(""), err
		}
		conf := getConfirmer(deps, ctx)
		p, err := s.Resolve(a.FilePath)
		if err != nil {
			return llm.TextContent("Security error: " + err.Error()), err
		}
		// Build confirm meta with a diff preview.
		raw := fmt.Sprintf("%s (%d bytes)", a.FilePath, len(a.Content))
		oldContent := ""
		if st, err := os.Stat(p); err == nil && !st.IsDir() {
			if b, err := os.ReadFile(p); err == nil {
				oldContent = string(b)
			}
		}
		diff := fullReplaceDiff(a.FilePath, oldContent, a.Content)
		meta := attachToolCallMeta(ctx, map[string]any{
			"category":  "filesystem_write",
			"summary":   raw,
			"file_path": strings.TrimSpace(a.FilePath),
			"diff":      diff,
			"raw":       raw,
		})
		ok, err := conf.Confirm(ctx, "write", buildConfirmDetail(meta))
		if err != nil {
			return llm.TextContent("Error: " + err.Error()), err
		}
		if !ok {
			return llm.TextContent("Denied"), nil
		}
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			return llm.TextContent("Error writing file: " + err.Error()), err
		}
		if err := os.WriteFile(p, []byte(a.Content), 0o644); err != nil {
			return llm.TextContent("Error writing file: " + err.Error()), err
		}
		return llm.TextContent(fmt.Sprintf("Wrote %d bytes to %s", len(a.Content), a.FilePath)), nil
	})
}

type editArgs struct {
	FilePath  string `json:"file_path"`
	OldString string `json:"old_string"`
	NewString string `json:"new_string"`
}

func editTool() tools.Tool {
	return toolWithArgs[editArgs]("edit", "Replace text in a file", func(ctx context.Context, a editArgs, deps *tools.Container) (llm.Content, error) {
		s, err := tools.Get(deps, ctx, Key)
		if err != nil {
			return llm.TextContent(""), err
		}
		conf := getConfirmer(deps, ctx)
		p, err := s.Resolve(a.FilePath)
		if err != nil {
			return llm.TextContent("Security error: " + err.Error()), err
		}
		b, err := os.ReadFile(p)
		if err != nil {
			if os.IsNotExist(err) {
				return llm.TextContent("File not found: " + a.FilePath), err
			}
			return llm.TextContent("Error editing file: " + err.Error()), err
		}
		content := string(b)
		if !strings.Contains(content, a.OldString) {
			return llm.TextContent("String not found in " + a.FilePath), fmt.Errorf("string not found")
		}
		count := strings.Count(content, a.OldString)
		newContent := strings.ReplaceAll(content, a.OldString, a.NewString)
		raw := fmt.Sprintf("%s (replace %d occurrence(s))", a.FilePath, count)
		meta := attachToolCallMeta(ctx, map[string]any{
			"category":  "filesystem_write",
			"summary":   raw,
			"file_path": strings.TrimSpace(a.FilePath),
			"diff":      fullReplaceDiff(a.FilePath, content, newContent),
			"raw":       raw + "\nold_string: " + truncateForMeta(a.OldString, 600) + "\nnew_string: " + truncateForMeta(a.NewString, 600),
		})
		ok, err := conf.Confirm(ctx, "edit", buildConfirmDetail(meta))
		if err != nil {
			return llm.TextContent("Error: " + err.Error()), err
		}
		if !ok {
			return llm.TextContent("Denied"), nil
		}
		if err := os.WriteFile(p, []byte(newContent), 0o644); err != nil {
			return llm.TextContent("Error editing file: " + err.Error()), err
		}
		return llm.TextContent(fmt.Sprintf("Replaced %d occurrence(s) in %s", count, a.FilePath)), nil
	})
}

type multiEditItem struct {
	OldString string `json:"old_string"`
	NewString string `json:"new_string"`
}

type multieditArgs struct {
	FilePath string          `json:"file_path"`
	Edits    []multiEditItem `json:"edits"`
}

func multieditTool() tools.Tool {
	return toolWithArgs[multieditArgs]("multiedit", "Apply multiple text replacements to a file (in order)", func(ctx context.Context, a multieditArgs, deps *tools.Container) (llm.Content, error) {
		s, err := tools.Get(deps, ctx, Key)
		if err != nil {
			return llm.TextContent(""), err
		}
		conf := getConfirmer(deps, ctx)
		p, err := s.Resolve(a.FilePath)
		if err != nil {
			return llm.TextContent("Security error: " + err.Error()), err
		}
		if strings.TrimSpace(a.FilePath) == "" {
			return llm.TextContent("Error: missing file_path"), fmt.Errorf("missing file_path")
		}
		if len(a.Edits) == 0 {
			return llm.TextContent("Error: edits is empty"), fmt.Errorf("empty edits")
		}
		b, err := os.ReadFile(p)
		if err != nil {
			if os.IsNotExist(err) {
				return llm.TextContent("File not found: " + a.FilePath), err
			}
			return llm.TextContent("Error editing file: " + err.Error()), err
		}
		orig := string(b)
		content := orig
		counts := make([]int, 0, len(a.Edits))
		for i, e := range a.Edits {
			if e.OldString == "" {
				return llm.TextContent(fmt.Sprintf("Error: edits[%d].old_string is empty", i)), fmt.Errorf("empty old_string")
			}
			if !strings.Contains(content, e.OldString) {
				return llm.TextContent(fmt.Sprintf("String not found for edits[%d] in %s", i, a.FilePath)), fmt.Errorf("string not found")
			}
			c := strings.Count(content, e.OldString)
			counts = append(counts, c)
			content = strings.ReplaceAll(content, e.OldString, e.NewString)
		}

		summary := fmt.Sprintf("%s (multiedit %d step(s))", a.FilePath, len(a.Edits))
		rawLines := []string{summary}
		for i, c := range counts {
			rawLines = append(rawLines, fmt.Sprintf("- step %d: replace %d occurrence(s)", i+1, c))
		}
		meta := attachToolCallMeta(ctx, map[string]any{
			"category":  "filesystem_write",
			"summary":   summary,
			"file_path": strings.TrimSpace(a.FilePath),
			"diff":      fullReplaceDiff(a.FilePath, orig, content),
			"raw":       strings.Join(rawLines, "\n"),
		})
		ok, err := conf.Confirm(ctx, "multiedit", buildConfirmDetail(meta))
		if err != nil {
			return llm.TextContent("Error: " + err.Error()), err
		}
		if !ok {
			return llm.TextContent("Denied"), nil
		}
		if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
			return llm.TextContent("Error editing file: " + err.Error()), err
		}
		return llm.TextContent(fmt.Sprintf("Updated %s with %d edit step(s)", a.FilePath, len(a.Edits))), nil
	})
}

type applyPatchArgs struct {
	Patch string `json:"patch"`
}

func applyPatchTool() tools.Tool {
	return toolWithArgs[applyPatchArgs]("apply_patch", "Apply a patch in apply_patch format (add/update/delete files)", func(ctx context.Context, a applyPatchArgs, deps *tools.Container) (llm.Content, error) {
		s, err := tools.Get(deps, ctx, Key)
		if err != nil {
			return llm.TextContent(""), err
		}
		conf := getConfirmer(deps, ctx)
		paths := []string{}
		if ops, err := parsePatchOpsForPreview(a.Patch); err == nil {
			seen := map[string]struct{}{}
			for _, op := range ops {
				p := strings.TrimSpace(op.path)
				if p != "" {
					if _, ok := seen[p]; !ok {
						seen[p] = struct{}{}
						paths = append(paths, p)
					}
				}
				mt := strings.TrimSpace(op.moveTo)
				if mt != "" {
					if _, ok := seen[mt]; !ok {
						seen[mt] = struct{}{}
						paths = append(paths, mt)
					}
				}
			}
			sort.Strings(paths)
		}
		raw := fmt.Sprintf("apply_patch (%d bytes)", len(a.Patch))
		meta := attachToolCallMeta(ctx, map[string]any{
			"category": "filesystem_write",
			"summary":  raw,
			"paths":    paths,
			"diff":     a.Patch,
			"raw":      raw,
		})
		ok, err := conf.Confirm(ctx, "apply_patch", buildConfirmDetail(meta))
		if err != nil {
			return llm.TextContent("Error: " + err.Error()), err
		}
		if !ok {
			return llm.TextContent("Denied"), nil
		}
		res, err := applyPatchToSandbox(s, a.Patch)
		if err != nil {
			return llm.TextContent("Error: " + err.Error()), err
		}
		return llm.TextContent(res), nil
	})
}

type patchOp struct {
	kind   string // add|update|delete
	path   string
	moveTo string
	// For add: lines are content lines without prefix.
	addLines []string
	// For update: list of hunks, each with raw prefixed lines.
	hunks [][]string
}

func parsePatchOpsForPreview(patch string) ([]patchOp, error) {
	norm := strings.ReplaceAll(patch, "\r\n", "\n")
	norm = strings.ReplaceAll(norm, "\r", "\n")
	lines := strings.Split(norm, "\n")
	if len(lines) > 0 && lines[len(lines)-1] == "" {
		lines = lines[:len(lines)-1]
	}
	if len(lines) < 2 || strings.TrimSpace(lines[0]) != "*** Begin Patch" {
		return nil, fmt.Errorf("patch must start with '*** Begin Patch'")
	}
	if strings.TrimSpace(lines[len(lines)-1]) != "*** End Patch" {
		return nil, fmt.Errorf("patch must end with '*** End Patch'")
	}
	inner := lines[1 : len(lines)-1]
	return parsePatchOps(inner)
}

func applyPatchToSandbox(s *Sandbox, patch string) (string, error) {
	if s == nil {
		return "", fmt.Errorf("nil sandbox")
	}
	norm := strings.ReplaceAll(patch, "\r\n", "\n")
	norm = strings.ReplaceAll(norm, "\r", "\n")
	lines := strings.Split(norm, "\n")
	// tolerate trailing newline
	if len(lines) > 0 && lines[len(lines)-1] == "" {
		lines = lines[:len(lines)-1]
	}
	if len(lines) < 2 || strings.TrimSpace(lines[0]) != "*** Begin Patch" {
		return "", fmt.Errorf("patch must start with '*** Begin Patch'")
	}
	if strings.TrimSpace(lines[len(lines)-1]) != "*** End Patch" {
		return "", fmt.Errorf("patch must end with '*** End Patch'")
	}
	lines = lines[1 : len(lines)-1]
	ops, err := parsePatchOps(lines)
	if err != nil {
		return "", err
	}
	if len(ops) == 0 {
		return "No changes", nil
	}

	changed := 0
	for _, op := range ops {
		switch op.kind {
		case "add":
			if err := applyAddFile(s, op.path, op.addLines); err != nil {
				return "", err
			}
			changed++
		case "delete":
			if err := applyDeleteFile(s, op.path); err != nil {
				return "", err
			}
			changed++
		case "update":
			if err := applyUpdateFile(s, op.path, op.hunks); err != nil {
				return "", err
			}
			if strings.TrimSpace(op.moveTo) != "" {
				if err := applyMoveFile(s, op.path, op.moveTo); err != nil {
					return "", err
				}
			}
			changed++
		default:
			return "", fmt.Errorf("unknown patch op: %s", op.kind)
		}
	}
	return fmt.Sprintf("Applied patch: %d file(s) updated", changed), nil
}

func parsePatchOps(lines []string) ([]patchOp, error) {
	ops := []patchOp{}
	for i := 0; i < len(lines); {
		line := strings.TrimSpace(lines[i])
		switch {
		case strings.HasPrefix(line, "*** Add File: "):
			p := strings.TrimSpace(strings.TrimPrefix(line, "*** Add File: "))
			if p == "" {
				return nil, fmt.Errorf("missing add file path")
			}
			i++
			addLines := []string{}
			for i < len(lines) {
				l := lines[i]
				if strings.HasPrefix(strings.TrimSpace(l), "*** ") {
					break
				}
				if !strings.HasPrefix(l, "+") {
					return nil, fmt.Errorf("add file content must start with '+': %q", l)
				}
				addLines = append(addLines, strings.TrimPrefix(l, "+"))
				i++
			}
			ops = append(ops, patchOp{kind: "add", path: p, addLines: addLines})
		case strings.HasPrefix(line, "*** Delete File: "):
			p := strings.TrimSpace(strings.TrimPrefix(line, "*** Delete File: "))
			if p == "" {
				return nil, fmt.Errorf("missing delete file path")
			}
			i++
			ops = append(ops, patchOp{kind: "delete", path: p})
		case strings.HasPrefix(line, "*** Update File: "):
			p := strings.TrimSpace(strings.TrimPrefix(line, "*** Update File: "))
			if p == "" {
				return nil, fmt.Errorf("missing update file path")
			}
			i++
			moveTo := ""
			if i < len(lines) {
				l2 := strings.TrimSpace(lines[i])
				if strings.HasPrefix(l2, "*** Move to: ") {
					moveTo = strings.TrimSpace(strings.TrimPrefix(l2, "*** Move to: "))
					i++
				}
			}
			hunks := [][]string{}
			for i < len(lines) {
				l := lines[i]
				lt := strings.TrimSpace(l)
				if strings.HasPrefix(lt, "*** ") {
					break
				}
				if strings.HasPrefix(lt, "@@") {
					i++
					h := []string{}
					for i < len(lines) {
						ll := lines[i]
						llt := strings.TrimSpace(ll)
						if strings.HasPrefix(llt, "@@") || strings.HasPrefix(llt, "*** ") {
							break
						}
						if ll == "*** End of File" {
							i++
							break
						}
						if ll == "" {
							// empty lines are valid, but must have a prefix; treat as invalid to avoid ambiguous patches
							return nil, fmt.Errorf("patch hunk line missing prefix")
						}
						pref := ll[0]
						if pref != ' ' && pref != '-' && pref != '+' {
							return nil, fmt.Errorf("invalid hunk line prefix: %q", ll)
						}
						h = append(h, ll)
						i++
					}
					if len(h) > 0 {
						hunks = append(hunks, h)
					}
					continue
				}
				// skip non-hunk noise (e.g., context headers)
				i++
			}
			ops = append(ops, patchOp{kind: "update", path: p, moveTo: moveTo, hunks: hunks})
		default:
			// ignore stray lines
			i++
		}
	}
	return ops, nil
}

func applyAddFile(s *Sandbox, relPath string, lines []string) error {
	p, err := s.Resolve(relPath)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		return err
	}
	content := strings.Join(lines, "\n")
	if len(lines) > 0 {
		content += "\n"
	}
	if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
		return err
	}
	return nil
}

func applyDeleteFile(s *Sandbox, relPath string) error {
	p, err := s.Resolve(relPath)
	if err != nil {
		return err
	}
	if err := os.Remove(p); err != nil {
		return err
	}
	return nil
}

func applyMoveFile(s *Sandbox, fromRel, toRel string) error {
	fromAbs, err := s.Resolve(fromRel)
	if err != nil {
		return err
	}
	toAbs, err := s.Resolve(toRel)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(toAbs), 0o755); err != nil {
		return err
	}
	return os.Rename(fromAbs, toAbs)
}

func applyUpdateFile(s *Sandbox, relPath string, hunks [][]string) error {
	p, err := s.Resolve(relPath)
	if err != nil {
		return err
	}
	b, err := os.ReadFile(p)
	if err != nil {
		return err
	}
	content := strings.ReplaceAll(string(b), "\r\n", "\n")
	content = strings.ReplaceAll(content, "\r", "\n")
	for _, h := range hunks {
		oldLines := []string{}
		newLines := []string{}
		for _, l := range h {
			if l == "" {
				continue
			}
			pref := l[0]
			body := l[1:]
			switch pref {
			case ' ':
				oldLines = append(oldLines, body)
				newLines = append(newLines, body)
			case '-':
				oldLines = append(oldLines, body)
			case '+':
				newLines = append(newLines, body)
			}
		}
		oldText := strings.Join(oldLines, "\n")
		newText := strings.Join(newLines, "\n")
		// prefer newline-terminated blocks
		candidates := []string{oldText + "\n", oldText}
		var found string
		for _, cand := range candidates {
			if cand == "" {
				continue
			}
			if strings.Contains(content, cand) {
				found = cand
				break
			}
		}
		if found == "" {
			return fmt.Errorf("hunk failed to apply to %s (context not found)", relPath)
		}
		repl := newText
		if strings.HasSuffix(found, "\n") {
			repl += "\n"
		}
		content = strings.Replace(content, found, repl, 1)
	}
	return os.WriteFile(p, []byte(content), 0o644)
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
		pat := strings.TrimSpace(a.Pattern)
		if pat == "" {
			return "Error: empty pattern", nil
		}
		// Support ** patterns (doublestar) and normal * patterns.
		pat = filepath.ToSlash(pat)
		matches, err := doublestar.Glob(os.DirFS(base), pat)
		if err != nil {
			return "Error: " + err.Error(), nil
		}
		files := []string{}
		for _, rel := range matches {
			abs := filepath.Join(base, filepath.FromSlash(rel))
			st, err := os.Stat(abs)
			if err != nil || st.IsDir() {
				continue
			}
			rel2, _ := filepath.Rel(s.RootDir, abs)
			files = append(files, filepath.ToSlash(rel2))
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

type externalDirectoryArgs struct {
	Path string `json:"path"`
}

func externalDirectoryTool() tools.Tool {
	return tools.Func[externalDirectoryArgs]("external_directory", "Allow access to a directory outside the sandbox root", func(ctx context.Context, a externalDirectoryArgs, deps *tools.Container) (any, error) {
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
			return fmt.Sprintf("External directory already allowed: %s", normalized), nil
		}

		conf := getConfirmer(deps, ctx)
		meta := attachToolCallMeta(ctx, map[string]any{
			"category":  "external_directory",
			"summary":   fmt.Sprintf("Allow external directory: %s", normalized),
			"file_path": normalized,
			"path":      normalized,
			"raw":       normalized,
		})
		ok, err := conf.Confirm(ctx, "external_directory", buildConfirmDetail(meta))
		if err != nil {
			return "", err
		}
		if !ok {
			return "Denied", nil
		}

		finalPath, added, err := s.AllowExternalDirectory(normalized)
		if err != nil {
			return "", err
		}
		if !added {
			return fmt.Sprintf("External directory already allowed: %s", finalPath), nil
		}
		return fmt.Sprintf("Allowed external directory: %s", finalPath), nil
	})
}

type grepArgs struct {
	Pattern string `json:"pattern"`
	Path    string `json:"path,omitempty"`

	Glob        string `json:"glob,omitempty"` // e.g. "*.go" or "**/*.ts"
	IgnoreCase  bool   `json:"ignore_case,omitempty"`
	Before      int    `json:"before,omitempty"`
	After       int    `json:"after,omitempty"`
	Context     int    `json:"context,omitempty"`
	MaxResults  int    `json:"max_results,omitempty"`  // output lines or entries, default 50
	OutputMode  string `json:"output_mode,omitempty"`  // "content"|"files_with_matches"|"count"
	LineNumbers bool   `json:"line_numbers,omitempty"` // default true
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
		pat := a.Pattern
		if a.IgnoreCase {
			pat = "(?i)" + pat
		}
		re, err := regexp.Compile(pat)
		if err != nil {
			return "Invalid regex: " + err.Error(), nil
		}
		before := a.Before
		after := a.After
		if a.Context > 0 {
			before = a.Context
			after = a.Context
		}
		if before < 0 {
			before = 0
		}
		if after < 0 {
			after = 0
		}
		maxOut := a.MaxResults
		if maxOut <= 0 {
			maxOut = 50
		}
		mode := strings.TrimSpace(a.OutputMode)
		if mode == "" {
			mode = "content"
		}
		showLineNumbers := true
		if a.LineNumbers == false {
			// default is true, so only false matters
			showLineNumbers = false
		}

		globFilter := strings.TrimSpace(a.Glob)
		if globFilter != "" {
			globFilter = filepath.ToSlash(globFilter)
		}

		results := []string{}
		files := []string{}
		counts := map[string]int{}
		stopped := false
		_ = filepath.WalkDir(base, func(path string, d os.DirEntry, err error) error {
			if err != nil || d.IsDir() {
				return nil
			}
			if stopped {
				return errors.New("_stop")
			}
			// filter file path
			relFromRoot, _ := filepath.Rel(s.RootDir, path)
			relFromRoot = filepath.ToSlash(relFromRoot)
			if globFilter != "" {
				ok, _ := doublestar.Match(globFilter, relFromRoot)
				if !ok {
					return nil
				}
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
			scannerBuf := make([]byte, 0, 64*1024)
			scanner.Buffer(scannerBuf, 1024*1024)
			type prevLine struct {
				no   int
				text string
			}
			prev := make([]prevLine, 0, before)
			pushPrev := func(no int, text string) {
				if before <= 0 {
					return
				}
				if len(prev) == before {
					copy(prev, prev[1:])
					prev[before-1] = prevLine{no: no, text: text}
					return
				}
				prev = append(prev, prevLine{no: no, text: text})
			}

			lineNo := 0
			afterRemain := 0
			lastEmitted := 0
			fileMatched := false
			matchCount := 0
			for scanner.Scan() {
				lineNo++
				line := scanner.Text()
				isMatch := re.MatchString(line)
				if isMatch {
					fileMatched = true
					matchCount++
					if mode == "files_with_matches" {
						files = append(files, relFromRoot)
						break
					}
					// emit before context
					for _, pl := range prev {
						if pl.no <= lastEmitted {
							continue
						}
						results = append(results, formatGrepLine(relFromRoot, pl.no, pl.text, showLineNumbers))
						lastEmitted = pl.no
						if len(results) >= maxOut {
							stopped = true
							break
						}
					}
					if stopped {
						break
					}
					// emit match line
					if lineNo > lastEmitted {
						results = append(results, formatGrepLine(relFromRoot, lineNo, line, showLineNumbers))
						lastEmitted = lineNo
						if len(results) >= maxOut {
							stopped = true
							break
						}
					}
					if afterRemain < after {
						afterRemain = after
					}
				} else if afterRemain > 0 {
					if mode == "content" {
						if lineNo > lastEmitted {
							results = append(results, formatGrepLine(relFromRoot, lineNo, line, showLineNumbers))
							lastEmitted = lineNo
							if len(results) >= maxOut {
								stopped = true
								break
							}
						}
					}
					afterRemain--
				}
				pushPrev(lineNo, line)
			}
			if mode == "count" {
				if matchCount > 0 {
					counts[relFromRoot] += matchCount
				}
				return nil
			}
			if stopped {
				return errors.New("_stop")
			}
			if fileMatched && mode == "files_with_matches" {
				// already added
				return nil
			}
			return nil
		})
		switch mode {
		case "files_with_matches":
			if len(files) == 0 {
				return "No matches for: " + a.Pattern, nil
			}
			// unique + cap
			uniq := []string{}
			seen := map[string]struct{}{}
			for _, f := range files {
				if _, ok := seen[f]; ok {
					continue
				}
				seen[f] = struct{}{}
				uniq = append(uniq, f)
				if len(uniq) >= maxOut {
					break
				}
			}
			return strings.Join(uniq, "\n"), nil
		case "count":
			if len(counts) == 0 {
				return "No matches for: " + a.Pattern, nil
			}
			lines := []string{}
			keys := make([]string, 0, len(counts))
			for k := range counts {
				keys = append(keys, k)
			}
			sort.Strings(keys)
			for _, k := range keys {
				lines = append(lines, fmt.Sprintf("%s: %d", k, counts[k]))
				if len(lines) >= maxOut {
					break
				}
			}
			return strings.Join(lines, "\n"), nil
		default:
			if len(results) == 0 {
				return "No matches for: " + a.Pattern, nil
			}
			return strings.Join(results, "\n"), nil
		}
	})
}

func formatGrepLine(file string, lineNo int, line string, showLineNumbers bool) string {
	line = truncate(line, 200)
	if !showLineNumbers {
		return fmt.Sprintf("%s: %s", file, line)
	}
	return fmt.Sprintf("%s:%d: %s", file, lineNo, line)
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
			return "Error: " + err.Error(), err
		}
		if !ok {
			return "Denied", nil
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
			return "Error: " + err.Error(), err
		}
		if !ok {
			return "Denied", nil
		}
		s.ReplaceTodos(a.Todos)
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
