package sandbox

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools/execenv"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools/execrunner"
)

type allowConfirmer struct{}

func (allowConfirmer) Confirm(context.Context, string, string) (bool, error) {
	return true, nil
}

type denyConfirmer struct{}

func (denyConfirmer) Confirm(context.Context, string, string) (bool, error) {
	return false, nil
}

type captureConfirmer struct {
	action string
	detail string
	allow  bool
}

func (c *captureConfirmer) Confirm(_ context.Context, action string, detail string) (bool, error) {
	c.action = action
	c.detail = detail
	return c.allow, nil
}

type errorConfirmer struct {
	err error
}

func (c errorConfirmer) Confirm(context.Context, string, string) (bool, error) {
	return false, c.err
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) {
	return f(r)
}

func useSandboxWebfetchResolver(t *testing.T, resolver func(context.Context, string) ([]net.IPAddr, error)) {
	t.Helper()
	orig := webfetchLookupIPAddrs
	webfetchLookupIPAddrs = resolver
	t.Cleanup(func() {
		webfetchLookupIPAddrs = orig
	})
}

func useSandboxPublicWebfetchResolver(t *testing.T) {
	t.Helper()
	useSandboxWebfetchResolver(t, func(_ context.Context, host string) ([]net.IPAddr, error) {
		if ip := net.ParseIP(host); ip != nil {
			return []net.IPAddr{{IP: ip}}, nil
		}
		switch host {
		case "example.test", "public.test":
			return []net.IPAddr{{IP: net.ParseIP("93.184.216.34")}}, nil
		case "internal.test":
			return []net.IPAddr{{IP: net.ParseIP("10.20.30.40")}}, nil
		default:
			return nil, fmt.Errorf("lookup %s: no such host", host)
		}
	})
}

func useSandboxPathRevalidateHook(t *testing.T, hook func(string)) {
	t.Helper()
	orig := beforeSandboxPathRevalidate
	beforeSandboxPathRevalidate = hook
	t.Cleanup(func() {
		beforeSandboxPathRevalidate = orig
	})
}

func useSandboxReadAllHook(t *testing.T, hook func(io.Reader) ([]byte, error)) {
	t.Helper()
	orig := sandboxReadAll
	sandboxReadAll = hook
	t.Cleanup(func() {
		sandboxReadAll = orig
	})
}

type failAfterReader struct {
	data []byte
}

func (r *failAfterReader) Read(p []byte) (int, error) {
	if len(r.data) == 0 {
		return 0, errors.New("simulated read failure")
	}
	n := copy(p, r.data)
	r.data = r.data[n:]
	return n, nil
}

func (r *failAfterReader) Close() error { return nil }

func installGrepHooksForTest(t *testing.T) {
	t.Helper()
	origWalk := grepWalkDirFn
	origOpen := grepOpenFile
	origReadSample := grepReadSample
	origSeek := grepSeekStart
	t.Cleanup(func() {
		grepWalkDirFn = origWalk
		grepOpenFile = origOpen
		grepReadSample = origReadSample
		grepSeekStart = origSeek
	})
}

func installGlobHooksForTest(t *testing.T) {
	t.Helper()
	origStat := globStatFile
	t.Cleanup(func() {
		globStatFile = origStat
	})
}

func TestSandboxResolveBlocksEscape(t *testing.T) {
	root := t.TempDir()
	s, err := New(root)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	_, err = s.Resolve("../etc/passwd")
	if err == nil {
		t.Fatalf("expected escape error")
	}
}

func TestSandboxResolveAllowsInside(t *testing.T) {
	root := t.TempDir()
	s, err := New(root)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	p, err := s.Resolve("a/b/../c.txt")
	if err != nil {
		t.Fatalf("resolve: %v", err)
	}
	if p == "" {
		t.Fatalf("expected non-empty path")
	}
}

func TestSandboxResolveAllowsExternalAfterAllow(t *testing.T) {
	root := t.TempDir()
	ext := t.TempDir()
	s, err := New(root)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	target := filepath.Join(ext, "file.txt")
	if _, err := s.Resolve(target); err == nil {
		t.Fatalf("expected escape error before allow")
	}
	if _, added, err := s.AllowExternalDirectory(ext); err != nil {
		t.Fatalf("allow external: %v", err)
	} else if !added {
		t.Fatalf("expected external directory to be added")
	}
	if _, err := s.Resolve(target); err != nil {
		t.Fatalf("expected resolve to succeed after allow, got %v", err)
	}
}

func TestSandboxAllowExternalDirectoryResolvesSymlinks(t *testing.T) {
	root := t.TempDir()
	ext := t.TempDir()
	link := filepath.Join(t.TempDir(), "ext-link")
	if err := os.Symlink(ext, link); err != nil {
		t.Skipf("symlink not supported: %v", err)
	}
	s, err := New(root)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	got, _, err := s.AllowExternalDirectory(link)
	if err != nil {
		t.Fatalf("allow external: %v", err)
	}
	if filepath.Clean(got) != filepath.Clean(ext) {
		t.Fatalf("expected resolved path %q, got %q", ext, got)
	}
}

func TestSandboxAllowExternalFile(t *testing.T) {
	root := t.TempDir()
	ext := t.TempDir()
	filePath := filepath.Join(ext, "config.env")
	if err := os.WriteFile(filePath, []byte("token=abc"), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}
	s, err := New(root)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	if _, added, err := s.AllowExternalDirectory(filePath); err != nil {
		t.Fatalf("allow external file: %v", err)
	} else if !added {
		t.Fatalf("expected external file to be added")
	}
	if _, err := s.Resolve(filePath); err != nil {
		t.Fatalf("expected resolve to succeed after allow, got %v", err)
	}
	otherPath := filepath.Join(ext, "other.txt")
	if err := os.WriteFile(otherPath, []byte("other"), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}
	if _, err := s.Resolve(otherPath); err == nil {
		t.Fatalf("expected other file to be blocked")
	}
}

func TestSandboxResolveBlocksSymlinkEscape(t *testing.T) {
	root := t.TempDir()
	ext := t.TempDir()
	link := filepath.Join(root, "ext-link")
	if err := os.Symlink(ext, link); err != nil {
		t.Skipf("symlink not supported: %v", err)
	}
	s, err := New(root)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	if _, err := s.Resolve(filepath.Join("ext-link", "file.txt")); err == nil {
		t.Fatalf("expected symlink escape error")
	}
}

func TestSandboxResolveAllowsSymlinkInsideRoot(t *testing.T) {
	root := t.TempDir()
	target := filepath.Join(root, "target")
	if err := os.MkdirAll(target, 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	link := filepath.Join(root, "link")
	if err := os.Symlink(target, link); err != nil {
		t.Skipf("symlink not supported: %v", err)
	}
	s, err := New(root)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	if _, err := s.Resolve(filepath.Join("link", "file.txt")); err != nil {
		t.Fatalf("expected resolve to succeed, got %v", err)
	}
}

func TestMissingConfirmerDeniesTool(t *testing.T) {
	root := t.TempDir()
	s, err := New(root)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	deps := tools.NewContainer()
	tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })

	tool := bashTool()
	_, err = tool.Execute(context.Background(), `{"command":"echo hi"}`, deps)
	if err == nil {
		t.Fatalf("expected confirmer error")
	}
	if !errors.Is(err, ErrMissingConfirmer) {
		t.Fatalf("expected ErrMissingConfirmer, got %v", err)
	}
}

func TestBashTool_NilConfirmerFailsClosedWithoutPanic(t *testing.T) {
	root := t.TempDir()
	s, err := New(root)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	deps := tools.NewContainer()
	tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })
	tools.Provide(deps, ConfirmKey, func(context.Context) (Confirmer, error) { return nil, nil })

	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("expected fail-closed error instead of panic, got panic: %v", r)
		}
	}()

	out, err := bashTool().Execute(context.Background(), `{"command":"echo hi"}`, deps)
	if err == nil {
		t.Fatalf("expected confirmer error")
	}
	if !errors.Is(err, ErrMissingConfirmer) {
		t.Fatalf("expected ErrMissingConfirmer, got %v", err)
	}
	if !strings.Contains(strings.ToLower(err.Error()), "nil confirmer") {
		t.Fatalf("expected nil confirmer context, got %v", err)
	}
	assertSandboxSeverityActionDiagnostic(t, out.PlainText())
}

func TestWriteTool_NilConfirmerFailsClosedWithoutPanic(t *testing.T) {
	root := t.TempDir()
	s, err := New(root)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	deps := tools.NewContainer()
	tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })
	tools.Provide(deps, ConfirmKey, func(context.Context) (Confirmer, error) { return nil, nil })

	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("expected fail-closed error instead of panic, got panic: %v", r)
		}
	}()

	out, err := writeTool().Execute(context.Background(), `{"file_path":"file.txt","content":"hello\n"}`, deps)
	if err == nil {
		t.Fatalf("expected confirmer error")
	}
	if !errors.Is(err, ErrMissingConfirmer) {
		t.Fatalf("expected ErrMissingConfirmer, got %v", err)
	}
	if !strings.Contains(strings.ToLower(err.Error()), "nil confirmer") {
		t.Fatalf("expected nil confirmer context, got %v", err)
	}
	assertSandboxSeverityActionDiagnostic(t, out.PlainText())
}

func TestEditTool_NilConfirmerFailsClosedWithoutPanic(t *testing.T) {
	root := t.TempDir()
	s, err := New(root)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	if err := os.WriteFile(filepath.Join(root, "file.txt"), []byte("hello\n"), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}
	deps := tools.NewContainer()
	tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })
	tools.Provide(deps, ConfirmKey, func(context.Context) (Confirmer, error) { return nil, nil })

	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("expected fail-closed error instead of panic, got panic: %v", r)
		}
	}()

	out, err := editTool().Execute(context.Background(), `{"file_path":"file.txt","old_string":"hello","new_string":"hi"}`, deps)
	if err == nil {
		t.Fatalf("expected confirmer error")
	}
	if !errors.Is(err, ErrMissingConfirmer) {
		t.Fatalf("expected ErrMissingConfirmer, got %v", err)
	}
	if !strings.Contains(strings.ToLower(err.Error()), "nil confirmer") {
		t.Fatalf("expected nil confirmer context, got %v", err)
	}
	assertSandboxSeverityActionDiagnostic(t, out.PlainText())
}

func TestSandboxToolErrorDiagnosticSeverityActionFormat(t *testing.T) {
	t.Parallel()

	root := t.TempDir()
	s, err := New(root)
	if err != nil {
		t.Fatalf("new: %v", err)
	}

	t.Run("tool args parse", func(t *testing.T) {
		deps := tools.NewContainer()
		tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })

		out, err := bashTool().Handler(context.Background(), json.RawMessage(`{"command":`), deps)
		if err == nil {
			t.Fatalf("expected parse error")
		}
		assertSandboxSeverityActionDiagnostic(t, out.PlainText())
	})

	t.Run("bash confirmer error", func(t *testing.T) {
		deps := tools.NewContainer()
		tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })

		out, err := bashTool().Execute(context.Background(), `{"command":"echo hi"}`, deps)
		if err == nil {
			t.Fatalf("expected confirmer error")
		}
		assertSandboxSeverityActionDiagnostic(t, out.PlainText())
	})

	t.Run("read not found", func(t *testing.T) {
		deps := tools.NewContainer()
		tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })

		out, err := readTool().Execute(context.Background(), `{"file_path":"missing.txt"}`, deps)
		if err == nil {
			t.Fatalf("expected missing-file error")
		}
		assertSandboxSeverityActionDiagnostic(t, out.PlainText())
	})
}

func TestTodoReadTool_ErrorDiagnosticSeverityActionFormat(t *testing.T) {
	t.Parallel()

	root := t.TempDir()
	s, err := New(root)
	if err != nil {
		t.Fatalf("new: %v", err)
	}

	t.Run("missing confirmer", func(t *testing.T) {
		deps := tools.NewContainer()
		tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })

		out, err := todoReadToolNamed("todo_read").Execute(context.Background(), `{}`, deps)
		if err == nil {
			t.Fatalf("expected confirmer error")
		}
		if !errors.Is(err, ErrMissingConfirmer) {
			t.Fatalf("expected ErrMissingConfirmer, got %v", err)
		}
		assertSandboxSeverityActionDiagnostic(t, out.PlainText())
	})

	t.Run("confirmer error", func(t *testing.T) {
		expectedErr := errors.New("policy backend unavailable")
		deps := tools.NewContainer()
		tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })
		tools.Provide(deps, ConfirmKey, func(context.Context) (Confirmer, error) { return errorConfirmer{err: expectedErr}, nil })

		out, err := todoReadToolNamed("todo_read").Execute(context.Background(), `{}`, deps)
		if err == nil {
			t.Fatalf("expected confirmer error")
		}
		if !errors.Is(err, expectedErr) {
			t.Fatalf("expected confirmer error %v, got %v", expectedErr, err)
		}
		assertSandboxSeverityActionDiagnostic(t, out.PlainText())
	})
}

func TestTodoWriteTool_ErrorDiagnosticSeverityActionFormat(t *testing.T) {
	t.Parallel()

	root := t.TempDir()
	s, err := New(root)
	if err != nil {
		t.Fatalf("new: %v", err)
	}

	t.Run("missing confirmer", func(t *testing.T) {
		deps := tools.NewContainer()
		tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })

		out, err := todoWriteToolNamed("todo_write").Execute(context.Background(), `{"todos":[]}`, deps)
		if err == nil {
			t.Fatalf("expected confirmer error")
		}
		if !errors.Is(err, ErrMissingConfirmer) {
			t.Fatalf("expected ErrMissingConfirmer, got %v", err)
		}
		assertSandboxSeverityActionDiagnostic(t, out.PlainText())
	})

	t.Run("confirmer error", func(t *testing.T) {
		expectedErr := errors.New("policy backend unavailable")
		deps := tools.NewContainer()
		tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })
		tools.Provide(deps, ConfirmKey, func(context.Context) (Confirmer, error) { return errorConfirmer{err: expectedErr}, nil })

		out, err := todoWriteToolNamed("todo_write").Execute(context.Background(), `{"todos":[]}`, deps)
		if err == nil {
			t.Fatalf("expected confirmer error")
		}
		if !errors.Is(err, expectedErr) {
			t.Fatalf("expected confirmer error %v, got %v", expectedErr, err)
		}
		assertSandboxSeverityActionDiagnostic(t, out.PlainText())
	})
}

func TestBashTool_DeniedReturnsErrorAndMetadata(t *testing.T) {
	root := t.TempDir()
	s, err := New(root)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	deps := tools.NewContainer()
	tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })
	tools.Provide(deps, ConfirmKey, func(context.Context) (Confirmer, error) { return denyConfirmer{}, nil })

	ctx := tools.WithToolResultMetadata(context.Background())
	out, err := bashTool().Execute(ctx, `{"command":"echo denied"}`, deps)
	if err == nil {
		t.Fatal("expected denied error")
	}
	if !errors.Is(err, ErrToolDenied) {
		t.Fatalf("expected ErrToolDenied, got %v", err)
	}
	assertSandboxSeverityActionDiagnostic(t, out.PlainText())
	assertDeniedMetadata(t, ctx, "bash")
}

func TestWriteTool_DeniedReturnsErrorAndMetadata(t *testing.T) {
	root := t.TempDir()
	s, err := New(root)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	deps := tools.NewContainer()
	tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })
	tools.Provide(deps, ConfirmKey, func(context.Context) (Confirmer, error) { return denyConfirmer{}, nil })

	ctx := tools.WithToolResultMetadata(context.Background())
	out, err := writeTool().Execute(ctx, `{"file_path":"file.txt","content":"hello\n"}`, deps)
	if err == nil {
		t.Fatal("expected denied error")
	}
	if !errors.Is(err, ErrToolDenied) {
		t.Fatalf("expected ErrToolDenied, got %v", err)
	}
	assertSandboxSeverityActionDiagnostic(t, out.PlainText())
	assertDeniedMetadata(t, ctx, "write")
}

func TestEditTool_DeniedReturnsErrorAndMetadata(t *testing.T) {
	root := t.TempDir()
	s, err := New(root)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	if err := os.WriteFile(filepath.Join(root, "file.txt"), []byte("hello\n"), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}
	deps := tools.NewContainer()
	tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })
	tools.Provide(deps, ConfirmKey, func(context.Context) (Confirmer, error) { return denyConfirmer{}, nil })

	ctx := tools.WithToolResultMetadata(context.Background())
	out, err := editTool().Execute(ctx, `{"file_path":"file.txt","old_string":"hello","new_string":"hi"}`, deps)
	if err == nil {
		t.Fatal("expected denied error")
	}
	if !errors.Is(err, ErrToolDenied) {
		t.Fatalf("expected ErrToolDenied, got %v", err)
	}
	assertSandboxSeverityActionDiagnostic(t, out.PlainText())
	assertDeniedMetadata(t, ctx, "edit")
}

func TestApplyPatchTool_DeniedReturnsErrorAndMetadata(t *testing.T) {
	root := t.TempDir()
	s, err := New(root)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	deps := tools.NewContainer()
	tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })
	tools.Provide(deps, ConfirmKey, func(context.Context) (Confirmer, error) { return denyConfirmer{}, nil })

	ctx := tools.WithToolResultMetadata(context.Background())
	patch := "*** Begin Patch\n*** Add File: a.txt\n+hello\n*** End Patch\n"
	out, err := applyPatchTool().Execute(ctx, fmt.Sprintf(`{"patch":%q}`, patch), deps)
	if err == nil {
		t.Fatal("expected denied error")
	}
	if !errors.Is(err, ErrToolDenied) {
		t.Fatalf("expected ErrToolDenied, got %v", err)
	}
	assertSandboxSeverityActionDiagnostic(t, out.PlainText())
	assertDeniedMetadata(t, ctx, "apply_patch")
}

func TestTodoWriteTool_DeniedReturnsErrorAndMetadata(t *testing.T) {
	root := t.TempDir()
	s, err := New(root)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	deps := tools.NewContainer()
	tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })
	tools.Provide(deps, ConfirmKey, func(context.Context) (Confirmer, error) { return denyConfirmer{}, nil })

	ctx := tools.WithToolResultMetadata(context.Background())
	out, err := todoWriteToolNamed("todo_write").Execute(ctx, `{"todos":[]}`, deps)
	if err == nil {
		t.Fatal("expected denied error")
	}
	if !errors.Is(err, ErrToolDenied) {
		t.Fatalf("expected ErrToolDenied, got %v", err)
	}
	assertSandboxSeverityActionDiagnostic(t, out.PlainText())
	assertDeniedMetadata(t, ctx, "todo_write")
}

func TestExternalDirectoryTool_DeniedReturnsErrorAndMetadata(t *testing.T) {
	root := t.TempDir()
	s, err := New(root)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	outside := t.TempDir()
	deps := tools.NewContainer()
	tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })
	tools.Provide(deps, ConfirmKey, func(context.Context) (Confirmer, error) { return denyConfirmer{}, nil })

	ctx := tools.WithToolResultMetadata(context.Background())
	out, err := externalDirectoryTool().Execute(ctx, fmt.Sprintf(`{"path":%q}`, outside), deps)
	if err == nil {
		t.Fatal("expected denied error")
	}
	if !errors.Is(err, ErrToolDenied) {
		t.Fatalf("expected ErrToolDenied, got %v", err)
	}
	assertSandboxSeverityActionDiagnostic(t, out.PlainText())
	assertDeniedMetadata(t, ctx, "external_directory")
}

func assertSandboxSeverityActionDiagnostic(t *testing.T, text string) {
	t.Helper()
	trimmed := strings.TrimSpace(text)
	if !strings.HasPrefix(trimmed, "[ERROR] ") {
		t.Fatalf("expected [ERROR] prefix, got %q", text)
	}
	if !strings.Contains(trimmed, " - ") {
		t.Fatalf("expected summary-action delimiter, got %q", text)
	}
	if strings.HasPrefix(trimmed, "Error: ") {
		t.Fatalf("expected structured diagnostic instead of bare error, got %q", text)
	}
}

func assertSandboxSeverityActionWarningDiagnostic(t *testing.T, text string) {
	t.Helper()
	trimmed := strings.TrimSpace(text)
	if !strings.HasPrefix(trimmed, "[WARN] ") {
		t.Fatalf("expected [WARN] prefix, got %q", text)
	}
	if !strings.Contains(trimmed, " - ") {
		t.Fatalf("expected summary-action delimiter, got %q", text)
	}
}

func assertDeniedMetadata(t *testing.T, ctx context.Context, toolName string) {
	t.Helper()
	meta := tools.ToolResultMetadataSnapshot(ctx)
	if meta == nil {
		t.Fatal("expected metadata")
	}
	if kind, _ := meta["error_kind"].(string); kind != "denied" {
		t.Fatalf("expected error_kind denied, got %#v", meta["error_kind"])
	}
	if deniedTool, _ := meta["denied_tool"].(string); deniedTool != toolName {
		t.Fatalf("expected denied_tool=%q, got %#v", toolName, meta["denied_tool"])
	}
	if reason, _ := meta["denied_reason"].(string); strings.TrimSpace(reason) == "" {
		t.Fatalf("expected denied_reason metadata, got %#v", meta["denied_reason"])
	}
}

func TestBashTool_DoesNotInheritSensitiveEnv(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("bash env inheritance test uses POSIX shell syntax")
	}

	root := t.TempDir()
	s, err := New(root)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	deps := tools.NewContainer()
	tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })
	tools.Provide(deps, ConfirmKey, func(context.Context) (Confirmer, error) { return allowConfirmer{}, nil })

	const secretName = "NEW006_SENSITIVE_TOKEN"
	const secretValue = "super-secret"
	t.Setenv(secretName, secretValue)

	command := fmt.Sprintf(`if [ -z "${%s+x}" ]; then echo hidden; else echo "$%s"; fi`, secretName, secretName)
	out, err := bashTool().Execute(context.Background(), fmt.Sprintf(`{"command":%q}`, command), deps)
	if err != nil {
		t.Fatalf("execute bash: %v", err)
	}
	got := strings.TrimSpace(out.PlainText())
	if got != "hidden" {
		t.Fatalf("expected hidden, got %q", got)
	}
}

func TestBashTool_AllowsExplicitAllowlistedEnv(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("bash env inheritance test uses POSIX shell syntax")
	}

	root := t.TempDir()
	s, err := New(root)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	deps := tools.NewContainer()
	tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })
	tools.Provide(deps, ConfirmKey, func(context.Context) (Confirmer, error) { return allowConfirmer{}, nil })
	tools.Provide(deps, execenv.ExtraAllowlistKey, func(context.Context) ([]string, error) {
		return []string{"NEW006_VISIBLE_TOKEN"}, nil
	})

	const visibleName = "NEW006_VISIBLE_TOKEN"
	const visibleValue = "allowed"
	t.Setenv(visibleName, visibleValue)

	command := fmt.Sprintf(`if [ -z "${%s+x}" ]; then echo missing; else echo "$%s"; fi`, visibleName, visibleName)
	out, err := bashTool().Execute(context.Background(), fmt.Sprintf(`{"command":%q}`, command), deps)
	if err != nil {
		t.Fatalf("execute bash: %v", err)
	}
	got := strings.TrimSpace(out.PlainText())
	if got != visibleValue {
		t.Fatalf("expected %q, got %q", visibleValue, got)
	}
}

func TestBashTool_StreamLimitsOutput(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("stream limit test uses POSIX shell commands")
	}

	root := t.TempDir()
	s, err := New(root)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	deps := tools.NewContainer()
	tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })
	tools.Provide(deps, ConfirmKey, func(context.Context) (Confirmer, error) { return allowConfirmer{}, nil })

	ctx := tools.WithToolResultMetadata(context.Background())
	cmd := `yes stream-limit-line | head -n 25000`
	out, err := bashTool().Execute(ctx, fmt.Sprintf(`{"command":%q}`, cmd), deps)
	if err != nil {
		t.Fatalf("execute bash: %v", err)
	}
	text := out.PlainText()
	if !strings.Contains(text, "output truncated after") {
		t.Fatalf("expected truncation notice, got %q", text)
	}

	meta := tools.ToolResultMetadataSnapshot(ctx)
	if capped, ok := meta["output_capped"].(bool); !ok || !capped {
		t.Fatalf("expected output_capped=true, got %#v", meta["output_capped"])
	}
	if limit, ok := meta["output_bytes_limit"].(int); !ok || limit != execrunner.DefaultMaxOutputBytes {
		t.Fatalf("expected output_bytes_limit=%d, got %#v", execrunner.DefaultMaxOutputBytes, meta["output_bytes_limit"])
	}
	path, _ := meta["output_path"].(string)
	if strings.TrimSpace(path) == "" {
		t.Fatalf("expected output_path in metadata, got %#v", meta["output_path"])
	}
	info, statErr := os.Stat(path)
	if statErr != nil {
		t.Fatalf("stat output_path: %v", statErr)
	}
	if info.Size() <= int64(execrunner.DefaultMaxOutputBytes) {
		t.Fatalf("expected artifact > %d bytes, got %d", execrunner.DefaultMaxOutputBytes, info.Size())
	}
}

func TestBashTool_KillsProcessGroupOnTimeout(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("process-group timeout test uses POSIX shell commands")
	}

	root := t.TempDir()
	s, err := New(root)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	deps := tools.NewContainer()
	tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })
	tools.Provide(deps, ConfirmKey, func(context.Context) (Confirmer, error) { return allowConfirmer{}, nil })

	ctx := tools.WithToolResultMetadata(context.Background())
	command := `sleep 30 & child=$!; echo $child > child.pid; wait $child`
	out, err := bashTool().Execute(ctx, fmt.Sprintf(`{"command":%q,"timeout":1}`, command), deps)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("expected deadline exceeded, got %v", err)
	}
	if !strings.Contains(strings.ToLower(out.PlainText()), "timed out") {
		t.Fatalf("expected timed out output, got %q", out.PlainText())
	}

	pidPath := filepath.Join(root, "child.pid")
	var pid int
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		data, readErr := os.ReadFile(pidPath)
		if readErr == nil {
			pid, err = strconv.Atoi(strings.TrimSpace(string(data)))
			if err != nil {
				t.Fatalf("parse child pid: %v", err)
			}
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	if pid <= 0 {
		t.Fatalf("expected child pid to be written at %s", pidPath)
	}
	t.Cleanup(func() {
		_ = syscall.Kill(pid, syscall.SIGKILL)
	})

	aliveDeadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(aliveDeadline) {
		if !processExists(pid) {
			return
		}
		time.Sleep(25 * time.Millisecond)
	}
	t.Fatalf("expected child process %d to be terminated with parent timeout", pid)
}

func processExists(pid int) bool {
	if pid <= 0 {
		return false
	}
	err := syscall.Kill(pid, 0)
	return err == nil || errors.Is(err, syscall.EPERM)
}

func TestWriteToolPreservesMode(t *testing.T) {
	root := t.TempDir()
	s, err := New(root)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	path := filepath.Join(root, "script.sh")
	if err := os.WriteFile(path, []byte("echo hi\n"), 0o755); err != nil {
		t.Fatalf("write: %v", err)
	}
	if runtime.GOOS != "windows" {
		if st, err := os.Stat(path); err != nil {
			t.Fatalf("stat: %v", err)
		} else if st.Mode().Perm() != 0o755 {
			t.Skipf("filesystem does not preserve exec perms (got %o)", st.Mode().Perm())
		}
	}
	deps := tools.NewContainer()
	tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })
	tools.Provide(deps, ConfirmKey, func(context.Context) (Confirmer, error) { return allowConfirmer{}, nil })

	tool := writeTool()
	if _, err := tool.Execute(context.Background(), `{"file_path":"script.sh","content":"echo bye\n"}`, deps); err != nil {
		t.Fatalf("execute: %v", err)
	}
	st, err := os.Stat(path)
	if err != nil {
		t.Fatalf("stat: %v", err)
	}
	if st.Mode().Perm() != 0o755 {
		t.Fatalf("expected mode 0755, got %o", st.Mode().Perm())
	}
}

func TestWriteTool_DiffPreviewLimitsLargeExistingFile(t *testing.T) {
	root := t.TempDir()
	s, err := New(root)
	if err != nil {
		t.Fatalf("new: %v", err)
	}

	largeContent := bytes.Repeat([]byte("old\n"), int(maxWriteDiffBytes/4)+64)
	path := filepath.Join(root, "large.txt")
	if err := os.WriteFile(path, largeContent, 0o644); err != nil {
		t.Fatalf("seed file: %v", err)
	}

	var readCalls int
	useSandboxReadAllHook(t, func(r io.Reader) ([]byte, error) {
		readCalls++
		lr, ok := r.(*io.LimitedReader)
		if !ok {
			t.Fatalf("expected bounded preview reader, got %T", r)
		}
		if lr.N != maxWriteDiffBytes+1 {
			t.Fatalf("limited reader cap = %d, want %d", lr.N, maxWriteDiffBytes+1)
		}
		return io.ReadAll(r)
	})

	conf := &captureConfirmer{allow: true}
	deps := tools.NewContainer()
	tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })
	tools.Provide(deps, ConfirmKey, func(context.Context) (Confirmer, error) { return conf, nil })

	if _, err := writeTool().Execute(context.Background(), `{"file_path":"large.txt","content":"new content\n"}`, deps); err != nil {
		t.Fatalf("execute: %v", err)
	}
	if readCalls != 1 {
		t.Fatalf("expected exactly one bounded preview read, got %d", readCalls)
	}
	if conf.action != "write" {
		t.Fatalf("confirm action = %q, want write", conf.action)
	}

	var meta map[string]any
	if err := json.Unmarshal([]byte(conf.detail), &meta); err != nil {
		t.Fatalf("decode confirm detail: %v", err)
	}
	if got, ok := meta["diff_truncated"].(bool); !ok || !got {
		t.Fatalf("expected diff_truncated=true, got %#v", meta["diff_truncated"])
	}
	if got, ok := meta["diff_bytes_limit"].(float64); !ok || int64(got) != maxWriteDiffBytes {
		t.Fatalf("expected diff_bytes_limit=%d, got %#v", maxWriteDiffBytes, meta["diff_bytes_limit"])
	}
}

func TestWriteTool_ErrorDiagnosticSeverityActionFormat(t *testing.T) {
	root := t.TempDir()
	s, err := New(root)
	if err != nil {
		t.Fatalf("new: %v", err)
	}

	deps := tools.NewContainer()
	tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })

	out, err := writeTool().Execute(context.Background(), `{"file_path":"file.txt","content":"hello\n"}`, deps)
	if err == nil {
		t.Fatalf("expected confirmer error")
	}
	assertSandboxSeverityActionDiagnostic(t, out.PlainText())
}

func TestWriteTool_ErrorDiagnosticSeverityActionFormat_Security(t *testing.T) {
	root := t.TempDir()
	s, err := New(root)
	if err != nil {
		t.Fatalf("new: %v", err)
	}

	deps := tools.NewContainer()
	tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })

	out, err := writeTool().Execute(context.Background(), `{"file_path":"../outside.txt","content":"hello\n"}`, deps)
	if err == nil {
		t.Fatalf("expected security error")
	}
	assertSandboxSeverityActionDiagnostic(t, out.PlainText())
}

func TestWriteFilePreserveModeWriteFailureKeepsOriginal(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, "file.txt")
	if err := os.WriteFile(path, []byte("original\n"), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}

	origWriter := writeFileBytes
	writeFileBytes = func(_ *os.File, _ []byte) (int, error) {
		return 1, nil
	}
	t.Cleanup(func() {
		writeFileBytes = origWriter
	})

	err := writeFilePreserveMode(path, []byte("updated\n"), 0o644)
	if !errors.Is(err, io.ErrShortWrite) {
		t.Fatalf("expected short write error, got %v", err)
	}
	got, readErr := os.ReadFile(path)
	if readErr != nil {
		t.Fatalf("read: %v", readErr)
	}
	if string(got) != "original\n" {
		t.Fatalf("expected original content to remain, got %q", string(got))
	}
}

func TestApplyPatchKeepsOriginalWhenWriteFails(t *testing.T) {
	root := t.TempDir()
	s, err := New(root)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	path := filepath.Join(root, "file.txt")
	if err := os.WriteFile(path, []byte("hello\n"), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}

	origWriter := writeFileBytes
	writeFileBytes = func(_ *os.File, _ []byte) (int, error) {
		return 1, nil
	}
	t.Cleanup(func() {
		writeFileBytes = origWriter
	})

	deps := tools.NewContainer()
	tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })
	tools.Provide(deps, ConfirmKey, func(context.Context) (Confirmer, error) { return allowConfirmer{}, nil })

	patch := "*** Begin Patch\n*** Update File: file.txt\n@@ -1,1 +1,1 @@\n-hello\n+hi\n*** End Patch\n"
	_, err = applyPatchTool().Execute(context.Background(), fmt.Sprintf(`{"patch":%q}`, patch), deps)
	if !errors.Is(err, io.ErrShortWrite) {
		t.Fatalf("expected short write error, got %v", err)
	}
	got, readErr := os.ReadFile(path)
	if readErr != nil {
		t.Fatalf("read: %v", readErr)
	}
	if string(got) != "hello\n" {
		t.Fatalf("expected original content to remain, got %q", string(got))
	}
}

func TestEditToolRequiresUniqueMatch(t *testing.T) {
	root := t.TempDir()
	s, err := New(root)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	path := filepath.Join(root, "file.txt")
	content := "hello\nhello\n"
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}
	deps := tools.NewContainer()
	tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })
	tools.Provide(deps, ConfirmKey, func(context.Context) (Confirmer, error) { return allowConfirmer{}, nil })

	tool := editTool()
	if _, err := tool.Execute(context.Background(), `{"file_path":"file.txt","old_string":"hello","new_string":"hi"}`, deps); err == nil {
		t.Fatalf("expected unique-match error")
	}
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if string(b) != content {
		t.Fatalf("expected content unchanged")
	}
}

func TestEditToolReplaceAllWhenRequested(t *testing.T) {
	root := t.TempDir()
	s, err := New(root)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	path := filepath.Join(root, "file.txt")
	content := "hello\nhello\n"
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}
	deps := tools.NewContainer()
	tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })
	tools.Provide(deps, ConfirmKey, func(context.Context) (Confirmer, error) { return allowConfirmer{}, nil })

	tool := editTool()
	if _, err := tool.Execute(context.Background(), `{"file_path":"file.txt","old_string":"hello","new_string":"hi","replace_all":true}`, deps); err != nil {
		t.Fatalf("execute: %v", err)
	}
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if string(b) != "hi\nhi\n" {
		t.Fatalf("unexpected content: %q", string(b))
	}
}

func TestEditTool_ErrorDiagnosticSeverityActionFormat(t *testing.T) {
	root := t.TempDir()
	s, err := New(root)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	if err := os.WriteFile(filepath.Join(root, "file.txt"), []byte("hello\n"), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}

	deps := tools.NewContainer()
	tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })

	out, err := editTool().Execute(context.Background(), `{"file_path":"file.txt","old_string":"hello","new_string":"hi"}`, deps)
	if err == nil {
		t.Fatalf("expected confirmer error")
	}
	assertSandboxSeverityActionDiagnostic(t, out.PlainText())
}

func TestEditTool_ErrorDiagnosticSeverityActionFormat_Security(t *testing.T) {
	root := t.TempDir()
	s, err := New(root)
	if err != nil {
		t.Fatalf("new: %v", err)
	}

	deps := tools.NewContainer()
	tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })

	out, err := editTool().Execute(context.Background(), `{"file_path":"../outside.txt","old_string":"hello","new_string":"hi"}`, deps)
	if err == nil {
		t.Fatalf("expected security error")
	}
	assertSandboxSeverityActionDiagnostic(t, out.PlainText())
}

func TestEditTool_ErrorDiagnosticSeverityActionFormat_NotFound(t *testing.T) {
	root := t.TempDir()
	s, err := New(root)
	if err != nil {
		t.Fatalf("new: %v", err)
	}

	deps := tools.NewContainer()
	tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })

	out, err := editTool().Execute(context.Background(), `{"file_path":"missing.txt","old_string":"hello","new_string":"hi"}`, deps)
	if err == nil {
		t.Fatalf("expected not-found error")
	}
	assertSandboxSeverityActionDiagnostic(t, out.PlainText())
}

func TestEditTool_ErrorDiagnosticSeverityActionFormat_Directory(t *testing.T) {
	root := t.TempDir()
	s, err := New(root)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	if err := os.Mkdir(filepath.Join(root, "dir"), 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}

	deps := tools.NewContainer()
	tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })

	out, err := editTool().Execute(context.Background(), `{"file_path":"dir","old_string":"hello","new_string":"hi"}`, deps)
	if err == nil {
		t.Fatalf("expected directory error")
	}
	assertSandboxSeverityActionDiagnostic(t, out.PlainText())
}

func TestEditTool_ErrorDiagnosticSeverityActionFormat_StringNotFound(t *testing.T) {
	root := t.TempDir()
	s, err := New(root)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	if err := os.WriteFile(filepath.Join(root, "file.txt"), []byte("hello\n"), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}

	deps := tools.NewContainer()
	tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })

	out, err := editTool().Execute(context.Background(), `{"file_path":"file.txt","old_string":"missing","new_string":"hi"}`, deps)
	if err == nil {
		t.Fatalf("expected string-not-found error")
	}
	assertSandboxSeverityActionDiagnostic(t, out.PlainText())
}

func TestMultieditToolRequiresUniqueMatch(t *testing.T) {
	root := t.TempDir()
	s, err := New(root)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	path := filepath.Join(root, "file.txt")
	content := "hello\nhello\n"
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}
	deps := tools.NewContainer()
	tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })
	tools.Provide(deps, ConfirmKey, func(context.Context) (Confirmer, error) { return allowConfirmer{}, nil })

	tool := multieditTool()
	if _, err := tool.Execute(context.Background(), `{"file_path":"file.txt","edits":[{"old_string":"hello","new_string":"hi"}]}`, deps); err == nil {
		t.Fatalf("expected unique-match error")
	}
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if string(b) != content {
		t.Fatalf("expected content unchanged")
	}
}

func TestMultieditToolReplaceAllWhenRequested(t *testing.T) {
	root := t.TempDir()
	s, err := New(root)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	path := filepath.Join(root, "file.txt")
	if err := os.WriteFile(path, []byte("hello\nhello\n"), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}
	deps := tools.NewContainer()
	tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })
	tools.Provide(deps, ConfirmKey, func(context.Context) (Confirmer, error) { return allowConfirmer{}, nil })

	tool := multieditTool()
	if _, err := tool.Execute(context.Background(), `{"file_path":"file.txt","edits":[{"old_string":"hello","new_string":"hi","replace_all":true}]}`, deps); err != nil {
		t.Fatalf("execute: %v", err)
	}
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if string(b) != "hi\nhi\n" {
		t.Fatalf("unexpected content: %q", string(b))
	}
}

func TestMultieditTool_ErrorDiagnosticSeverityActionFormat(t *testing.T) {
	root := t.TempDir()
	s, err := New(root)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	if err := os.WriteFile(filepath.Join(root, "file.txt"), []byte("hello\n"), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}

	t.Run("missing file path", func(t *testing.T) {
		deps := tools.NewContainer()
		tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })

		out, err := multieditTool().Execute(context.Background(), `{"file_path":"   ","edits":[{"old_string":"hello","new_string":"hi"}]}`, deps)
		if err == nil {
			t.Fatalf("expected missing-file-path error")
		}
		assertSandboxSeverityActionDiagnostic(t, out.PlainText())
	})

	t.Run("empty edits", func(t *testing.T) {
		deps := tools.NewContainer()
		tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })

		out, err := multieditTool().Execute(context.Background(), `{"file_path":"file.txt","edits":[]}`, deps)
		if err == nil {
			t.Fatalf("expected empty-edits error")
		}
		assertSandboxSeverityActionDiagnostic(t, out.PlainText())
	})

	t.Run("empty old string", func(t *testing.T) {
		deps := tools.NewContainer()
		tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })

		out, err := multieditTool().Execute(context.Background(), `{"file_path":"file.txt","edits":[{"old_string":"","new_string":"hi"}]}`, deps)
		if err == nil {
			t.Fatalf("expected empty-old-string error")
		}
		assertSandboxSeverityActionDiagnostic(t, out.PlainText())
	})

	t.Run("confirmer error", func(t *testing.T) {
		deps := tools.NewContainer()
		tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })

		out, err := multieditTool().Execute(context.Background(), `{"file_path":"file.txt","edits":[{"old_string":"hello","new_string":"hi"}]}`, deps)
		if err == nil {
			t.Fatalf("expected confirmer error")
		}
		assertSandboxSeverityActionDiagnostic(t, out.PlainText())
	})
}

func TestMultieditTool_ErrorDiagnosticSeverityActionFormat_Security(t *testing.T) {
	root := t.TempDir()
	s, err := New(root)
	if err != nil {
		t.Fatalf("new: %v", err)
	}

	deps := tools.NewContainer()
	tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })

	out, err := multieditTool().Execute(context.Background(), `{"file_path":"../outside.txt","edits":[{"old_string":"hello","new_string":"hi"}]}`, deps)
	if err == nil {
		t.Fatalf("expected security error")
	}
	assertSandboxSeverityActionDiagnostic(t, out.PlainText())
}

func TestMultieditTool_ErrorDiagnosticSeverityActionFormat_NotFound(t *testing.T) {
	root := t.TempDir()
	s, err := New(root)
	if err != nil {
		t.Fatalf("new: %v", err)
	}

	deps := tools.NewContainer()
	tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })

	out, err := multieditTool().Execute(context.Background(), `{"file_path":"missing.txt","edits":[{"old_string":"hello","new_string":"hi"}]}`, deps)
	if err == nil {
		t.Fatalf("expected not-found error")
	}
	assertSandboxSeverityActionDiagnostic(t, out.PlainText())
}

func TestMultieditTool_ErrorDiagnosticSeverityActionFormat_Directory(t *testing.T) {
	root := t.TempDir()
	s, err := New(root)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	if err := os.Mkdir(filepath.Join(root, "dir"), 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}

	deps := tools.NewContainer()
	tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })

	out, err := multieditTool().Execute(context.Background(), `{"file_path":"dir","edits":[{"old_string":"hello","new_string":"hi"}]}`, deps)
	if err == nil {
		t.Fatalf("expected directory error")
	}
	assertSandboxSeverityActionDiagnostic(t, out.PlainText())
}

func TestMultieditTool_ErrorDiagnosticSeverityActionFormat_StringNotFound(t *testing.T) {
	root := t.TempDir()
	s, err := New(root)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	if err := os.WriteFile(filepath.Join(root, "file.txt"), []byte("hello\n"), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}

	deps := tools.NewContainer()
	tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })

	out, err := multieditTool().Execute(context.Background(), `{"file_path":"file.txt","edits":[{"old_string":"missing","new_string":"hi"}]}`, deps)
	if err == nil {
		t.Fatalf("expected string-not-found error")
	}
	assertSandboxSeverityActionDiagnostic(t, out.PlainText())
}

func TestEditToolRejectsLargeFile(t *testing.T) {
	root := t.TempDir()
	s, err := New(root)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	path := filepath.Join(root, "big.txt")
	f, err := os.Create(path)
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	if err := f.Truncate(maxEditFileBytes + 1); err != nil {
		_ = f.Close()
		t.Fatalf("truncate: %v", err)
	}
	if _, err := f.WriteAt([]byte("hello"), 0); err != nil {
		_ = f.Close()
		t.Fatalf("write: %v", err)
	}
	if err := f.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}

	deps := tools.NewContainer()
	tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })
	tools.Provide(deps, ConfirmKey, func(context.Context) (Confirmer, error) { return allowConfirmer{}, nil })

	tool := editTool()
	if _, err := tool.Execute(context.Background(), `{"file_path":"big.txt","old_string":"hello","new_string":"hi"}`, deps); err == nil {
		t.Fatalf("expected size error")
	} else if !strings.Contains(err.Error(), "file too large") {
		t.Fatalf("expected size error, got %v", err)
	}
	f, err = os.Open(path)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	defer f.Close()
	buf := make([]byte, 5)
	if _, err := f.ReadAt(buf, 0); err != nil {
		t.Fatalf("read: %v", err)
	}
	if string(buf) != "hello" {
		t.Fatalf("expected content unchanged")
	}
}

func TestEditTool_RejectsLargeFileBeforeRead(t *testing.T) {
	root := t.TempDir()
	s, err := New(root)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	path := filepath.Join(root, "big.txt")
	f, err := os.Create(path)
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	if err := f.Truncate(maxEditFileBytes + 1); err != nil {
		_ = f.Close()
		t.Fatalf("truncate: %v", err)
	}
	if _, err := f.WriteAt([]byte("hello"), 0); err != nil {
		_ = f.Close()
		t.Fatalf("write: %v", err)
	}
	if err := f.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}

	useSandboxReadAllHook(t, func(io.Reader) ([]byte, error) {
		return nil, errors.New("unexpected full read")
	})

	deps := tools.NewContainer()
	tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })
	tools.Provide(deps, ConfirmKey, func(context.Context) (Confirmer, error) { return allowConfirmer{}, nil })

	_, err = editTool().Execute(context.Background(), `{"file_path":"big.txt","old_string":"hello","new_string":"hi"}`, deps)
	if err == nil {
		t.Fatalf("expected size error")
	}
	if !strings.Contains(err.Error(), "file too large") {
		t.Fatalf("expected file-too-large error, got %v", err)
	}
	if strings.Contains(err.Error(), "unexpected full read") {
		t.Fatalf("expected size check before read, got %v", err)
	}
}

func TestMultieditTool_RejectsLargeFileBeforeRead(t *testing.T) {
	root := t.TempDir()
	s, err := New(root)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	path := filepath.Join(root, "big.txt")
	f, err := os.Create(path)
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	if err := f.Truncate(maxEditFileBytes + 1); err != nil {
		_ = f.Close()
		t.Fatalf("truncate: %v", err)
	}
	if _, err := f.WriteAt([]byte("hello"), 0); err != nil {
		_ = f.Close()
		t.Fatalf("write: %v", err)
	}
	if err := f.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}

	useSandboxReadAllHook(t, func(io.Reader) ([]byte, error) {
		return nil, errors.New("unexpected full read")
	})

	deps := tools.NewContainer()
	tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })
	tools.Provide(deps, ConfirmKey, func(context.Context) (Confirmer, error) { return allowConfirmer{}, nil })

	_, err = multieditTool().Execute(context.Background(), `{"file_path":"big.txt","edits":[{"old_string":"hello","new_string":"hi"}]}`, deps)
	if err == nil {
		t.Fatalf("expected size error")
	}
	if !strings.Contains(err.Error(), "file too large") {
		t.Fatalf("expected file-too-large error, got %v", err)
	}
	if strings.Contains(err.Error(), "unexpected full read") {
		t.Fatalf("expected size check before read, got %v", err)
	}
}

func TestSandboxToolsReturnErrors(t *testing.T) {
	root := t.TempDir()
	s, err := New(root)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	path := filepath.Join(root, "file.txt")
	if err := os.WriteFile(path, []byte("hello\n"), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}

	deps := tools.NewContainer()
	tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })
	tools.Provide(deps, ConfirmKey, func(context.Context) (Confirmer, error) { return allowConfirmer{}, nil })

	if _, err := webfetchTool().Execute(context.Background(), `{}`, deps); err == nil {
		t.Fatalf("expected webfetch error for missing url")
	}
	if _, err := lsTool().Execute(context.Background(), `{"path":"file.txt"}`, deps); err == nil {
		t.Fatalf("expected ls error for non-directory")
	}
	if _, err := globTool().Execute(context.Background(), `{"pattern":""}`, deps); err == nil {
		t.Fatalf("expected glob error for empty pattern")
	}
	if _, err := grepTool().Execute(context.Background(), `{"pattern":"["}`, deps); err == nil {
		t.Fatalf("expected grep error for invalid regex")
	}
}

func TestLsToolInvalidIgnorePatternReturnsError(t *testing.T) {
	root := t.TempDir()
	s, err := New(root)
	if err != nil {
		t.Fatalf("new: %v", err)
	}

	deps := tools.NewContainer()
	tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })

	res, err := lsTool().Execute(context.Background(), `{"ignore":["[abc"]}`, deps)
	if err == nil {
		t.Fatalf("expected invalid ignore glob error")
	}
	if !strings.Contains(err.Error(), "invalid ignore pattern for ls tool") {
		t.Fatalf("expected invalid ls ignore error, got %v", err)
	}
	if !strings.Contains(err.Error(), "[abc") {
		t.Fatalf("expected pattern context in error, got %v", err)
	}
	if !strings.Contains(res.PlainText(), "invalid ignore pattern for ls tool") {
		t.Fatalf("expected user-facing error content, got %q", res.PlainText())
	}
}

func TestLsTool_StopsReadingAfterScanCapWithActionableWarning(t *testing.T) {
	root := t.TempDir()
	s, err := New(root)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	for i := 0; i < maxLsScanEntries+32; i++ {
		path := filepath.Join(root, fmt.Sprintf("entry-%03d.txt", i))
		if err := os.WriteFile(path, []byte("hello\n"), 0o644); err != nil {
			t.Fatalf("write: %v", err)
		}
	}

	deps := tools.NewContainer()
	tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })

	ctx := tools.WithToolResultMetadata(context.Background())
	res, err := lsTool().Execute(ctx, `{"ignore":["*"]}`, deps)
	if err != nil {
		t.Fatalf("execute: %v", err)
	}
	out := res.PlainText()
	if !strings.Contains(out, "(empty)") {
		t.Fatalf("expected empty body when all entries ignored, got %q", out)
	}
	if !strings.Contains(out, "[WARN] ls scan stopped after") {
		t.Fatalf("expected scan-cap warning, got %q", out)
	}
	idx := strings.LastIndex(out, "[WARN]")
	if idx < 0 {
		t.Fatalf("expected warning line in output, got %q", out)
	}
	assertSandboxSeverityActionWarningDiagnostic(t, out[idx:])

	meta := tools.ToolResultMetadataSnapshot(ctx)
	if meta == nil {
		t.Fatalf("expected metadata")
	}
	if got, ok := meta["scan_truncated"].(bool); !ok || !got {
		t.Fatalf("expected scan_truncated=true, got %#v", meta["scan_truncated"])
	}
	if got, ok := meta["scan_cap"].(int); !ok || got != maxLsScanEntries {
		t.Fatalf("expected scan_cap=%d, got %#v", maxLsScanEntries, meta["scan_cap"])
	}
	if got, ok := meta["scanned_entries"].(int); !ok || got != maxLsScanEntries {
		t.Fatalf("expected scanned_entries=%d, got %#v", maxLsScanEntries, meta["scanned_entries"])
	}
	if got, ok := meta["skipped_due_to_cap"].(bool); !ok || !got {
		t.Fatalf("expected skipped_due_to_cap=true, got %#v", meta["skipped_due_to_cap"])
	}
	if kind, _ := meta["warning_kind"].(string); kind != scanCapWarningKind {
		t.Fatalf("expected warning_kind=%q, got %#v", scanCapWarningKind, meta["warning_kind"])
	}
	if reason, _ := meta["truncated_reason"].(string); reason != "scan_cap" {
		t.Fatalf("expected truncated_reason=scan_cap, got %#v", meta["truncated_reason"])
	}
}

func TestGrepToolInvalidGlobPatternReturnsError(t *testing.T) {
	root := t.TempDir()
	s, err := New(root)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	if err := os.WriteFile(filepath.Join(root, "visible.txt"), []byte("needle\n"), 0o644); err != nil {
		t.Fatalf("write visible: %v", err)
	}

	deps := tools.NewContainer()
	tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })

	res, err := grepTool().Execute(context.Background(), `{"pattern":"needle","glob":"[abc"}`, deps)
	if err == nil {
		t.Fatalf("expected invalid grep glob error")
	}
	if !strings.Contains(err.Error(), "invalid glob pattern for grep tool") {
		t.Fatalf("expected invalid grep glob error, got %v", err)
	}
	if !strings.Contains(err.Error(), "[abc") {
		t.Fatalf("expected pattern context in error, got %v", err)
	}
	if !strings.Contains(res.PlainText(), "invalid glob pattern for grep tool") {
		t.Fatalf("expected user-facing error content, got %q", res.PlainText())
	}
}

func TestWebfetchToolTruncationNotice(t *testing.T) {
	useSandboxPublicWebfetchResolver(t)

	origTransport := http.DefaultTransport
	http.DefaultTransport = roundTripFunc(func(r *http.Request) (*http.Response, error) {
		body := io.NopCloser(strings.NewReader("abcdefghijklmnopqrstuvwxyz"))
		return &http.Response{
			Status:     "200 OK",
			StatusCode: http.StatusOK,
			Body:       body,
			Header:     make(http.Header),
			Request:    r,
		}, nil
	})
	t.Cleanup(func() { http.DefaultTransport = origTransport })

	deps := tools.NewContainer()
	tools.Provide(deps, ConfirmKey, func(context.Context) (Confirmer, error) { return allowConfirmer{}, nil })

	res, err := webfetchTool().Execute(context.Background(), `{"url":"https://example.test","max_bytes":10}`, deps)
	if err != nil {
		t.Fatalf("execute: %v", err)
	}
	out := res.PlainText()
	if !strings.Contains(out, "200 OK") {
		t.Fatalf("expected status line, got %q", out)
	}
	if !strings.Contains(out, "truncated after 10 bytes") {
		t.Fatalf("expected truncation notice, got %q", out)
	}
	if !strings.Contains(out, "Increase max_bytes") {
		t.Fatalf("expected guidance to increase max_bytes, got %q", out)
	}
	if !strings.Contains(out, "abcdefghij") {
		t.Fatalf("expected truncated body, got %q", out)
	}
	if strings.Contains(out, "klm") {
		t.Fatalf("expected body to be truncated, got %q", out)
	}
}

func TestWebfetchToolReadErrorIsSurfaced(t *testing.T) {
	useSandboxPublicWebfetchResolver(t)

	origTransport := http.DefaultTransport
	http.DefaultTransport = roundTripFunc(func(r *http.Request) (*http.Response, error) {
		body := &failAfterReader{data: []byte("partial")}
		return &http.Response{
			Status:     "200 OK",
			StatusCode: http.StatusOK,
			Body:       body,
			Header:     make(http.Header),
			Request:    r,
		}, nil
	})
	t.Cleanup(func() { http.DefaultTransport = origTransport })

	deps := tools.NewContainer()
	tools.Provide(deps, ConfirmKey, func(context.Context) (Confirmer, error) { return allowConfirmer{}, nil })

	res, err := webfetchTool().Execute(context.Background(), `{"url":"https://example.test"}`, deps)
	if err == nil {
		t.Fatalf("expected read error")
	}
	if !strings.Contains(err.Error(), "read response body after") {
		t.Fatalf("expected read error context, got %v", err)
	}
	if !strings.Contains(err.Error(), "simulated read failure") {
		t.Fatalf("expected wrapped read failure, got %v", err)
	}
	out := res.PlainText()
	if !strings.Contains(out, "partial body") {
		t.Fatalf("expected partial-body marker, got %q", out)
	}
}

func TestWebfetchToolBlocksPrivateDestination(t *testing.T) {
	useSandboxPublicWebfetchResolver(t)

	origTransport := http.DefaultTransport
	http.DefaultTransport = roundTripFunc(func(*http.Request) (*http.Response, error) {
		t.Fatalf("webfetch request should not run for blocked private destination")
		return nil, nil
	})
	t.Cleanup(func() { http.DefaultTransport = origTransport })

	deps := tools.NewContainer()
	tools.Provide(deps, ConfirmKey, func(context.Context) (Confirmer, error) { return allowConfirmer{}, nil })

	tests := []struct {
		name          string
		url           string
		wantSubstring string
	}{
		{name: "loopback", url: "http://127.0.0.1:8080", wantSubstring: "loopback"},
		{name: "rfc1918", url: "http://10.1.2.3", wantSubstring: "private"},
		{name: "link-local", url: "http://169.254.10.20", wantSubstring: "link-local"},
		{name: "private-dns", url: "http://internal.test/private", wantSubstring: "private"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			args := fmt.Sprintf(`{"url":%q}`, tt.url)
			res, err := webfetchTool().Execute(context.Background(), args, deps)
			if err == nil {
				t.Fatalf("expected destination block for %s", tt.url)
			}
			gotErr := strings.ToLower(err.Error())
			if !strings.Contains(gotErr, "blocked request target") {
				t.Fatalf("expected blocked request-target diagnostic, got %q", err.Error())
			}
			if !strings.Contains(gotErr, tt.wantSubstring) {
				t.Fatalf("expected %q diagnostic, got %q", tt.wantSubstring, err.Error())
			}
			if got := strings.ToLower(res.PlainText()); !strings.Contains(got, tt.wantSubstring) {
				t.Fatalf("expected user-facing %q diagnostic, got %q", tt.wantSubstring, res.PlainText())
			}
		})
	}
}

func TestWebfetchToolBlocksRedirectToPrivateDestination(t *testing.T) {
	useSandboxPublicWebfetchResolver(t)

	origTransport := http.DefaultTransport
	var calls int
	http.DefaultTransport = roundTripFunc(func(r *http.Request) (*http.Response, error) {
		calls++
		if calls > 1 {
			t.Fatalf("redirect request should not run for blocked private destination")
		}
		resp := &http.Response{
			Status:     "302 Found",
			StatusCode: http.StatusFound,
			Header:     make(http.Header),
			Body:       io.NopCloser(strings.NewReader("redirect")),
			Request:    r,
		}
		resp.Header.Set("Location", "http://internal.test/private")
		return resp, nil
	})
	t.Cleanup(func() { http.DefaultTransport = origTransport })

	deps := tools.NewContainer()
	tools.Provide(deps, ConfirmKey, func(context.Context) (Confirmer, error) { return allowConfirmer{}, nil })

	res, err := webfetchTool().Execute(context.Background(), `{"url":"http://public.test/start"}`, deps)
	if err == nil {
		t.Fatalf("expected redirect destination block")
	}
	gotErr := strings.ToLower(err.Error())
	if !strings.Contains(gotErr, "redirect target") {
		t.Fatalf("expected redirect-target diagnostic, got %q", err.Error())
	}
	if !strings.Contains(gotErr, "private") {
		t.Fatalf("expected private-destination diagnostic, got %q", err.Error())
	}
	if calls != 1 {
		t.Fatalf("expected exactly one outbound request before redirect block, got %d", calls)
	}
	if got := strings.ToLower(res.PlainText()); !strings.Contains(got, "redirect target") {
		t.Fatalf("expected user-facing redirect-target diagnostic, got %q", res.PlainText())
	}
}

func TestReadToolRejectsLargeFile(t *testing.T) {
	root := t.TempDir()
	s, err := New(root)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	path := filepath.Join(root, "big.txt")
	f, err := os.Create(path)
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	if err := f.Truncate(maxReadFileBytes + 1); err != nil {
		_ = f.Close()
		t.Fatalf("truncate: %v", err)
	}
	if _, err := f.WriteAt([]byte("hello"), 0); err != nil {
		_ = f.Close()
		t.Fatalf("write: %v", err)
	}
	if err := f.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}

	deps := tools.NewContainer()
	tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })

	res, err := readTool().Execute(context.Background(), `{"file_path":"big.txt"}`, deps)
	if err == nil {
		t.Fatalf("expected size error")
	}
	if !strings.Contains(err.Error(), "file too large") {
		t.Fatalf("expected size error, got %v", err)
	}
	if !strings.Contains(res.PlainText(), "read refuses to load") {
		t.Fatalf("expected size message, got %q", res.PlainText())
	}
}

func TestReadTool_BlocksSymlinkSwapAfterResolve(t *testing.T) {
	root := t.TempDir()
	outside := t.TempDir()

	insideDir := filepath.Join(root, "inside")
	if err := os.MkdirAll(insideDir, 0o755); err != nil {
		t.Fatalf("mkdir inside: %v", err)
	}
	insideFile := filepath.Join(insideDir, "target.txt")
	if err := os.WriteFile(insideFile, []byte("inside\n"), 0o644); err != nil {
		t.Fatalf("write inside file: %v", err)
	}
	outsideFile := filepath.Join(outside, "target.txt")
	if err := os.WriteFile(outsideFile, []byte("outside\n"), 0o644); err != nil {
		t.Fatalf("write outside file: %v", err)
	}
	link := filepath.Join(root, "swap")
	if err := os.Symlink(insideDir, link); err != nil {
		t.Skipf("symlink not supported: %v", err)
	}

	s, err := New(root)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	expectedAbs := filepath.Clean(filepath.Join(root, "swap", "target.txt"))
	var once sync.Once
	var hookErr error
	useSandboxPathRevalidateHook(t, func(abs string) {
		if filepath.Clean(abs) != expectedAbs {
			return
		}
		once.Do(func() {
			if err := os.Remove(link); err != nil {
				hookErr = err
				return
			}
			if err := os.Symlink(outside, link); err != nil {
				hookErr = err
			}
		})
	})

	deps := tools.NewContainer()
	tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })

	res, err := readTool().Execute(context.Background(), `{"file_path":"swap/target.txt"}`, deps)
	if hookErr != nil {
		t.Fatalf("hook error: %v", hookErr)
	}
	if err == nil {
		t.Fatalf("expected security error")
	}
	var secErr *SecurityError
	if !errors.As(err, &secErr) {
		t.Fatalf("expected SecurityError, got %v", err)
	}
	if !strings.Contains(strings.ToLower(res.PlainText()), "security error") {
		t.Fatalf("expected security diagnostic, got %q", res.PlainText())
	}
}

func TestWriteTool_BlocksSymlinkSwapAfterResolve(t *testing.T) {
	root := t.TempDir()
	outside := t.TempDir()

	insideDir := filepath.Join(root, "inside")
	if err := os.MkdirAll(insideDir, 0o755); err != nil {
		t.Fatalf("mkdir inside: %v", err)
	}
	insideFile := filepath.Join(insideDir, "target.txt")
	if err := os.WriteFile(insideFile, []byte("inside-before\n"), 0o644); err != nil {
		t.Fatalf("write inside file: %v", err)
	}
	outsideFile := filepath.Join(outside, "target.txt")
	if err := os.WriteFile(outsideFile, []byte("outside-before\n"), 0o644); err != nil {
		t.Fatalf("write outside file: %v", err)
	}
	link := filepath.Join(root, "swap")
	if err := os.Symlink(insideDir, link); err != nil {
		t.Skipf("symlink not supported: %v", err)
	}

	s, err := New(root)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	expectedAbs := filepath.Clean(filepath.Join(root, "swap", "target.txt"))
	var once sync.Once
	var hookErr error
	useSandboxPathRevalidateHook(t, func(abs string) {
		if filepath.Clean(abs) != expectedAbs {
			return
		}
		once.Do(func() {
			if err := os.Remove(link); err != nil {
				hookErr = err
				return
			}
			if err := os.Symlink(outside, link); err != nil {
				hookErr = err
			}
		})
	})

	deps := tools.NewContainer()
	tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })
	tools.Provide(deps, ConfirmKey, func(context.Context) (Confirmer, error) { return allowConfirmer{}, nil })

	_, err = writeTool().Execute(context.Background(), `{"file_path":"swap/target.txt","content":"updated\n"}`, deps)
	if hookErr != nil {
		t.Fatalf("hook error: %v", hookErr)
	}
	if err == nil {
		t.Fatalf("expected security error")
	}
	var secErr *SecurityError
	if !errors.As(err, &secErr) {
		t.Fatalf("expected SecurityError, got %v", err)
	}
	gotOutside, readOutsideErr := os.ReadFile(outsideFile)
	if readOutsideErr != nil {
		t.Fatalf("read outside: %v", readOutsideErr)
	}
	if string(gotOutside) != "outside-before\n" {
		t.Fatalf("expected outside file unchanged, got %q", string(gotOutside))
	}
	gotInside, readInsideErr := os.ReadFile(insideFile)
	if readInsideErr != nil {
		t.Fatalf("read inside: %v", readInsideErr)
	}
	if string(gotInside) != "inside-before\n" {
		t.Fatalf("expected inside file unchanged, got %q", string(gotInside))
	}
}

func TestEditTool_BlocksSymlinkSwapAfterResolve(t *testing.T) {
	root := t.TempDir()
	outside := t.TempDir()

	insideDir := filepath.Join(root, "inside")
	if err := os.MkdirAll(insideDir, 0o755); err != nil {
		t.Fatalf("mkdir inside: %v", err)
	}
	insideFile := filepath.Join(insideDir, "target.txt")
	if err := os.WriteFile(insideFile, []byte("hello\n"), 0o644); err != nil {
		t.Fatalf("write inside file: %v", err)
	}
	outsideFile := filepath.Join(outside, "target.txt")
	if err := os.WriteFile(outsideFile, []byte("outside-before\n"), 0o644); err != nil {
		t.Fatalf("write outside file: %v", err)
	}
	link := filepath.Join(root, "swap")
	if err := os.Symlink(insideDir, link); err != nil {
		t.Skipf("symlink not supported: %v", err)
	}

	s, err := New(root)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	expectedAbs := filepath.Clean(filepath.Join(root, "swap", "target.txt"))
	var once sync.Once
	var hookErr error
	useSandboxPathRevalidateHook(t, func(abs string) {
		if filepath.Clean(abs) != expectedAbs {
			return
		}
		once.Do(func() {
			if err := os.Remove(link); err != nil {
				hookErr = err
				return
			}
			if err := os.Symlink(outside, link); err != nil {
				hookErr = err
			}
		})
	})

	deps := tools.NewContainer()
	tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })
	tools.Provide(deps, ConfirmKey, func(context.Context) (Confirmer, error) { return allowConfirmer{}, nil })

	_, err = editTool().Execute(context.Background(), `{"file_path":"swap/target.txt","old_string":"hello","new_string":"hi"}`, deps)
	if hookErr != nil {
		t.Fatalf("hook error: %v", hookErr)
	}
	if err == nil {
		t.Fatalf("expected security error")
	}
	var secErr *SecurityError
	if !errors.As(err, &secErr) {
		t.Fatalf("expected SecurityError, got %v", err)
	}
	gotOutside, readOutsideErr := os.ReadFile(outsideFile)
	if readOutsideErr != nil {
		t.Fatalf("read outside: %v", readOutsideErr)
	}
	if string(gotOutside) != "outside-before\n" {
		t.Fatalf("expected outside file unchanged, got %q", string(gotOutside))
	}
	gotInside, readInsideErr := os.ReadFile(insideFile)
	if readInsideErr != nil {
		t.Fatalf("read inside: %v", readInsideErr)
	}
	if string(gotInside) != "hello\n" {
		t.Fatalf("expected inside file unchanged, got %q", string(gotInside))
	}
}

func TestLsTool_BlocksSymlinkSwapAfterResolve(t *testing.T) {
	root := t.TempDir()
	outside := t.TempDir()

	insideDir := filepath.Join(root, "inside")
	if err := os.MkdirAll(insideDir, 0o755); err != nil {
		t.Fatalf("mkdir inside: %v", err)
	}
	if err := os.WriteFile(filepath.Join(insideDir, "inside.txt"), []byte("inside\n"), 0o644); err != nil {
		t.Fatalf("write inside file: %v", err)
	}
	if err := os.WriteFile(filepath.Join(outside, "outside.txt"), []byte("outside\n"), 0o644); err != nil {
		t.Fatalf("write outside file: %v", err)
	}
	link := filepath.Join(root, "swap")
	if err := os.Symlink(insideDir, link); err != nil {
		t.Skipf("symlink not supported: %v", err)
	}

	s, err := New(root)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	expectedAbs := filepath.Clean(filepath.Join(root, "swap"))
	var once sync.Once
	var hookErr error
	useSandboxPathRevalidateHook(t, func(abs string) {
		if filepath.Clean(abs) != expectedAbs {
			return
		}
		once.Do(func() {
			if err := os.Remove(link); err != nil {
				hookErr = err
				return
			}
			if err := os.Symlink(outside, link); err != nil {
				hookErr = err
			}
		})
	})

	deps := tools.NewContainer()
	tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })

	_, err = lsTool().Execute(context.Background(), `{"path":"swap"}`, deps)
	if hookErr != nil {
		t.Fatalf("hook error: %v", hookErr)
	}
	if err == nil {
		t.Fatalf("expected security error")
	}
	var secErr *SecurityError
	if !errors.As(err, &secErr) {
		t.Fatalf("expected SecurityError, got %v", err)
	}
}

func TestGlobTool_BlocksSymlinkSwapAfterResolve(t *testing.T) {
	root := t.TempDir()
	outside := t.TempDir()

	insideDir := filepath.Join(root, "inside")
	if err := os.MkdirAll(insideDir, 0o755); err != nil {
		t.Fatalf("mkdir inside: %v", err)
	}
	if err := os.WriteFile(filepath.Join(insideDir, "inside.txt"), []byte("inside\n"), 0o644); err != nil {
		t.Fatalf("write inside file: %v", err)
	}
	if err := os.WriteFile(filepath.Join(outside, "outside.txt"), []byte("outside\n"), 0o644); err != nil {
		t.Fatalf("write outside file: %v", err)
	}
	link := filepath.Join(root, "swap")
	if err := os.Symlink(insideDir, link); err != nil {
		t.Skipf("symlink not supported: %v", err)
	}

	s, err := New(root)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	expectedAbs := filepath.Clean(filepath.Join(root, "swap"))
	var once sync.Once
	var hookErr error
	useSandboxPathRevalidateHook(t, func(abs string) {
		if filepath.Clean(abs) != expectedAbs {
			return
		}
		once.Do(func() {
			if err := os.Remove(link); err != nil {
				hookErr = err
				return
			}
			if err := os.Symlink(outside, link); err != nil {
				hookErr = err
			}
		})
	})

	deps := tools.NewContainer()
	tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })

	_, err = globTool().Execute(context.Background(), `{"path":"swap","pattern":"*.txt"}`, deps)
	if hookErr != nil {
		t.Fatalf("hook error: %v", hookErr)
	}
	if err == nil {
		t.Fatalf("expected security error")
	}
	var secErr *SecurityError
	if !errors.As(err, &secErr) {
		t.Fatalf("expected SecurityError, got %v", err)
	}
}

func TestGrepTool_BlocksSymlinkSwapAfterResolve(t *testing.T) {
	root := t.TempDir()
	outside := t.TempDir()

	insideDir := filepath.Join(root, "inside")
	if err := os.MkdirAll(insideDir, 0o755); err != nil {
		t.Fatalf("mkdir inside: %v", err)
	}
	if err := os.WriteFile(filepath.Join(insideDir, "inside.txt"), []byte("needle\n"), 0o644); err != nil {
		t.Fatalf("write inside file: %v", err)
	}
	if err := os.WriteFile(filepath.Join(outside, "outside.txt"), []byte("needle\n"), 0o644); err != nil {
		t.Fatalf("write outside file: %v", err)
	}
	link := filepath.Join(root, "swap")
	if err := os.Symlink(insideDir, link); err != nil {
		t.Skipf("symlink not supported: %v", err)
	}

	s, err := New(root)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	expectedAbs := filepath.Clean(filepath.Join(root, "swap"))
	var once sync.Once
	var hookErr error
	useSandboxPathRevalidateHook(t, func(abs string) {
		if filepath.Clean(abs) != expectedAbs {
			return
		}
		once.Do(func() {
			if err := os.Remove(link); err != nil {
				hookErr = err
				return
			}
			if err := os.Symlink(outside, link); err != nil {
				hookErr = err
			}
		})
	})

	deps := tools.NewContainer()
	tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })

	_, err = grepTool().Execute(context.Background(), `{"path":"swap","pattern":"needle"}`, deps)
	if hookErr != nil {
		t.Fatalf("hook error: %v", hookErr)
	}
	if err == nil {
		t.Fatalf("expected security error")
	}
	var secErr *SecurityError
	if !errors.As(err, &secErr) {
		t.Fatalf("expected SecurityError, got %v", err)
	}
}

func TestGlobTool_BlocksSymlinkFileEscapeOutsideSandbox(t *testing.T) {
	root := t.TempDir()
	outside := t.TempDir()

	if err := os.WriteFile(filepath.Join(root, "inside.txt"), []byte("inside\n"), 0o644); err != nil {
		t.Fatalf("write inside file: %v", err)
	}
	outsideFile := filepath.Join(outside, "secret.txt")
	if err := os.WriteFile(outsideFile, []byte("top secret\n"), 0o644); err != nil {
		t.Fatalf("write outside file: %v", err)
	}
	link := filepath.Join(root, "link.txt")
	if err := os.Symlink(outsideFile, link); err != nil {
		t.Skipf("symlink not supported: %v", err)
	}

	s, err := New(root)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	deps := tools.NewContainer()
	tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })

	ctx := tools.WithToolResultMetadata(context.Background())
	res, err := globTool().Execute(ctx, `{"pattern":"*.txt"}`, deps)
	if err != nil {
		t.Fatalf("execute: %v", err)
	}
	out := res.PlainText()
	if !strings.Contains(out, "inside.txt") {
		t.Fatalf("expected inside file in output, got %q", out)
	}
	if !strings.Contains(out, "[WARN] glob skipped 1 matched path(s) that are symbolic links") {
		t.Fatalf("expected symlink warning in output, got %q", out)
	}

	meta := tools.ToolResultMetadataSnapshot(ctx)
	if meta == nil {
		t.Fatal("expected metadata")
	}
	if got, ok := meta["count"].(int); !ok || got != 1 {
		t.Fatalf("expected count=1, got %#v", meta["count"])
	}
	if got, ok := meta["skipped_symlink_paths"].(int); !ok || got != 1 {
		t.Fatalf("expected skipped_symlink_paths=1, got %#v", meta["skipped_symlink_paths"])
	}
	if reason, _ := meta["skipped_reason"].(string); reason != "symlink_target" {
		t.Fatalf("expected skipped_reason symlink_target, got %#v", meta["skipped_reason"])
	}
	if samples, ok := meta["skipped_symlink_samples"].([]string); !ok || len(samples) != 1 || samples[0] != "link.txt" {
		t.Fatalf("expected skipped_symlink_samples [link.txt], got %#v", meta["skipped_symlink_samples"])
	}
}

func TestGrepTool_BlocksSymlinkFileEscapeOutsideSandbox(t *testing.T) {
	root := t.TempDir()
	outside := t.TempDir()

	if err := os.WriteFile(filepath.Join(root, "inside.txt"), []byte("inside\n"), 0o644); err != nil {
		t.Fatalf("write inside file: %v", err)
	}
	outsideFile := filepath.Join(outside, "secret.txt")
	if err := os.WriteFile(outsideFile, []byte("TOP_SECRET_NEEDLE\n"), 0o644); err != nil {
		t.Fatalf("write outside file: %v", err)
	}
	link := filepath.Join(root, "link.txt")
	if err := os.Symlink(outsideFile, link); err != nil {
		t.Skipf("symlink not supported: %v", err)
	}

	s, err := New(root)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	deps := tools.NewContainer()
	tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })

	ctx := tools.WithToolResultMetadata(context.Background())
	res, err := grepTool().Execute(ctx, `{"pattern":"TOP_SECRET_NEEDLE"}`, deps)
	if err != nil {
		t.Fatalf("execute: %v", err)
	}
	out := res.PlainText()
	if !strings.Contains(out, "No matches for: TOP_SECRET_NEEDLE") {
		t.Fatalf("expected no matches output, got %q", out)
	}
	if !strings.Contains(out, "[WARN] grep skipped 1 path(s) due scan errors") {
		t.Fatalf("expected scan warning in output, got %q", out)
	}
	if !strings.Contains(out, "link.txt") {
		t.Fatalf("expected skipped symlink path in warning, got %q", out)
	}

	meta := tools.ToolResultMetadataSnapshot(ctx)
	if meta == nil {
		t.Fatal("expected metadata")
	}
	if reason, _ := meta["skipped_reason"].(string); reason != "scan_error" {
		t.Fatalf("expected skipped_reason scan_error, got %#v", meta["skipped_reason"])
	}
	if got, ok := meta["skipped_scan_errors"].(int); !ok || got != 1 {
		t.Fatalf("expected skipped_scan_errors=1, got %#v", meta["skipped_scan_errors"])
	}
	if samples, ok := meta["skipped_scan_samples"].([]string); !ok || len(samples) != 1 || !strings.Contains(samples[0], "link.txt") {
		t.Fatalf("expected skipped_scan_samples to contain link.txt, got %#v", meta["skipped_scan_samples"])
	}
}

func TestGlobTool_BlocksSymlinkFileSwapOutsideSandbox(t *testing.T) {
	root := t.TempDir()
	outside := t.TempDir()

	if err := os.WriteFile(filepath.Join(root, "inside.txt"), []byte("inside\n"), 0o644); err != nil {
		t.Fatalf("write inside file: %v", err)
	}
	swapPath := filepath.Join(root, "swap.txt")
	if err := os.WriteFile(swapPath, []byte("before\n"), 0o644); err != nil {
		t.Fatalf("write swap file: %v", err)
	}
	outsideFile := filepath.Join(outside, "secret.txt")
	if err := os.WriteFile(outsideFile, []byte("top secret\n"), 0o644); err != nil {
		t.Fatalf("write outside file: %v", err)
	}
	probe := filepath.Join(root, "probe-link")
	if err := os.Symlink(outsideFile, probe); err != nil {
		t.Skipf("symlink not supported: %v", err)
	}
	if err := os.Remove(probe); err != nil {
		t.Fatalf("remove symlink probe: %v", err)
	}

	s, err := New(root)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	deps := tools.NewContainer()
	tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })

	installGlobHooksForTest(t)
	var once sync.Once
	var hookErr error
	globStatFile = func(path string) (os.FileInfo, error) {
		info, err := os.Lstat(path)
		if err != nil {
			return nil, err
		}
		if filepath.Clean(path) == filepath.Clean(swapPath) {
			once.Do(func() {
				if err := os.Remove(swapPath); err != nil {
					hookErr = err
					return
				}
				if err := os.Symlink(outsideFile, swapPath); err != nil {
					hookErr = err
				}
			})
		}
		return info, nil
	}

	ctx := tools.WithToolResultMetadata(context.Background())
	res, err := globTool().Execute(ctx, `{"pattern":"*.txt"}`, deps)
	if hookErr != nil {
		t.Fatalf("hook error: %v", hookErr)
	}
	if err != nil {
		t.Fatalf("execute: %v", err)
	}
	out := res.PlainText()
	if !strings.Contains(out, "inside.txt") {
		t.Fatalf("expected inside file in output, got %q", out)
	}
	if !strings.Contains(out, "[WARN] glob skipped 1 matched path(s) that are symbolic links") {
		t.Fatalf("expected symlink warning in output, got %q", out)
	}
	if !strings.Contains(out, "swap.txt") {
		t.Fatalf("expected swapped path in warning, got %q", out)
	}

	meta := tools.ToolResultMetadataSnapshot(ctx)
	if meta == nil {
		t.Fatal("expected metadata")
	}
	if got, ok := meta["count"].(int); !ok || got != 1 {
		t.Fatalf("expected count=1, got %#v", meta["count"])
	}
	if got, ok := meta["skipped_symlink_paths"].(int); !ok || got != 1 {
		t.Fatalf("expected skipped_symlink_paths=1, got %#v", meta["skipped_symlink_paths"])
	}
	if samples, ok := meta["skipped_symlink_samples"].([]string); !ok || len(samples) != 1 || samples[0] != "swap.txt" {
		t.Fatalf("expected skipped_symlink_samples [swap.txt], got %#v", meta["skipped_symlink_samples"])
	}
}

func TestGrepTool_BlocksSymlinkFileSwapOutsideSandbox(t *testing.T) {
	root := t.TempDir()
	outside := t.TempDir()

	if err := os.WriteFile(filepath.Join(root, "inside.txt"), []byte("inside\n"), 0o644); err != nil {
		t.Fatalf("write inside file: %v", err)
	}
	swapPath := filepath.Join(root, "swap.txt")
	if err := os.WriteFile(swapPath, []byte("before\n"), 0o644); err != nil {
		t.Fatalf("write swap file: %v", err)
	}
	outsideFile := filepath.Join(outside, "secret.txt")
	if err := os.WriteFile(outsideFile, []byte("TOP_SECRET_SWAP\n"), 0o644); err != nil {
		t.Fatalf("write outside file: %v", err)
	}
	probe := filepath.Join(root, "probe-link")
	if err := os.Symlink(outsideFile, probe); err != nil {
		t.Skipf("symlink not supported: %v", err)
	}
	if err := os.Remove(probe); err != nil {
		t.Fatalf("remove symlink probe: %v", err)
	}

	s, err := New(root)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	deps := tools.NewContainer()
	tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })

	installGrepHooksForTest(t)
	origOpen := grepOpenFile
	var once sync.Once
	var hookErr error
	grepOpenFile = func(path string) (*os.File, error) {
		if filepath.Clean(path) == filepath.Clean(swapPath) {
			once.Do(func() {
				if err := os.Remove(swapPath); err != nil {
					hookErr = err
					return
				}
				if err := os.Symlink(outsideFile, swapPath); err != nil {
					hookErr = err
				}
			})
		}
		return origOpen(path)
	}

	ctx := tools.WithToolResultMetadata(context.Background())
	res, err := grepTool().Execute(ctx, `{"pattern":"TOP_SECRET_SWAP"}`, deps)
	if hookErr != nil {
		t.Fatalf("hook error: %v", hookErr)
	}
	if err != nil {
		t.Fatalf("execute: %v", err)
	}
	out := res.PlainText()
	if !strings.Contains(out, "No matches for: TOP_SECRET_SWAP") {
		t.Fatalf("expected no matches output, got %q", out)
	}
	if !strings.Contains(out, "[WARN] grep skipped 1 path(s) due scan errors") {
		t.Fatalf("expected scan warning in output, got %q", out)
	}
	if !strings.Contains(out, "swap.txt") {
		t.Fatalf("expected swapped path in warning, got %q", out)
	}

	meta := tools.ToolResultMetadataSnapshot(ctx)
	if meta == nil {
		t.Fatal("expected metadata")
	}
	if reason, _ := meta["skipped_reason"].(string); reason != "scan_error" {
		t.Fatalf("expected skipped_reason scan_error, got %#v", meta["skipped_reason"])
	}
	if got, ok := meta["skipped_scan_errors"].(int); !ok || got != 1 {
		t.Fatalf("expected skipped_scan_errors=1, got %#v", meta["skipped_scan_errors"])
	}
	if samples, ok := meta["skipped_scan_samples"].([]string); !ok || len(samples) != 1 || !strings.Contains(samples[0], "swap.txt") {
		t.Fatalf("expected skipped_scan_samples to contain swap.txt, got %#v", meta["skipped_scan_samples"])
	}
}

func TestBashTool_BlocksDefaultWorkingDirSymlinkSwapAfterResolve(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("shell test requires POSIX shell command")
	}
	root := t.TempDir()
	outside := t.TempDir()

	insideDir := filepath.Join(root, "inside")
	if err := os.MkdirAll(insideDir, 0o755); err != nil {
		t.Fatalf("mkdir inside: %v", err)
	}
	link := filepath.Join(root, "swap")
	if err := os.Symlink(insideDir, link); err != nil {
		t.Skipf("symlink not supported: %v", err)
	}

	s, err := New(root)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	s.WorkingDir = link

	expectedAbs := filepath.Clean(link)
	var once sync.Once
	var hookErr error
	useSandboxPathRevalidateHook(t, func(abs string) {
		if filepath.Clean(abs) != expectedAbs {
			return
		}
		once.Do(func() {
			if err := os.Remove(link); err != nil {
				hookErr = err
				return
			}
			if err := os.Symlink(outside, link); err != nil {
				hookErr = err
			}
		})
	})

	deps := tools.NewContainer()
	tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })
	tools.Provide(deps, ConfirmKey, func(context.Context) (Confirmer, error) { return allowConfirmer{}, nil })

	res, err := bashTool().Execute(context.Background(), `{"command":"pwd","timeout":1}`, deps)
	if hookErr != nil {
		t.Fatalf("hook error: %v", hookErr)
	}
	if err == nil {
		t.Fatalf("expected security error")
	}
	var secErr *SecurityError
	if !errors.As(err, &secErr) {
		t.Fatalf("expected SecurityError, got %v", err)
	}
	if !strings.Contains(strings.ToLower(res.PlainText()), "security error") {
		t.Fatalf("expected security diagnostic, got %q", res.PlainText())
	}
}

func TestReadToolRejectsBinaryFile(t *testing.T) {
	root := t.TempDir()
	s, err := New(root)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	path := filepath.Join(root, "binary.dat")
	if err := os.WriteFile(path, []byte{0x00, 0x01, 0x02}, 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}

	deps := tools.NewContainer()
	tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })

	res, err := readTool().Execute(context.Background(), `{"file_path":"binary.dat"}`, deps)
	if err == nil {
		t.Fatalf("expected binary error")
	}
	if !strings.Contains(err.Error(), "binary") {
		t.Fatalf("expected binary error, got %v", err)
	}
	if !strings.Contains(res.PlainText(), "Cannot read binary file") {
		t.Fatalf("expected binary message, got %q", res.PlainText())
	}
}

func TestReadToolAllowsUTF16WithBOM(t *testing.T) {
	root := t.TempDir()
	s, err := New(root)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	path := filepath.Join(root, "utf16.txt")
	// UTF-16LE BOM + "hi\n".
	if err := os.WriteFile(path, []byte{0xFF, 0xFE, 0x68, 0x00, 0x69, 0x00, 0x0A, 0x00}, 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}

	deps := tools.NewContainer()
	tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })

	if _, err := readTool().Execute(context.Background(), `{"file_path":"utf16.txt"}`, deps); err != nil {
		t.Fatalf("expected UTF-16 text file not to be rejected as binary, got %v", err)
	}
}

func TestSplitLinesKeepsTrailingEmptyLine(t *testing.T) {
	got := splitLines("alpha\n")
	if len(got) != 2 {
		t.Fatalf("expected 2 lines including trailing empty line, got %d", len(got))
	}
	if got[0] != "alpha" || got[1] != "" {
		t.Fatalf("unexpected split result: %#v", got)
	}

	if got := splitLines(""); len(got) != 0 {
		t.Fatalf("expected empty input to produce zero lines, got %#v", got)
	}
}

func TestFullReplaceDiffShowsTrailingNewlineChange(t *testing.T) {
	diff := fullReplaceDiff("file.txt", "alpha", "alpha\n")
	if !strings.Contains(diff, "@@ -1,1 +1,2 @@") {
		t.Fatalf("expected trailing newline to affect hunk lengths, got:\n%s", diff)
	}
	if !strings.Contains(diff, "+alpha\n+\n") {
		t.Fatalf("expected trailing empty line in diff preview, got:\n%s", diff)
	}
}

func TestReadToolTruncatesLongLine(t *testing.T) {
	root := t.TempDir()
	s, err := New(root)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	path := filepath.Join(root, "long.txt")
	line := strings.Repeat("a", maxReadLineChars+10)
	if err := os.WriteFile(path, []byte(line+"\n"), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}

	deps := tools.NewContainer()
	tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })

	res, err := readTool().Execute(context.Background(), `{"file_path":"long.txt","limit":1}`, deps)
	if err != nil {
		t.Fatalf("execute: %v", err)
	}
	out := res.PlainText()
	if strings.Contains(out, line) {
		t.Fatalf("expected line to be truncated")
	}
	want := line[:maxReadLineChars-3] + "..."
	if !strings.Contains(out, want) {
		t.Fatalf("expected truncated line, got %q", out)
	}
}

func TestReadToolTruncatesOutputBytes(t *testing.T) {
	root := t.TempDir()
	s, err := New(root)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	path := filepath.Join(root, "many.txt")
	line := strings.Repeat("x", 200)
	var b strings.Builder
	b.Grow((len(line) + 1) * 2000)
	for i := 0; i < 2000; i++ {
		b.WriteString(line)
		b.WriteString("\n")
	}
	if err := os.WriteFile(path, []byte(b.String()), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}

	deps := tools.NewContainer()
	tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })

	res, err := readTool().Execute(context.Background(), `{"file_path":"many.txt","limit":2000}`, deps)
	if err != nil {
		t.Fatalf("execute: %v", err)
	}
	out := res.PlainText()
	if !strings.Contains(out, fmt.Sprintf("output truncated after %d bytes", maxReadOutputBytes)) {
		t.Fatalf("expected truncation notice, got %q", out)
	}
	if !strings.Contains(out, "offset=") {
		t.Fatalf("expected offset guidance, got %q", out)
	}
}

func TestGrepToolLineNumbersDefault(t *testing.T) {
	root := t.TempDir()
	s, err := New(root)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	path := filepath.Join(root, "file.txt")
	if err := os.WriteFile(path, []byte("alpha\nbeta\n"), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}

	deps := tools.NewContainer()
	tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })

	tool := grepTool()
	res, err := tool.Execute(context.Background(), `{"pattern":"alpha"}`, deps)
	if err != nil {
		t.Fatalf("execute default: %v", err)
	}
	out := res.PlainText()
	if !strings.Contains(out, "file.txt:1: alpha") {
		t.Fatalf("expected default to include line numbers, got %q", out)
	}

	res, err = tool.Execute(context.Background(), `{"pattern":"alpha","line_numbers":false}`, deps)
	if err != nil {
		t.Fatalf("execute disabled: %v", err)
	}
	out = res.PlainText()
	if strings.Contains(out, "file.txt:1:") {
		t.Fatalf("expected line numbers disabled, got %q", out)
	}
	if !strings.Contains(out, "file.txt: alpha") {
		t.Fatalf("expected output without line numbers, got %q", out)
	}
}

func TestGlobToolExternalPathsAreUsable(t *testing.T) {
	root := t.TempDir()
	ext := t.TempDir()
	path := filepath.Join(ext, "file.txt")
	if err := os.WriteFile(path, []byte("hello\n"), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}
	s, err := New(root)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	if _, added, err := s.AllowExternalDirectory(ext); err != nil {
		t.Fatalf("allow external: %v", err)
	} else if !added {
		t.Fatalf("expected external directory to be added")
	}
	deps := tools.NewContainer()
	tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })

	tool := globTool()
	res, err := tool.Execute(context.Background(), fmt.Sprintf(`{"pattern":"*.txt","path":%q}`, ext), deps)
	if err != nil {
		t.Fatalf("execute: %v", err)
	}
	out := res.PlainText()
	want := filepath.ToSlash(path)
	if !strings.Contains(out, want) {
		t.Fatalf("expected output to contain %q, got %q", want, out)
	}
	if strings.Contains(out, "../") || strings.Contains(out, `..\\`) {
		t.Fatalf("unexpected parent traversal in output: %q", out)
	}
}

func TestGlobToolTruncationIncludesWarning(t *testing.T) {
	root := t.TempDir()
	s, err := New(root)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	for i := 0; i < 60; i++ {
		path := filepath.Join(root, fmt.Sprintf("file-%03d.txt", i))
		if err := os.WriteFile(path, []byte("hello\n"), 0o644); err != nil {
			t.Fatalf("write: %v", err)
		}
	}

	deps := tools.NewContainer()
	tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })

	tool := globTool()
	res, err := tool.Execute(context.Background(), `{"pattern":"*.txt"}`, deps)
	if err != nil {
		t.Fatalf("execute: %v", err)
	}
	out := res.PlainText()
	if !strings.Contains(out, "Found 60 file(s)") {
		t.Fatalf("expected total count in output, got %q", out)
	}
	if !strings.Contains(out, "Showing first 50") {
		t.Fatalf("expected truncation warning, got %q", out)
	}
	lines := strings.Split(out, "\n")
	if got := len(lines) - 1; got != 50 {
		t.Fatalf("expected 50 files listed, got %d", got)
	}
}

func TestGlobTool_StopsAccumulatingAfterScanCapWithActionableWarning(t *testing.T) {
	root := t.TempDir()
	s, err := New(root)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	for i := 0; i < maxGlobScanFiles+40; i++ {
		path := filepath.Join(root, fmt.Sprintf("file-%03d.txt", i))
		if err := os.WriteFile(path, []byte("hello\n"), 0o644); err != nil {
			t.Fatalf("write: %v", err)
		}
	}

	deps := tools.NewContainer()
	tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })

	ctx := tools.WithToolResultMetadata(context.Background())
	res, err := globTool().Execute(ctx, `{"pattern":"*.txt"}`, deps)
	if err != nil {
		t.Fatalf("execute: %v", err)
	}
	out := res.PlainText()
	if !strings.Contains(out, "[WARN] glob scan stopped after") {
		t.Fatalf("expected scan-cap warning, got %q", out)
	}
	if !strings.Contains(out, "scan cap") {
		t.Fatalf("expected scan-cap context in output, got %q", out)
	}
	idx := strings.LastIndex(out, "[WARN]")
	if idx < 0 {
		t.Fatalf("expected warning line in output, got %q", out)
	}
	assertSandboxSeverityActionWarningDiagnostic(t, out[idx:])

	meta := tools.ToolResultMetadataSnapshot(ctx)
	if meta == nil {
		t.Fatalf("expected metadata")
	}
	if got, ok := meta["scan_truncated"].(bool); !ok || !got {
		t.Fatalf("expected scan_truncated=true, got %#v", meta["scan_truncated"])
	}
	if got, ok := meta["scan_cap"].(int); !ok || got != maxGlobScanFiles {
		t.Fatalf("expected scan_cap=%d, got %#v", maxGlobScanFiles, meta["scan_cap"])
	}
	if got, ok := meta["scanned_files"].(int); !ok || got != maxGlobScanFiles {
		t.Fatalf("expected scanned_files=%d, got %#v", maxGlobScanFiles, meta["scanned_files"])
	}
	if got, ok := meta["skipped_due_to_cap"].(bool); !ok || !got {
		t.Fatalf("expected skipped_due_to_cap=true, got %#v", meta["skipped_due_to_cap"])
	}
	if kind, _ := meta["warning_kind"].(string); kind != scanCapWarningKind {
		t.Fatalf("expected warning_kind %q, got %#v", scanCapWarningKind, meta["warning_kind"])
	}
	if got, ok := meta["count"].(int); !ok || got != maxGlobScanFiles {
		t.Fatalf("expected count=%d after scan cap, got %#v", maxGlobScanFiles, meta["count"])
	}
}

func TestGlobTool_ScanCapStopsTraversalOnSparseOrNoMatchesWithActionableWarning(t *testing.T) {
	root := t.TempDir()
	s, err := New(root)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	for i := 0; i < maxGlobScanFiles+40; i++ {
		path := filepath.Join(root, fmt.Sprintf("file-%03d.txt", i))
		if err := os.WriteFile(path, []byte("hello\n"), 0o644); err != nil {
			t.Fatalf("write: %v", err)
		}
	}

	deps := tools.NewContainer()
	tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })

	ctx := tools.WithToolResultMetadata(context.Background())
	res, err := globTool().Execute(ctx, `{"pattern":"*.nomatch"}`, deps)
	if err != nil {
		t.Fatalf("execute: %v", err)
	}
	out := res.PlainText()
	if !strings.Contains(out, "No files match pattern: *.nomatch") {
		t.Fatalf("expected no-match body, got %q", out)
	}
	if !strings.Contains(out, "[WARN] glob scan stopped after") {
		t.Fatalf("expected scan-cap warning, got %q", out)
	}
	if !strings.Contains(out, "file candidate(s) at cap") {
		t.Fatalf("expected candidate-scan warning context, got %q", out)
	}
	idx := strings.LastIndex(out, "[WARN]")
	if idx < 0 {
		t.Fatalf("expected warning line in output, got %q", out)
	}
	assertSandboxSeverityActionWarningDiagnostic(t, out[idx:])

	meta := tools.ToolResultMetadataSnapshot(ctx)
	if meta == nil {
		t.Fatalf("expected metadata")
	}
	if got, ok := meta["scan_truncated"].(bool); !ok || !got {
		t.Fatalf("expected scan_truncated=true, got %#v", meta["scan_truncated"])
	}
	if got, ok := meta["scan_cap"].(int); !ok || got != maxGlobScanFiles {
		t.Fatalf("expected scan_cap=%d, got %#v", maxGlobScanFiles, meta["scan_cap"])
	}
	if got, ok := meta["scanned_candidates"].(int); !ok || got != maxGlobScanFiles {
		t.Fatalf("expected scanned_candidates=%d, got %#v", maxGlobScanFiles, meta["scanned_candidates"])
	}
	if got, ok := meta["scanned_files"].(int); !ok || got != maxGlobScanFiles {
		t.Fatalf("expected scanned_files=%d, got %#v", maxGlobScanFiles, meta["scanned_files"])
	}
	if got, ok := meta["count"].(int); !ok || got != 0 {
		t.Fatalf("expected count=0 for sparse no-match scan cap, got %#v", meta["count"])
	}
	if kind, _ := meta["warning_kind"].(string); kind != scanCapWarningKind {
		t.Fatalf("expected warning_kind %q, got %#v", scanCapWarningKind, meta["warning_kind"])
	}
}

func TestGlobToolWarnsOnMatchedPathStatFailures(t *testing.T) {
	root := t.TempDir()
	s, err := New(root)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	if err := os.WriteFile(filepath.Join(root, "visible.txt"), []byte("visible\n"), 0o644); err != nil {
		t.Fatalf("write visible: %v", err)
	}
	if err := os.WriteFile(filepath.Join(root, "flaky.txt"), []byte("flaky\n"), 0o644); err != nil {
		t.Fatalf("write flaky: %v", err)
	}

	deps := tools.NewContainer()
	tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })

	installGlobHooksForTest(t)
	globStatFile = func(path string) (os.FileInfo, error) {
		if filepath.Base(path) == "flaky.txt" {
			return nil, errors.New("simulated stat failure")
		}
		return os.Stat(path)
	}

	ctx := tools.WithToolResultMetadata(context.Background())
	res, err := globTool().Execute(ctx, `{"pattern":"*.txt"}`, deps)
	if err != nil {
		t.Fatalf("execute: %v", err)
	}
	out := res.PlainText()
	if !strings.Contains(out, "visible.txt") {
		t.Fatalf("expected visible match in output, got %q", out)
	}
	if !strings.Contains(out, "[WARN] glob skipped 1 matched path(s) due stat errors") {
		t.Fatalf("expected stat warning, got %q", out)
	}
	if !strings.Contains(out, "flaky.txt") {
		t.Fatalf("expected warning to include flaky sample path, got %q", out)
	}

	meta := tools.ToolResultMetadataSnapshot(ctx)
	if meta == nil {
		t.Fatalf("expected metadata")
	}
	if got, ok := meta["has_errors"].(bool); !ok || !got {
		t.Fatalf("expected has_errors=true in metadata, got %#v", meta["has_errors"])
	}
	if got, ok := meta["skipped_paths"].(int); !ok || got != 1 {
		t.Fatalf("expected skipped_paths=1 in metadata, got %#v", meta["skipped_paths"])
	}
	if samples, ok := meta["skipped_path_samples"].([]string); !ok || len(samples) != 1 || !strings.Contains(samples[0], "flaky.txt") {
		t.Fatalf("expected skipped_path_samples to include flaky.txt, got %#v", meta["skipped_path_samples"])
	}
}

func TestGlobToolNoMatchIncludesWarningWhenAllMatchesStatFail(t *testing.T) {
	root := t.TempDir()
	s, err := New(root)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	if err := os.WriteFile(filepath.Join(root, "a.txt"), []byte("a\n"), 0o644); err != nil {
		t.Fatalf("write a: %v", err)
	}
	if err := os.WriteFile(filepath.Join(root, "b.txt"), []byte("b\n"), 0o644); err != nil {
		t.Fatalf("write b: %v", err)
	}

	deps := tools.NewContainer()
	tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })

	installGlobHooksForTest(t)
	globStatFile = func(path string) (os.FileInfo, error) {
		if strings.HasSuffix(path, ".txt") {
			return nil, errors.New("simulated stat failure")
		}
		return os.Stat(path)
	}

	ctx := tools.WithToolResultMetadata(context.Background())
	res, err := globTool().Execute(ctx, `{"pattern":"*.txt"}`, deps)
	if err != nil {
		t.Fatalf("execute: %v", err)
	}
	out := res.PlainText()
	if !strings.Contains(out, "No files match pattern: *.txt") {
		t.Fatalf("expected no-match message, got %q", out)
	}
	if !strings.Contains(out, "[WARN] glob skipped 2 matched path(s) due stat errors") {
		t.Fatalf("expected skipped-path warning, got %q", out)
	}

	meta := tools.ToolResultMetadataSnapshot(ctx)
	if meta == nil {
		t.Fatalf("expected metadata")
	}
	if got, ok := meta["has_errors"].(bool); !ok || !got {
		t.Fatalf("expected has_errors=true in metadata, got %#v", meta["has_errors"])
	}
	if got, ok := meta["skipped_paths"].(int); !ok || got != 2 {
		t.Fatalf("expected skipped_paths=2 in metadata, got %#v", meta["skipped_paths"])
	}
}

func TestGlobTool_PartialFailureWarningUsesSeverityActionAndMetadata(t *testing.T) {
	root := t.TempDir()
	s, err := New(root)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	if err := os.WriteFile(filepath.Join(root, "visible.txt"), []byte("visible\n"), 0o644); err != nil {
		t.Fatalf("write visible: %v", err)
	}
	if err := os.WriteFile(filepath.Join(root, "flaky.txt"), []byte("flaky\n"), 0o644); err != nil {
		t.Fatalf("write flaky: %v", err)
	}

	deps := tools.NewContainer()
	tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })

	installGlobHooksForTest(t)
	globStatFile = func(path string) (os.FileInfo, error) {
		if filepath.Base(path) == "flaky.txt" {
			return nil, errors.New("simulated stat failure")
		}
		return os.Stat(path)
	}

	ctx := tools.WithToolResultMetadata(context.Background())
	res, err := globTool().Execute(ctx, `{"pattern":"*.txt"}`, deps)
	if err != nil {
		t.Fatalf("execute: %v", err)
	}
	out := res.PlainText()
	if !strings.Contains(out, "visible.txt") {
		t.Fatalf("expected visible match in output, got %q", out)
	}
	if !strings.Contains(out, "[WARN] glob skipped 1 matched path(s) due stat errors") {
		t.Fatalf("expected glob warning, got %q", out)
	}
	warningLine := out[strings.LastIndex(out, "[WARN]"):]
	assertSandboxSeverityActionWarningDiagnostic(t, warningLine)

	meta := tools.ToolResultMetadataSnapshot(ctx)
	if meta == nil {
		t.Fatal("expected metadata")
	}
	if kind, _ := meta["warning_kind"].(string); kind != "partial_scan_failure" {
		t.Fatalf("expected warning_kind partial_scan_failure, got %#v", meta["warning_kind"])
	}
	if got, ok := meta["skipped_count"].(int); !ok || got != 1 {
		t.Fatalf("expected skipped_count=1, got %#v", meta["skipped_count"])
	}
	if reason, _ := meta["skipped_reason"].(string); reason != "stat_error" {
		t.Fatalf("expected skipped_reason stat_error, got %#v", meta["skipped_reason"])
	}
}

func TestGrepToolExternalPathsRespectGlob(t *testing.T) {
	root := t.TempDir()
	ext := t.TempDir()
	path := filepath.Join(ext, "file.txt")
	if err := os.WriteFile(path, []byte("hello\n"), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}
	s, err := New(root)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	if _, added, err := s.AllowExternalDirectory(ext); err != nil {
		t.Fatalf("allow external: %v", err)
	} else if !added {
		t.Fatalf("expected external directory to be added")
	}
	deps := tools.NewContainer()
	tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })

	tool := grepTool()
	res, err := tool.Execute(context.Background(), fmt.Sprintf(`{"pattern":"hello","path":%q,"glob":"*.txt"}`, ext), deps)
	if err != nil {
		t.Fatalf("execute: %v", err)
	}
	out := res.PlainText()
	if strings.Contains(out, "No matches") {
		t.Fatalf("expected matches, got %q", out)
	}
	want := filepath.ToSlash(path)
	if !strings.Contains(out, want) {
		t.Fatalf("expected output to contain %q, got %q", want, out)
	}
	if strings.Contains(out, "../") || strings.Contains(out, `..\\`) {
		t.Fatalf("unexpected parent traversal in output: %q", out)
	}
}

func TestGrepToolWarnsOnPermissionDeniedWalkEntries(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("permission-denied walk behavior differs on Windows")
	}

	root := t.TempDir()
	s, err := New(root)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	if err := os.WriteFile(filepath.Join(root, "visible.txt"), []byte("needle\n"), 0o644); err != nil {
		t.Fatalf("write visible: %v", err)
	}
	blocked := filepath.Join(root, "blocked")
	if err := os.MkdirAll(blocked, 0o755); err != nil {
		t.Fatalf("mkdir blocked: %v", err)
	}
	if err := os.WriteFile(filepath.Join(blocked, "hidden.txt"), []byte("needle\n"), 0o644); err != nil {
		t.Fatalf("write hidden: %v", err)
	}
	if err := os.Chmod(blocked, 0o000); err != nil {
		t.Skipf("chmod not supported: %v", err)
	}
	defer func() { _ = os.Chmod(blocked, 0o755) }()

	deps := tools.NewContainer()
	tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })

	res, err := grepTool().Execute(context.Background(), `{"pattern":"needle"}`, deps)
	if err != nil {
		t.Fatalf("execute: %v", err)
	}
	out := res.PlainText()
	if !strings.Contains(out, "visible.txt") {
		t.Fatalf("expected accessible match in output, got %q", out)
	}
	if !strings.Contains(out, "[WARN] grep skipped") {
		t.Skip("permission-denied walk entries not reproducible in this environment")
	}
	if !strings.Contains(out, "blocked") {
		t.Fatalf("expected blocked path in warning, got %q", out)
	}
}

func TestGrepToolWarnsOnOpenReadSeekFailures(t *testing.T) {
	root := t.TempDir()
	s, err := New(root)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	if err := os.WriteFile(filepath.Join(root, "visible.txt"), []byte("needle\n"), 0o644); err != nil {
		t.Fatalf("write visible: %v", err)
	}
	if err := os.WriteFile(filepath.Join(root, "openfail.txt"), []byte("needle\n"), 0o644); err != nil {
		t.Fatalf("write openfail: %v", err)
	}
	if err := os.WriteFile(filepath.Join(root, "readfail.txt"), []byte("needle\n"), 0o644); err != nil {
		t.Fatalf("write readfail: %v", err)
	}
	if err := os.WriteFile(filepath.Join(root, "seekfail.txt"), []byte("needle\n"), 0o644); err != nil {
		t.Fatalf("write seekfail: %v", err)
	}

	installGrepHooksForTest(t)
	origOpen := grepOpenFile
	grepOpenFile = func(path string) (*os.File, error) {
		if strings.HasSuffix(path, "openfail.txt") {
			return nil, errors.New("open boom")
		}
		return origOpen(path)
	}
	origReadSample := grepReadSample
	grepReadSample = func(f *os.File, n int) ([]byte, error) {
		if strings.HasSuffix(f.Name(), "readfail.txt") {
			return nil, errors.New("read boom")
		}
		return origReadSample(f, n)
	}
	origSeek := grepSeekStart
	grepSeekStart = func(f *os.File) error {
		if strings.HasSuffix(f.Name(), "seekfail.txt") {
			return errors.New("seek boom")
		}
		return origSeek(f)
	}

	deps := tools.NewContainer()
	tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })

	res, err := grepTool().Execute(context.Background(), `{"pattern":"needle"}`, deps)
	if err != nil {
		t.Fatalf("execute: %v", err)
	}
	out := res.PlainText()
	if !strings.Contains(out, "visible.txt") {
		t.Fatalf("expected visible match in output, got %q", out)
	}
	if !strings.Contains(out, "[WARN] grep skipped 3 path(s) due scan errors") {
		t.Fatalf("expected scan warning count, got %q", out)
	}
	if !strings.Contains(out, "openfail.txt (open: open boom)") {
		t.Fatalf("expected open failure details, got %q", out)
	}
	if !strings.Contains(out, "readfail.txt (read: read boom)") {
		t.Fatalf("expected read failure details, got %q", out)
	}
	if !strings.Contains(out, "seekfail.txt (seek: seek boom)") {
		t.Fatalf("expected seek failure details, got %q", out)
	}
}

func TestGrepToolWarnsOnTerminalWalkError(t *testing.T) {
	root := t.TempDir()
	s, err := New(root)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	if err := os.WriteFile(filepath.Join(root, "visible.txt"), []byte("needle\n"), 0o644); err != nil {
		t.Fatalf("write visible: %v", err)
	}

	installGrepHooksForTest(t)
	grepWalkDirFn = func(root string, fn fs.WalkDirFunc) error {
		return errors.New("walk boom")
	}

	deps := tools.NewContainer()
	tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })

	res, err := grepTool().Execute(context.Background(), `{"pattern":"needle"}`, deps)
	if err != nil {
		t.Fatalf("execute: %v", err)
	}
	out := res.PlainText()
	if !strings.Contains(out, "No matches for: needle") {
		t.Fatalf("expected no-matches body, got %q", out)
	}
	if !strings.Contains(out, "[WARN] grep skipped 1 path(s) due scan errors") {
		t.Fatalf("expected scan warning count, got %q", out)
	}
	if !strings.Contains(out, "walk: walk boom") {
		t.Fatalf("expected walk error details, got %q", out)
	}
}

func TestGrepTool_PartialFailureWarningUsesSeverityActionAndMetadata(t *testing.T) {
	root := t.TempDir()
	s, err := New(root)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	if err := os.WriteFile(filepath.Join(root, "visible.txt"), []byte("needle\n"), 0o644); err != nil {
		t.Fatalf("write visible: %v", err)
	}
	if err := os.WriteFile(filepath.Join(root, "openfail.txt"), []byte("needle\n"), 0o644); err != nil {
		t.Fatalf("write openfail: %v", err)
	}

	installGrepHooksForTest(t)
	origOpen := grepOpenFile
	grepOpenFile = func(path string) (*os.File, error) {
		if strings.HasSuffix(path, "openfail.txt") {
			return nil, errors.New("open boom")
		}
		return origOpen(path)
	}

	deps := tools.NewContainer()
	tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })

	ctx := tools.WithToolResultMetadata(context.Background())
	res, err := grepTool().Execute(ctx, `{"pattern":"needle"}`, deps)
	if err != nil {
		t.Fatalf("execute: %v", err)
	}
	out := res.PlainText()
	if !strings.Contains(out, "visible.txt") {
		t.Fatalf("expected visible match in output, got %q", out)
	}
	if !strings.Contains(out, "[WARN] grep skipped 1 path(s) due scan errors") {
		t.Fatalf("expected grep scan warning, got %q", out)
	}
	warningLine := out[strings.LastIndex(out, "[WARN]"):]
	assertSandboxSeverityActionWarningDiagnostic(t, warningLine)

	meta := tools.ToolResultMetadataSnapshot(ctx)
	if meta == nil {
		t.Fatal("expected metadata")
	}
	if kind, _ := meta["warning_kind"].(string); kind != "partial_scan_failure" {
		t.Fatalf("expected warning_kind partial_scan_failure, got %#v", meta["warning_kind"])
	}
	if got, ok := meta["skipped_count"].(int); !ok || got != 1 {
		t.Fatalf("expected skipped_count=1, got %#v", meta["skipped_count"])
	}
	if reason, _ := meta["skipped_reason"].(string); reason != "scan_error" {
		t.Fatalf("expected skipped_reason scan_error, got %#v", meta["skipped_reason"])
	}
	if got, ok := meta["skipped_scan_errors"].(int); !ok || got != 1 {
		t.Fatalf("expected skipped_scan_errors=1, got %#v", meta["skipped_scan_errors"])
	}
	if samples, ok := meta["skipped_scan_samples"].([]string); !ok || len(samples) != 1 || !strings.Contains(samples[0], "openfail.txt") {
		t.Fatalf("expected skipped_scan_samples to include openfail.txt, got %#v", meta["skipped_scan_samples"])
	}
}

func TestGrepTool_StopsAccumulatingAfterScanCapWithActionableWarning(t *testing.T) {
	root := t.TempDir()
	s, err := New(root)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	for i := 0; i < 20; i++ {
		path := filepath.Join(root, fmt.Sprintf("file-%02d.txt", i))
		if err := os.WriteFile(path, []byte("needle\n"), 0o644); err != nil {
			t.Fatalf("write: %v", err)
		}
	}

	deps := tools.NewContainer()
	tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })

	for _, mode := range []string{"files_with_matches", "count"} {
		t.Run(mode, func(t *testing.T) {
			ctx := tools.WithToolResultMetadata(context.Background())
			res, err := grepTool().Execute(ctx, fmt.Sprintf(`{"pattern":"needle","output_mode":%q,"max_results":5}`, mode), deps)
			if err != nil {
				t.Fatalf("execute: %v", err)
			}

			out := res.PlainText()
			if !strings.Contains(out, "[WARN] grep scan stopped after") {
				t.Fatalf("expected scan-cap warning, got %q", out)
			}
			idx := strings.LastIndex(out, "[WARN]")
			if idx < 0 {
				t.Fatalf("expected warning line, got %q", out)
			}
			assertSandboxSeverityActionWarningDiagnostic(t, out[idx:])

			meta := tools.ToolResultMetadataSnapshot(ctx)
			if meta == nil {
				t.Fatalf("expected metadata")
			}
			if got, ok := meta["scan_truncated"].(bool); !ok || !got {
				t.Fatalf("expected scan_truncated=true, got %#v", meta["scan_truncated"])
			}
			if got, ok := meta["scan_cap"].(int); !ok || got != 5 {
				t.Fatalf("expected scan_cap=5, got %#v", meta["scan_cap"])
			}
			if got, ok := meta["scanned_files"].(int); !ok || got != 5 {
				t.Fatalf("expected scanned_files=5, got %#v", meta["scanned_files"])
			}
			if got, ok := meta["scanned_candidates"].(int); !ok || got != 5 {
				t.Fatalf("expected scanned_candidates=5, got %#v", meta["scanned_candidates"])
			}
			if got, ok := meta["skipped_due_to_cap"].(bool); !ok || !got {
				t.Fatalf("expected skipped_due_to_cap=true, got %#v", meta["skipped_due_to_cap"])
			}
			if kind, _ := meta["warning_kind"].(string); kind != scanCapWarningKind {
				t.Fatalf("expected warning_kind %q, got %#v", scanCapWarningKind, meta["warning_kind"])
			}
		})
	}
}

func TestGrepTool_ScanCapStopsTraversalOnSparseOrNoMatchesWithActionableWarning(t *testing.T) {
	root := t.TempDir()
	s, err := New(root)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	for i := 0; i < 20; i++ {
		path := filepath.Join(root, fmt.Sprintf("file-%02d.txt", i))
		if err := os.WriteFile(path, []byte("haystack\n"), 0o644); err != nil {
			t.Fatalf("write: %v", err)
		}
	}

	deps := tools.NewContainer()
	tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })

	ctx := tools.WithToolResultMetadata(context.Background())
	res, err := grepTool().Execute(ctx, `{"pattern":"needle","output_mode":"files_with_matches","max_results":5}`, deps)
	if err != nil {
		t.Fatalf("execute: %v", err)
	}

	out := res.PlainText()
	if !strings.Contains(out, "No matches for: needle") {
		t.Fatalf("expected no-match body, got %q", out)
	}
	if !strings.Contains(out, "[WARN] grep scan stopped after") {
		t.Fatalf("expected scan-cap warning, got %q", out)
	}
	if !strings.Contains(out, "file candidate(s) at cap 5") {
		t.Fatalf("expected candidate-scan warning context, got %q", out)
	}
	idx := strings.LastIndex(out, "[WARN]")
	if idx < 0 {
		t.Fatalf("expected warning line in output, got %q", out)
	}
	assertSandboxSeverityActionWarningDiagnostic(t, out[idx:])

	meta := tools.ToolResultMetadataSnapshot(ctx)
	if meta == nil {
		t.Fatalf("expected metadata")
	}
	if got, ok := meta["scan_truncated"].(bool); !ok || !got {
		t.Fatalf("expected scan_truncated=true, got %#v", meta["scan_truncated"])
	}
	if got, ok := meta["scan_cap"].(int); !ok || got != 5 {
		t.Fatalf("expected scan_cap=5, got %#v", meta["scan_cap"])
	}
	if got, ok := meta["scanned_candidates"].(int); !ok || got != 5 {
		t.Fatalf("expected scanned_candidates=5, got %#v", meta["scanned_candidates"])
	}
	if got, ok := meta["scanned_files"].(int); !ok || got != 5 {
		t.Fatalf("expected scanned_files=5, got %#v", meta["scanned_files"])
	}
	if got, ok := meta["skipped_due_to_cap"].(bool); !ok || !got {
		t.Fatalf("expected skipped_due_to_cap=true, got %#v", meta["skipped_due_to_cap"])
	}
	if kind, _ := meta["warning_kind"].(string); kind != scanCapWarningKind {
		t.Fatalf("expected warning_kind %q, got %#v", scanCapWarningKind, meta["warning_kind"])
	}
}

func TestApplyUpdateFilePreservesMode(t *testing.T) {
	root := t.TempDir()
	s, err := New(root)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	path := filepath.Join(root, "script.sh")
	if err := os.WriteFile(path, []byte("hello\n"), 0o755); err != nil {
		t.Fatalf("write: %v", err)
	}
	if runtime.GOOS != "windows" {
		if st, err := os.Stat(path); err != nil {
			t.Fatalf("stat: %v", err)
		} else if st.Mode().Perm() != 0o755 {
			t.Skipf("filesystem does not preserve exec perms (got %o)", st.Mode().Perm())
		}
	}
	if err := applyUpdateFile(s, "script.sh", []patchHunk{{lines: []string{"-hello", "+hi"}}}); err != nil {
		t.Fatalf("applyUpdateFile: %v", err)
	}
	st, err := os.Stat(path)
	if err != nil {
		t.Fatalf("stat: %v", err)
	}
	if st.Mode().Perm() != 0o755 {
		t.Fatalf("expected mode 0755, got %o", st.Mode().Perm())
	}
}

func TestApplyPatchTargetsLineNumberedHunk(t *testing.T) {
	root := t.TempDir()
	s, err := New(root)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	path := filepath.Join(root, "file.txt")
	content := "alpha\nbeta\nalpha\nbeta\n"
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}
	patch := "*** Begin Patch\n" +
		"*** Update File: file.txt\n" +
		"@@ -3,2 +3,2 @@\n" +
		"-alpha\n" +
		"-beta\n" +
		"+alpha\n" +
		"+gamma\n" +
		"*** End Patch\n"
	if _, err := applyPatchToSandbox(s, patch); err != nil {
		t.Fatalf("applyPatchToSandbox: %v", err)
	}
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	want := "alpha\nbeta\nalpha\ngamma\n"
	if string(b) != want {
		t.Fatalf("unexpected content:\n%s", string(b))
	}
}

func TestApplyPatchAcceptsBlankContextLineWithoutPrefix(t *testing.T) {
	root := t.TempDir()
	s, err := New(root)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	path := filepath.Join(root, "file.txt")
	content := "alpha\n\nbeta\n"
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}
	patch := "*** Begin Patch\n" +
		"*** Update File: file.txt\n" +
		"@@ -1,3 +1,3 @@\n" +
		" alpha\n" +
		"\n" +
		"-beta\n" +
		"+gamma\n" +
		"*** End Patch\n"
	if _, err := applyPatchToSandbox(s, patch); err != nil {
		t.Fatalf("applyPatchToSandbox: %v", err)
	}
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	want := "alpha\n\ngamma\n"
	if string(b) != want {
		t.Fatalf("unexpected content:\n%s", string(b))
	}
}

func TestApplyPatchRejectsAmbiguousHunk(t *testing.T) {
	root := t.TempDir()
	s, err := New(root)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	path := filepath.Join(root, "file.txt")
	content := "alpha\nbeta\nalpha\nbeta\n"
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}
	patch := "*** Begin Patch\n" +
		"*** Update File: file.txt\n" +
		"@@\n" +
		"-alpha\n" +
		"-beta\n" +
		"+alpha\n" +
		"+gamma\n" +
		"*** End Patch\n"
	if _, err := applyPatchToSandbox(s, patch); err == nil {
		t.Fatalf("expected ambiguous hunk error")
	} else if !strings.Contains(err.Error(), "ambiguous") {
		t.Fatalf("expected ambiguous error, got %v", err)
	}
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if string(b) != content {
		t.Fatalf("expected content unchanged")
	}
}

func TestApplyPatchReportsHunkIndex(t *testing.T) {
	root := t.TempDir()
	s, err := New(root)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	path := filepath.Join(root, "file.txt")
	content := "alpha\nbeta\ngamma\n"
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}
	patch := "*** Begin Patch\n" +
		"*** Update File: file.txt\n" +
		"@@\n" +
		"-beta\n" +
		"+bravo\n" +
		"@@\n" +
		"-missing\n" +
		"+found\n" +
		"*** End Patch\n"
	if _, err := applyPatchToSandbox(s, patch); err == nil {
		t.Fatalf("expected hunk error")
	} else {
		if !strings.Contains(err.Error(), "hunk 2") {
			t.Fatalf("expected hunk index in error, got %v", err)
		}
		if !strings.Contains(err.Error(), "context not found") {
			t.Fatalf("expected context not found error, got %v", err)
		}
	}
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if string(b) != content {
		t.Fatalf("expected content unchanged")
	}
}

func TestApplyPatchRejectsMalformedPatch(t *testing.T) {
	root := t.TempDir()
	s, err := New(root)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	patch := "*** Begin Patch\n" +
		"this is not a patch op\n" +
		"*** End Patch\n"
	if _, err := applyPatchToSandbox(s, patch); err == nil {
		t.Fatalf("expected malformed patch error")
	} else if !strings.Contains(err.Error(), "unexpected patch content") {
		t.Fatalf("expected malformed patch diagnostic, got %v", err)
	}
}

func TestApplyPatchRejectsUpdateWithoutHunks(t *testing.T) {
	root := t.TempDir()
	s, err := New(root)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	path := filepath.Join(root, "file.txt")
	original := "before\n"
	if err := os.WriteFile(path, []byte(original), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}
	patch := "*** Begin Patch\n" +
		"*** Update File: file.txt\n" +
		"*** End Patch\n"
	if _, err := applyPatchToSandbox(s, patch); err == nil {
		t.Fatalf("expected update-without-hunks error")
	} else if !strings.Contains(err.Error(), "must include at least one hunk") {
		t.Fatalf("expected update-without-hunks diagnostic, got %v", err)
	}
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if string(got) != original {
		t.Fatalf("expected file unchanged, got %q", string(got))
	}
}

func TestApplyPatchRejectsPatchWithNoOperations(t *testing.T) {
	root := t.TempDir()
	s, err := New(root)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	patch := "*** Begin Patch\n*** End Patch\n"
	if _, err := applyPatchToSandbox(s, patch); err == nil {
		t.Fatalf("expected no-operations error")
	} else if !strings.Contains(err.Error(), "contains no operations") {
		t.Fatalf("expected no-operations diagnostic, got %v", err)
	}
}

func TestApplyPatchTool_ErrorDiagnosticSeverityActionFormat(t *testing.T) {
	root := t.TempDir()
	s, err := New(root)
	if err != nil {
		t.Fatalf("new: %v", err)
	}

	t.Run("confirmer error", func(t *testing.T) {
		deps := tools.NewContainer()
		tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })

		patch := "*** Begin Patch\n*** End Patch\n"
		out, err := applyPatchTool().Execute(context.Background(), fmt.Sprintf(`{"patch":%q}`, patch), deps)
		if err == nil {
			t.Fatalf("expected confirmer error")
		}
		assertSandboxSeverityActionDiagnostic(t, out.PlainText())
	})

	t.Run("execution error", func(t *testing.T) {
		deps := tools.NewContainer()
		tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })
		tools.Provide(deps, ConfirmKey, func(context.Context) (Confirmer, error) { return allowConfirmer{}, nil })

		patch := "*** Begin Patch\nthis is not a patch op\n*** End Patch\n"
		out, err := applyPatchTool().Execute(context.Background(), fmt.Sprintf(`{"patch":%q}`, patch), deps)
		if err == nil {
			t.Fatalf("expected apply_patch execution error")
		}
		assertSandboxSeverityActionDiagnostic(t, out.PlainText())
	})
}

func TestApplyPatch_RejectsLargeTargetFile(t *testing.T) {
	root := t.TempDir()
	s, err := New(root)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	path := filepath.Join(root, "big.txt")
	f, err := os.Create(path)
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	if err := f.Truncate(maxEditFileBytes + 1); err != nil {
		_ = f.Close()
		t.Fatalf("truncate: %v", err)
	}
	if _, err := f.WriteAt([]byte("hello"), 0); err != nil {
		_ = f.Close()
		t.Fatalf("write: %v", err)
	}
	if err := f.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}

	useSandboxReadAllHook(t, func(io.Reader) ([]byte, error) {
		return nil, errors.New("unexpected full read")
	})

	patch := "*** Begin Patch\n" +
		"*** Update File: big.txt\n" +
		"@@\n" +
		"-hello\n" +
		"+hi\n" +
		"*** End Patch\n"
	if _, err := applyPatchToSandbox(s, patch); err == nil {
		t.Fatalf("expected size error")
	} else {
		if !strings.Contains(err.Error(), "apply_patch refuses to load") {
			t.Fatalf("expected apply_patch size diagnostic, got %v", err)
		}
		if !strings.Contains(err.Error(), "max") {
			t.Fatalf("expected size limit context, got %v", err)
		}
		if strings.Contains(err.Error(), "unexpected full read") {
			t.Fatalf("expected size check before read, got %v", err)
		}
	}
}
