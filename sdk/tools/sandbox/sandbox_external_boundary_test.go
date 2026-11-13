package sandbox_test

import (
	"context"
	"encoding/json"
	"errors"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools/sandbox"
	"os"
	"path/filepath"
	"testing"
)

type confirmFunc func(context.Context, string, string) (bool, error)

func (f confirmFunc) Confirm(c context.Context, a, b string) (bool, error) { return f(c, a, b) }
func TestExternalApprovalMustBindConfirmedTarget(t *testing.T) {
	root := t.TempDir()
	approved := filepath.Join(t.TempDir(), "approved")
	victim := t.TempDir()
	if err := os.Mkdir(approved, 0700); err != nil {
		t.Fatal(err)
	}
	s, err := sandbox.New(root)
	if err != nil {
		t.Fatal(err)
	}
	deps := tools.NewContainer()
	tools.Provide(deps, sandbox.Key, func(context.Context) (*sandbox.Sandbox, error) { return s, nil })
	tools.Provide(deps, sandbox.ConfirmKey, func(context.Context) (sandbox.Confirmer, error) {
		return confirmFunc(func(_ context.Context, _ string, detail string) (bool, error) {
			t.Logf("confirmed=%s", detail)
			if err := os.Remove(approved); err != nil {
				t.Fatal(err)
			}
			if err := os.Symlink(victim, approved); err != nil {
				t.Fatal(err)
			}
			return true, nil
		}), nil
	})
	raw, _ := json.Marshal(map[string]string{"path": approved})
	for _, tool := range sandbox.Tools() {
		if tool.Name == "external_directory" {
			result, err := tool.Execute(context.Background(), string(raw), deps)
			t.Logf("err=%v result=%s allowed=%v", err, result.PlainText(), s.AllowedRootsSnapshot())
			if _, err := s.Resolve(filepath.Join(victim, "private.txt")); err == nil {
				t.Fatal("unconfirmed destination was granted sandbox access")
			}
			return
		}
	}
	t.Fatal("tool not found")
}
func TestExternalApprovalCancellationDoesNotGrantAccess(t *testing.T) {
	s, err := sandbox.New(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	external := t.TempDir()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	deps := tools.NewContainer()
	tools.Provide(deps, sandbox.Key, func(context.Context) (*sandbox.Sandbox, error) { return s, nil })
	tools.Provide(deps, sandbox.ConfirmKey, func(context.Context) (sandbox.Confirmer, error) {
		return confirmFunc(func(context.Context, string, string) (bool, error) { cancel(); return true, nil }), nil
	})
	raw, _ := json.Marshal(map[string]string{"path": external})
	for _, tool := range sandbox.Tools() {
		if tool.Name == "external_directory" {
			result, err := tool.Execute(ctx, string(raw), deps)
			t.Logf("err=%v result=%s allowed=%v", err, result.PlainText(), s.AllowedRootsSnapshot())
			if !errors.Is(err, context.Canceled) {
				t.Fatalf("expected cancellation error, got %v", err)
			}
			if len(s.AllowedRootsSnapshot()) != 0 {
				t.Fatal("cancelled call changed sandbox permission state")
			}
			return
		}
	}
	t.Fatal("tool not found")
}

func TestExternalApprovalRejectsReplacedDirectory(t *testing.T) {
	s, err := sandbox.New(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	approved := filepath.Join(t.TempDir(), "approved")
	if err := os.Mkdir(approved, 0700); err != nil {
		t.Fatal(err)
	}
	deps := tools.NewContainer()
	tools.Provide(deps, sandbox.Key, func(context.Context) (*sandbox.Sandbox, error) { return s, nil })
	tools.Provide(deps, sandbox.ConfirmKey, func(context.Context) (sandbox.Confirmer, error) {
		return confirmFunc(func(context.Context, string, string) (bool, error) {
			if err := os.Rename(approved, approved+"-old"); err != nil {
				t.Fatal(err)
			}
			if err := os.Mkdir(approved, 0700); err != nil {
				t.Fatal(err)
			}
			return true, nil
		}), nil
	})
	raw, _ := json.Marshal(map[string]string{"path": approved})
	for _, tool := range sandbox.Tools() {
		if tool.Name == "external_directory" {
			_, err := tool.Execute(context.Background(), string(raw), deps)
			if err == nil || len(s.AllowedRootsSnapshot()) != 0 {
				t.Fatalf("replaced directory granted: err=%v roots=%v", err, s.AllowedRootsSnapshot())
			}
			return
		}
	}
	t.Fatal("tool unavailable")
}
