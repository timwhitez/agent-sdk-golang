package sandbox

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

type cancelingBashConfirmer struct {
	cancel context.CancelFunc
	allow  bool
	err    error
	calls  int
}

func (c *cancelingBashConfirmer) Confirm(_ context.Context, action, _ string) (bool, error) {
	if action != "bash" {
		return false, errors.New("unexpected confirmation action")
	}
	c.calls++
	c.cancel()
	return c.allow, c.err
}

func TestBashCancellationDuringConfirmation(t *testing.T) {
	confirmErr := errors.New("confirmation failed")
	for _, tc := range []struct {
		name    string
		allow   bool
		err     error
		wantErr error
	}{
		{"approve", true, nil, context.Canceled},
		{"deny", false, nil, ErrToolDenied},
		{"error", true, confirmErr, confirmErr},
	} {
		t.Run(tc.name, func(t *testing.T) {
			root := t.TempDir()
			sb, err := New(root)
			if err != nil {
				t.Fatal(err)
			}
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			ctx = tools.WithToolResultMetadata(ctx)
			conf := &cancelingBashConfirmer{cancel: cancel, allow: tc.allow, err: tc.err}
			deps := tools.NewContainer()
			tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return sb, nil })
			tools.Provide(deps, ConfirmKey, func(context.Context) (Confirmer, error) { return conf, nil })
			_, err = bashTool().Execute(ctx, `{"command":"echo harmless > cancel-marker.txt"}`, deps)
			if !errors.Is(err, tc.wantErr) || conf.calls != 1 {
				t.Errorf("error=%v calls=%d; want %v, 1", err, conf.calls, tc.wantErr)
			}
			if _, err := os.Stat(filepath.Join(root, "cancel-marker.txt")); !errors.Is(err, os.ErrNotExist) {
				t.Errorf("command created marker after confirmation cancellation: %v", err)
			}
			if tc.name == "approve" {
				meta := tools.ToolResultMetadataSnapshot(ctx)
				if meta["exit_code"] != -1 || meta["timed_out"] != false || meta["output_bytes"] != int64(0) {
					t.Errorf("canceled command metadata=%v", meta)
				}
			}
		})
	}
}
