package sandbox_test

import (
	"context"
	"encoding/json"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools/sandbox"
	"os"
	"path/filepath"
	"syscall"
	"testing"
	"time"
)

func TestReadFIFORespectsCancellation(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, "pipe")
	if err := syscall.Mkfifo(path, 0600); err != nil {
		t.Fatal(err)
	}
	s, err := sandbox.New(root)
	if err != nil {
		t.Fatal(err)
	}
	deps := tools.NewContainer()
	tools.Provide(deps, sandbox.Key, func(context.Context) (*sandbox.Sandbox, error) { return s, nil })
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	raw, _ := json.Marshal(map[string]string{"file_path": "pipe"})
	done := make(chan error, 1)
	for _, tool := range sandbox.Tools() {
		if tool.Name == "read" {
			go func() { _, err := tool.Execute(ctx, string(raw), deps); done <- err }()
			break
		}
	}
	select {
	case err := <-done:
		if err == nil {
			t.Fatal("FIFO read unexpectedly succeeded")
		}
		t.Logf("returned safely: %v", err)
		return
	case <-time.After(250 * time.Millisecond):
		t.Errorf("read still blocked in FIFO open after context deadline: %v", ctx.Err())
	}
	// Release the blocked syscall so the regression leaves no goroutine behind.
	writer, err := os.OpenFile(path, os.O_WRONLY|syscall.O_NONBLOCK, 0600)
	if err != nil {
		t.Fatal(err)
	}
	writer.Close()
	select {
	case err := <-done:
		t.Logf("returned only after external FIFO writer opened: %v", err)
	case <-time.After(time.Second):
		t.Fatal("failed to release FIFO reader")
	}
}
