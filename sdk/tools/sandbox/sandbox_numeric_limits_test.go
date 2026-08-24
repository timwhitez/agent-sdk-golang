package sandbox

import (
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

func numericLimitDeps(t *testing.T) *tools.Container {
	t.Helper()
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "x.txt"), []byte("needle\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	s, err := New(root)
	if err != nil {
		t.Fatal(err)
	}
	deps := tools.NewContainer()
	tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })
	return deps
}

func TestReadNumericLimitsRejectBeforeAllocation(t *testing.T) {
	deps := numericLimitDeps(t)
	for _, value := range []int{maxReadLineLimit + 1, math.MaxInt} {
		value := value
		t.Run(fmt.Sprint(value), func(t *testing.T) {
			defer func() {
				if recovered := recover(); recovered != nil {
					t.Fatalf("read panicked for limit %d: %v", value, recovered)
				}
			}()
			_, err := readTool().Execute(context.Background(), fmt.Sprintf(`{"file_path":"x.txt","limit":%d}`, value), deps)
			if err == nil || !strings.Contains(err.Error(), "exceeds maximum") {
				t.Fatalf("read limit error = %v", err)
			}
		})
	}
	if err := validateReadNumericArgs(readArgs{Offset: maxReadLineOffset, Limit: maxReadLineLimit}); err != nil {
		t.Fatalf("maximum read range rejected: %v", err)
	}
	if err := validateReadNumericArgs(readArgs{Offset: maxReadLineOffset + 1}); err == nil {
		t.Fatal("read offset maximum+1 accepted")
	}
}

func TestGrepNumericLimitsRejectBeforeAllocation(t *testing.T) {
	deps := numericLimitDeps(t)
	tests := []string{
		fmt.Sprintf(`{"pattern":"needle","context":%d}`, maxGrepContextLines+1),
		fmt.Sprintf(`{"pattern":"needle","before":%d}`, math.MaxInt),
		fmt.Sprintf(`{"pattern":"needle","after":%d}`, math.MaxInt),
		fmt.Sprintf(`{"pattern":"needle","max_results":%d}`, math.MaxInt),
	}
	for _, args := range tests {
		args := args
		t.Run(args, func(t *testing.T) {
			defer func() {
				if recovered := recover(); recovered != nil {
					t.Fatalf("grep panicked for %s: %v", args, recovered)
				}
			}()
			_, err := grepTool().Execute(context.Background(), args, deps)
			if err == nil || !strings.Contains(err.Error(), "exceeds maximum") {
				t.Fatalf("grep limits error = %v", err)
			}
		})
	}
	if err := validateGrepNumericArgs(grepArgs{Context: maxGrepContextLines, MaxResults: maxGrepResults}); err != nil {
		t.Fatalf("maximum grep limits rejected: %v", err)
	}
}
