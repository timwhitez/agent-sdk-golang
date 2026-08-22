from pathlib import Path

read = Path("sdk/tools/sandbox/sandbox_read.go")
text = read.read_text()
anchor = '''type readArgs struct {
\tFilePath string `json:"file_path"`
\tOffset   int    `json:"offset,omitempty"` // 1-based line offset
\tLimit    int    `json:"limit,omitempty"`  // number of lines
}
'''
replacement = '''type readArgs struct {
\tFilePath string `json:"file_path"`
\tOffset   int    `json:"offset,omitempty"` // 1-based line offset
\tLimit    int    `json:"limit,omitempty"`  // number of lines
}

const (
\tmaxReadLineOffset = 10_000_000
\tmaxReadLineLimit  = 10_000
\treadLineLimitDefault = 2_000
)

func validateReadNumericArgs(args readArgs) error {
\tif args.Offset > maxReadLineOffset {
\t\treturn fmt.Errorf("read offset %d exceeds maximum %d", args.Offset, maxReadLineOffset)
\t}
\tif args.Limit > maxReadLineLimit {
\t\treturn fmt.Errorf("read limit %d exceeds maximum %d", args.Limit, maxReadLineLimit)
\t}
\treturn nil
}
'''
if text.count(anchor) != 1:
    raise SystemExit(f"read args anchor count={text.count(anchor)}")
text = text.replace(anchor, replacement)
old = '''\t\ts, err := tools.Get(deps, ctx, Key)
\t\tif err != nil {
\t\t\treturn llm.TextContent(""), err
\t\t}
'''
new = '''\t\tif err := validateReadNumericArgs(a); err != nil {
\t\t\tmsg := formatErrorDiagnosticFromErr("Invalid read range", err, fmt.Sprintf("Use offset <= %d and limit <= %d, then retry.", maxReadLineOffset, maxReadLineLimit))
\t\t\treturn llm.TextContent(msg), err
\t\t}
\t\ts, err := tools.Get(deps, ctx, Key)
\t\tif err != nil {
\t\t\treturn llm.TextContent(""), err
\t\t}
'''
if text.count(old) < 1:
    raise SystemExit("read handler anchor missing")
text = text.replace(old, new, 1)
text = text.replace('''\t\tlimit := a.Limit
\t\tif limit <= 0 {
\t\t\tlimit = 2000
\t\t}
''', '''\t\tlimit := a.Limit
\t\tif limit <= 0 {
\t\t\tlimit = readLineLimitDefault
\t\t}
''', 1)
# Avoid even bounded model input directly controlling a large allocation.
text = text.replace('''\t\tout := make([]string, 0, limit)
''', '''\t\tinitialCapacity := limit
\t\tif initialCapacity > 256 {
\t\t\tinitialCapacity = 256
\t\t}
\t\tout := make([]string, 0, initialCapacity)
''', 1)
read.write_text(text)

grep = Path("sdk/tools/sandbox/sandbox_grep.go")
text = grep.read_text()
anchor = '''type grepArgs struct {
\tPattern string `json:"pattern"`
\tPath    string `json:"path,omitempty"`

\tGlob        string `json:"glob,omitempty"` // e.g. "*.go" or "**/*.ts"
\tIgnoreCase  bool   `json:"ignore_case,omitempty"`
\tBefore      int    `json:"before,omitempty"`
\tAfter       int    `json:"after,omitempty"`
\tContext     int    `json:"context,omitempty"`
\tMaxResults  int    `json:"max_results,omitempty"`  // output lines or entries, default 50
\tOutputMode  string `json:"output_mode,omitempty"`  // "content"|"files_with_matches"|"count"
\tLineNumbers *bool  `json:"line_numbers,omitempty"` // default true
}
'''
replacement = '''type grepArgs struct {
\tPattern string `json:"pattern"`
\tPath    string `json:"path,omitempty"`

\tGlob        string `json:"glob,omitempty"` // e.g. "*.go" or "**/*.ts"
\tIgnoreCase  bool   `json:"ignore_case,omitempty"`
\tBefore      int    `json:"before,omitempty"`
\tAfter       int    `json:"after,omitempty"`
\tContext     int    `json:"context,omitempty"`
\tMaxResults  int    `json:"max_results,omitempty"`  // output lines or entries, default 50
\tOutputMode  string `json:"output_mode,omitempty"`  // "content"|"files_with_matches"|"count"
\tLineNumbers *bool  `json:"line_numbers,omitempty"` // default true
}

const (
\tmaxGrepContextLines = 1_000
\tmaxGrepResults      = 10_000
\tgrepResultsDefault  = 50
)

func validateGrepNumericArgs(args grepArgs) error {
\tfor name, value := range map[string]int{
\t\t"before":  args.Before,
\t\t"after":   args.After,
\t\t"context": args.Context,
\t} {
\t\tif value > maxGrepContextLines {
\t\t\treturn fmt.Errorf("grep %s %d exceeds maximum %d", name, value, maxGrepContextLines)
\t\t}
\t}
\tif args.MaxResults > maxGrepResults {
\t\treturn fmt.Errorf("grep max_results %d exceeds maximum %d", args.MaxResults, maxGrepResults)
\t}
\treturn nil
}
'''
if text.count(anchor) != 1:
    raise SystemExit(f"grep args anchor count={text.count(anchor)}")
text = text.replace(anchor, replacement)
old = '''\t\ts, err := tools.Get(deps, ctx, Key)
\t\tif err != nil {
\t\t\treturn "", err
\t\t}
'''
new = '''\t\tif err := validateGrepNumericArgs(a); err != nil {
\t\t\treturn "", fmt.Errorf("invalid grep limits: %w; use context/before/after <= %d and max_results <= %d", err, maxGrepContextLines, maxGrepResults)
\t\t}
\t\ts, err := tools.Get(deps, ctx, Key)
\t\tif err != nil {
\t\t\treturn "", err
\t\t}
'''
if text.count(old) < 1:
    raise SystemExit("grep handler anchor missing")
text = text.replace(old, new, 1)
text = text.replace('''\t\tmaxOut := a.MaxResults
\t\tif maxOut <= 0 {
\t\t\tmaxOut = 50
\t\t}
''', '''\t\tmaxOut := a.MaxResults
\t\tif maxOut <= 0 {
\t\t\tmaxOut = grepResultsDefault
\t\t}
''', 1)
text = text.replace('''\t\t\tprev := make([]prevLine, 0, before)
''', '''\t\t\tinitialContextCapacity := before
\t\t\tif initialContextCapacity > 64 {
\t\t\t\tinitialContextCapacity = 64
\t\t\t}
\t\t\tprev := make([]prevLine, 0, initialContextCapacity)
''', 1)
grep.write_text(text)

Path("sdk/tools/sandbox/sandbox_numeric_limits_test.go").write_text(r'''package sandbox

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
''')
