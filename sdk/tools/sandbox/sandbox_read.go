package sandbox

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

// readArgs holds the arguments for the read tool.
type readArgs struct {
	FilePath string `json:"file_path"`
	Offset   int    `json:"offset,omitempty"` // 1-based line offset
	Limit    int    `json:"limit,omitempty"`  // number of lines
}

const (
	maxReadLineOffset    = 10_000_000
	maxReadLineLimit     = 10_000
	readLineLimitDefault = 2_000
)

func validateReadNumericArgs(args readArgs) error {
	if args.Offset > maxReadLineOffset {
		return fmt.Errorf("read offset %d exceeds maximum %d", args.Offset, maxReadLineOffset)
	}
	if args.Limit > maxReadLineLimit {
		return fmt.Errorf("read limit %d exceeds maximum %d", args.Limit, maxReadLineLimit)
	}
	return nil
}

// readTool returns a tool that reads file contents with line numbers,
// offset/limit support, and binary file detection.
func readTool() tools.Tool {
	return toolWithArgs[readArgs]("read", "Read contents of a file", func(ctx context.Context, a readArgs, deps *tools.Container) (llm.Content, error) {
		if err := validateReadNumericArgs(a); err != nil {
			msg := formatErrorDiagnosticFromErr("Invalid read range", err, fmt.Sprintf("Use offset <= %d and limit <= %d, then retry.", maxReadLineOffset, maxReadLineLimit))
			return llm.TextContent(msg), err
		}
		s, err := tools.Get(deps, ctx, Key)
		if err != nil {
			return llm.TextContent(""), err
		}
		p, err := s.resolveForAccess(a.FilePath)
		if err != nil {
			msg := formatErrorDiagnosticFromErr("Security error", err, "Use a file path inside the sandbox root and retry.")
			return llm.TextContent(msg), err
		}
		f, st, _, err := s.openReadPath(p)
		if err != nil {
			var secErr *SecurityError
			if errors.As(err, &secErr) {
				msg := formatErrorDiagnosticFromErr("Security error", err, "Use a file path inside the sandbox root and retry.")
				return llm.TextContent(msg), err
			}
			if os.IsNotExist(err) {
				msg := formatErrorDiagnostic(fmt.Sprintf("File not found: %s", a.FilePath), "Verify the path exists (use ls/glob) and retry.")
				return llm.TextContent(msg), err
			}
			msg := formatErrorDiagnosticFromErr("Unable to open file", err, "Check file permissions/path and retry.")
			return llm.TextContent(msg), err
		}
		defer f.Close()
		if st.IsDir() {
			err := fmt.Errorf("is a directory")
			msg := formatErrorDiagnostic(fmt.Sprintf("Path is a directory: %s", a.FilePath), "Provide a file path (not a directory) and retry.")
			return llm.TextContent(msg), err
		}
		if st.Size() > maxReadFileBytes {
			msg := fmt.Sprintf("[ERROR] read refuses to load %s (%d bytes) - max %d bytes; use bash or split the file", a.FilePath, st.Size(), maxReadFileBytes)
			return llm.TextContent(msg), fmt.Errorf("file too large")
		}

		sample, err := readSampleBytes(f, binaryDetectSampleBytes)
		if err != nil {
			msg := formatErrorDiagnosticFromErr("Unable to read file sample", err, "Check file permissions and retry.")
			return llm.TextContent(msg), err
		}
		if isBinaryData(a.FilePath, sample) {
			err := fmt.Errorf("binary file")
			msg := formatErrorDiagnostic(fmt.Sprintf("Cannot read binary file: %s", a.FilePath), "Use a text file path or another tool for binary content.")
			return llm.TextContent(msg), err
		}
		if _, err := f.Seek(0, io.SeekStart); err != nil {
			msg := formatErrorDiagnosticFromErr("Unable to reset file cursor", err, "Retry the read request.")
			return llm.TextContent(msg), err
		}

		offset := a.Offset
		if offset <= 0 {
			offset = 1
		}
		limit := a.Limit
		if limit <= 0 {
			limit = readLineLimitDefault
		}

		scanner := bufio.NewScanner(f)
		scanner.Buffer(make([]byte, 0, 64*1024), int(maxReadFileBytes))

		initialCapacity := limit
		if initialCapacity > 256 {
			initialCapacity = 256
		}
		out := make([]string, 0, initialCapacity)
		truncatedOutput := false
		truncatedLines := false
		outputBytes := 0
		lastLine := offset - 1
		processed := 0
		lineNo := 0
		for scanner.Scan() {
			lineNo++
			if lineNo < offset {
				continue
			}
			if processed >= limit {
				break
			}
			line := strings.TrimSuffix(scanner.Text(), "\r")
			if maxReadLineChars > 0 && len(line) > maxReadLineChars {
				line, _ = truncateLine(line, maxReadLineChars)
				truncatedLines = true
			}
			formatted := fmt.Sprintf("%4d  %s", lineNo, line)
			lineBytes := len(formatted)
			if len(out) > 0 {
				lineBytes++
			}
			if maxReadOutputBytes > 0 && outputBytes+lineBytes > maxReadOutputBytes {
				truncatedOutput = true
				break
			}
			out = append(out, formatted)
			outputBytes += lineBytes
			lastLine = lineNo
			processed++
		}
		if err := scanner.Err(); err != nil {
			msg := formatErrorDiagnosticFromErr("Unable to read file", err, "Check file encoding/path and retry.")
			return llm.TextContent(msg), err
		}
		if len(out) == 0 {
			if truncatedOutput {
				nextOffset := lastLine + 1
				note := fmt.Sprintf("... (output truncated after %d bytes; use offset=%d to continue)", maxReadOutputBytes, nextOffset)
				return llm.TextContent(note), nil
			}
			return llm.TextContent("(no content)"), nil
		}
		result := strings.Join(out, "\n")
		if truncatedLines {
			result += fmt.Sprintf("\n... (lines truncated to %d chars)", maxReadLineChars)
		}
		if truncatedOutput {
			nextOffset := lastLine + 1
			result += fmt.Sprintf("\n... (output truncated after %d bytes; use offset=%d to continue)", maxReadOutputBytes, nextOffset)
		}
		return llm.TextContent(result), nil
	})
}

// readSampleBytes reads up to n bytes from a file.
func readSampleBytes(f *os.File, n int) ([]byte, error) {
	if n <= 0 {
		return nil, nil
	}
	buf := make([]byte, n)
	readN, err := f.Read(buf)
	if err != nil && !errors.Is(err, io.EOF) {
		return nil, err
	}
	return buf[:readN], nil
}

// isBinaryData heuristically determines if a byte slice represents binary data.
func isBinaryData(path string, b []byte) bool {
	if len(b) == 0 {
		return false
	}
	n := len(b)
	if n > binaryDetectSampleBytes {
		n = binaryDetectSampleBytes
	}
	sample := b[:n]
	if hasUnicodeTextBOM(sample) {
		return false
	}
	if bytes.IndexByte(sample, 0) >= 0 {
		if hasLikelyTextExtension(path) {
			return false
		}
		return true
	}
	nonPrintable := 0
	for i := 0; i < n; i++ {
		c := sample[i]
		if c < 9 || (c > 13 && c < 32) {
			nonPrintable++
		}
	}
	ratio := float64(nonPrintable) / float64(n)
	if ratio > 0.3 {
		if hasLikelyTextExtension(path) {
			return false
		}
		return true
	}
	return false
}

// hasUnicodeTextBOM checks if a byte slice starts with a Unicode BOM.
func hasUnicodeTextBOM(sample []byte) bool {
	if len(sample) >= 4 {
		if bytes.Equal(sample[:4], []byte{0x00, 0x00, 0xFE, 0xFF}) {
			return true // UTF-32 BE
		}
		if bytes.Equal(sample[:4], []byte{0xFF, 0xFE, 0x00, 0x00}) {
			return true // UTF-32 LE
		}
	}
	if len(sample) >= 3 && bytes.Equal(sample[:3], []byte{0xEF, 0xBB, 0xBF}) {
		return true // UTF-8 BOM
	}
	if len(sample) >= 2 {
		if bytes.Equal(sample[:2], []byte{0xFE, 0xFF}) {
			return true // UTF-16 BE
		}
		if bytes.Equal(sample[:2], []byte{0xFF, 0xFE}) {
			return true // UTF-16 LE
		}
	}
	return false
}

// hasLikelyTextExtension checks if a path has a common text file extension.
func hasLikelyTextExtension(path string) bool {
	ext := strings.ToLower(strings.TrimSpace(filepath.Ext(path)))
	if ext == "" {
		return false
	}
	_, ok := likelyTextExtensions[ext]
	return ok
}

// likelyTextExtensions is a set of file extensions that typically indicate text files.
var likelyTextExtensions = map[string]struct{}{
	".c":         {},
	".cc":        {},
	".cfg":       {},
	".conf":      {},
	".cpp":       {},
	".css":       {},
	".csv":       {},
	".env":       {},
	".gitignore": {},
	".go":        {},
	".h":         {},
	".hpp":       {},
	".html":      {},
	".ini":       {},
	".java":      {},
	".js":        {},
	".json":      {},
	".jsonl":     {},
	".log":       {},
	".md":        {},
	".php":       {},
	".py":        {},
	".rb":        {},
	".rs":        {},
	".sh":        {},
	".sql":       {},
	".svg":       {},
	".toml":      {},
	".ts":        {},
	".tsx":       {},
	".txt":       {},
	".xml":       {},
	".yaml":      {},
	".yml":       {},
	".zsh":       {},
}
