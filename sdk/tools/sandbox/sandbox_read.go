package sandbox

import (
	"bufio"
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"unicode/utf16"
	"unicode/utf8"

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
		raw, err := io.ReadAll(io.LimitReader(f, maxReadFileBytes+1))
		if err != nil {
			msg := formatErrorDiagnosticFromErr("Unable to read file", err, "Check file permissions and retry.")
			return llm.TextContent(msg), err
		}
		if int64(len(raw)) > maxReadFileBytes {
			msg := fmt.Sprintf("[ERROR] read refuses to load %s (%d bytes) - max %d bytes; use bash or split the file", a.FilePath, len(raw), maxReadFileBytes)
			return llm.TextContent(msg), fmt.Errorf("file too large")
		}
		decoded, err := decodeUnicodeTextBOM(raw)
		if err != nil {
			msg := formatErrorDiagnosticFromErr("Unable to decode Unicode text file", err, "Convert the file to valid UTF-8/UTF-16/UTF-32 text and retry.")
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

		scanner := bufio.NewScanner(bytes.NewReader(decoded))
		scanner.Buffer(make([]byte, 0, 64*1024), maxReadDecodedBytes)

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

const maxReadDecodedBytes = int(maxReadFileBytes * 2)

type unicodeTextEncoding struct {
	name      string
	bomBytes  int
	unitBytes int
	order     binary.ByteOrder
}

func detectUnicodeTextEncoding(raw []byte) (unicodeTextEncoding, bool) {
	switch {
	case len(raw) >= 4 && bytes.Equal(raw[:4], []byte{0x00, 0x00, 0xFE, 0xFF}):
		return unicodeTextEncoding{name: "UTF-32BE", bomBytes: 4, unitBytes: 4, order: binary.BigEndian}, true
	case len(raw) >= 4 && bytes.Equal(raw[:4], []byte{0xFF, 0xFE, 0x00, 0x00}):
		return unicodeTextEncoding{name: "UTF-32LE", bomBytes: 4, unitBytes: 4, order: binary.LittleEndian}, true
	case len(raw) >= 3 && bytes.Equal(raw[:3], []byte{0xEF, 0xBB, 0xBF}):
		return unicodeTextEncoding{name: "UTF-8", bomBytes: 3, unitBytes: 1}, true
	case len(raw) >= 2 && bytes.Equal(raw[:2], []byte{0xFE, 0xFF}):
		return unicodeTextEncoding{name: "UTF-16BE", bomBytes: 2, unitBytes: 2, order: binary.BigEndian}, true
	case len(raw) >= 2 && bytes.Equal(raw[:2], []byte{0xFF, 0xFE}):
		return unicodeTextEncoding{name: "UTF-16LE", bomBytes: 2, unitBytes: 2, order: binary.LittleEndian}, true
	default:
		return unicodeTextEncoding{}, false
	}
}

func decodeUnicodeTextBOM(raw []byte) ([]byte, error) {
	encoding, ok := detectUnicodeTextEncoding(raw)
	if !ok {
		return raw, nil
	}
	payload := raw[encoding.bomBytes:]
	if len(payload)%encoding.unitBytes != 0 {
		return nil, fmt.Errorf("malformed %s: truncated code unit", encoding.name)
	}
	if encoding.unitBytes == 1 {
		if !utf8.Valid(payload) {
			return nil, fmt.Errorf("malformed UTF-8 after BOM")
		}
		return append([]byte(nil), payload...), nil
	}
	out := make([]byte, 0, minInt(len(payload)*2, maxReadDecodedBytes))
	appendRune := func(r rune) error {
		if r < 0 || r > utf8.MaxRune || (r >= 0xD800 && r <= 0xDFFF) {
			return fmt.Errorf("malformed %s: invalid Unicode code point U+%X", encoding.name, r)
		}
		if len(out)+utf8.RuneLen(r) > maxReadDecodedBytes {
			return fmt.Errorf("decoded %s output exceeds %d bytes", encoding.name, maxReadDecodedBytes)
		}
		out = utf8.AppendRune(out, r)
		return nil
	}
	if encoding.unitBytes == 4 {
		for i := 0; i < len(payload); i += 4 {
			value := encoding.order.Uint32(payload[i : i+4])
			if value > utf8.MaxRune || (value >= 0xD800 && value <= 0xDFFF) {
				return nil, fmt.Errorf("malformed %s at byte %d: invalid Unicode code point U+%X", encoding.name, encoding.bomBytes+i, value)
			}
			if err := appendRune(rune(value)); err != nil {
				return nil, err
			}
		}
		return out, nil
	}
	for i := 0; i < len(payload); i += 2 {
		first := encoding.order.Uint16(payload[i : i+2])
		switch {
		case first >= 0xD800 && first <= 0xDBFF:
			if i+4 > len(payload) {
				return nil, fmt.Errorf("malformed %s at byte %d: unpaired high surrogate", encoding.name, encoding.bomBytes+i)
			}
			second := encoding.order.Uint16(payload[i+2 : i+4])
			if second < 0xDC00 || second > 0xDFFF {
				return nil, fmt.Errorf("malformed %s at byte %d: high surrogate is not followed by a low surrogate", encoding.name, encoding.bomBytes+i)
			}
			if err := appendRune(utf16.DecodeRune(rune(first), rune(second))); err != nil {
				return nil, err
			}
			i += 2
		case first >= 0xDC00 && first <= 0xDFFF:
			return nil, fmt.Errorf("malformed %s at byte %d: unpaired low surrogate", encoding.name, encoding.bomBytes+i)
		default:
			if err := appendRune(rune(first)); err != nil {
				return nil, err
			}
		}
	}
	return out, nil
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
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
