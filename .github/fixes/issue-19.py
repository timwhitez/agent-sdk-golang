from pathlib import Path

path = Path("sdk/tools/sandbox/sandbox_read.go")
text = path.read_text()
old_import = '''\t"context"
\t"errors"
\t"fmt"
\t"io"
'''
new_import = '''\t"context"
\t"encoding/binary"
\t"errors"
\t"fmt"
\t"io"
'''
if text.count(old_import) != 1:
    raise SystemExit(f"import anchor count={text.count(old_import)}")
text = text.replace(old_import, new_import)
old_import2 = '''\t"path/filepath"
\t"strings"
'''
new_import2 = '''\t"path/filepath"
\t"strings"
\t"unicode/utf16"
\t"unicode/utf8"
'''
if text.count(old_import2) != 1:
    raise SystemExit(f"unicode import anchor count={text.count(old_import2)}")
text = text.replace(old_import2, new_import2)

old = '''\t\tif _, err := f.Seek(0, io.SeekStart); err != nil {
\t\t\tmsg := formatErrorDiagnosticFromErr("Unable to reset file cursor", err, "Retry the read request.")
\t\t\treturn llm.TextContent(msg), err
\t\t}

\t\toffset := a.Offset
'''
new = '''\t\tif _, err := f.Seek(0, io.SeekStart); err != nil {
\t\t\tmsg := formatErrorDiagnosticFromErr("Unable to reset file cursor", err, "Retry the read request.")
\t\t\treturn llm.TextContent(msg), err
\t\t}
\t\traw, err := io.ReadAll(io.LimitReader(f, maxReadFileBytes+1))
\t\tif err != nil {
\t\t\tmsg := formatErrorDiagnosticFromErr("Unable to read file", err, "Check file permissions and retry.")
\t\t\treturn llm.TextContent(msg), err
\t\t}
\t\tif int64(len(raw)) > maxReadFileBytes {
\t\t\tmsg := fmt.Sprintf("[ERROR] read refuses to load %s (%d bytes) - max %d bytes; use bash or split the file", a.FilePath, len(raw), maxReadFileBytes)
\t\t\treturn llm.TextContent(msg), fmt.Errorf("file too large")
\t\t}
\t\tdecoded, err := decodeUnicodeTextBOM(raw)
\t\tif err != nil {
\t\t\tmsg := formatErrorDiagnosticFromErr("Unable to decode Unicode text file", err, "Convert the file to valid UTF-8/UTF-16/UTF-32 text and retry.")
\t\t\treturn llm.TextContent(msg), err
\t\t}

\t\toffset := a.Offset
'''
if text.count(old) != 1:
    raise SystemExit(f"read body anchor count={text.count(old)}")
text = text.replace(old, new)
text = text.replace('''\t\tscanner := bufio.NewScanner(f)
\t\tscanner.Buffer(make([]byte, 0, 64*1024), int(maxReadFileBytes))
''', '''\t\tscanner := bufio.NewScanner(bytes.NewReader(decoded))
\t\tscanner.Buffer(make([]byte, 0, 64*1024), maxReadDecodedBytes)
''', 1)

anchor = '''// readSampleBytes reads up to n bytes from a file.
func readSampleBytes(f *os.File, n int) ([]byte, error) {
'''
insert = '''const maxReadDecodedBytes = int(maxReadFileBytes * 2)

type unicodeTextEncoding struct {
\tname      string
\tbomBytes  int
\tunitBytes int
\torder     binary.ByteOrder
}

func detectUnicodeTextEncoding(raw []byte) (unicodeTextEncoding, bool) {
\tswitch {
\tcase len(raw) >= 4 && bytes.Equal(raw[:4], []byte{0x00, 0x00, 0xFE, 0xFF}):
\t\treturn unicodeTextEncoding{name: "UTF-32BE", bomBytes: 4, unitBytes: 4, order: binary.BigEndian}, true
\tcase len(raw) >= 4 && bytes.Equal(raw[:4], []byte{0xFF, 0xFE, 0x00, 0x00}):
\t\treturn unicodeTextEncoding{name: "UTF-32LE", bomBytes: 4, unitBytes: 4, order: binary.LittleEndian}, true
\tcase len(raw) >= 3 && bytes.Equal(raw[:3], []byte{0xEF, 0xBB, 0xBF}):
\t\treturn unicodeTextEncoding{name: "UTF-8", bomBytes: 3, unitBytes: 1}, true
\tcase len(raw) >= 2 && bytes.Equal(raw[:2], []byte{0xFE, 0xFF}):
\t\treturn unicodeTextEncoding{name: "UTF-16BE", bomBytes: 2, unitBytes: 2, order: binary.BigEndian}, true
\tcase len(raw) >= 2 && bytes.Equal(raw[:2], []byte{0xFF, 0xFE}):
\t\treturn unicodeTextEncoding{name: "UTF-16LE", bomBytes: 2, unitBytes: 2, order: binary.LittleEndian}, true
\tdefault:
\t\treturn unicodeTextEncoding{}, false
\t}
}

func decodeUnicodeTextBOM(raw []byte) ([]byte, error) {
\tencoding, ok := detectUnicodeTextEncoding(raw)
\tif !ok {
\t\treturn raw, nil
\t}
\tpayload := raw[encoding.bomBytes:]
\tif len(payload)%encoding.unitBytes != 0 {
\t\treturn nil, fmt.Errorf("malformed %s: truncated code unit", encoding.name)
\t}
\tif encoding.unitBytes == 1 {
\t\tif !utf8.Valid(payload) {
\t\t\treturn nil, fmt.Errorf("malformed UTF-8 after BOM")
\t\t}
\t\treturn append([]byte(nil), payload...), nil
\t}
\tout := make([]byte, 0, minInt(len(payload)*2, maxReadDecodedBytes))
\tappendRune := func(r rune) error {
\t\tif r < 0 || r > utf8.MaxRune || (r >= 0xD800 && r <= 0xDFFF) {
\t\t\treturn fmt.Errorf("malformed %s: invalid Unicode code point U+%X", encoding.name, r)
\t\t}
\t\tif len(out)+utf8.RuneLen(r) > maxReadDecodedBytes {
\t\t\treturn fmt.Errorf("decoded %s output exceeds %d bytes", encoding.name, maxReadDecodedBytes)
\t\t}
\t\tout = utf8.AppendRune(out, r)
\t\treturn nil
\t}
\tif encoding.unitBytes == 4 {
\t\tfor i := 0; i < len(payload); i += 4 {
\t\t\tvalue := encoding.order.Uint32(payload[i : i+4])
\t\t\tif value > utf8.MaxRune || (value >= 0xD800 && value <= 0xDFFF) {
\t\t\t\treturn nil, fmt.Errorf("malformed %s at byte %d: invalid Unicode code point U+%X", encoding.name, encoding.bomBytes+i, value)
\t\t\t}
\t\t\tif err := appendRune(rune(value)); err != nil {
\t\t\t\treturn nil, err
\t\t\t}
\t\t}
\t\treturn out, nil
\t}
\tfor i := 0; i < len(payload); i += 2 {
\t\tfirst := encoding.order.Uint16(payload[i : i+2])
\t\tswitch {
\t\tcase first >= 0xD800 && first <= 0xDBFF:
\t\t\tif i+4 > len(payload) {
\t\t\t\treturn nil, fmt.Errorf("malformed %s at byte %d: unpaired high surrogate", encoding.name, encoding.bomBytes+i)
\t\t\t}
\t\t\tsecond := encoding.order.Uint16(payload[i+2 : i+4])
\t\t\tif second < 0xDC00 || second > 0xDFFF {
\t\t\t\treturn nil, fmt.Errorf("malformed %s at byte %d: high surrogate is not followed by a low surrogate", encoding.name, encoding.bomBytes+i)
\t\t\t}
\t\t\tif err := appendRune(utf16.DecodeRune(rune(first), rune(second))); err != nil {
\t\t\t\treturn nil, err
\t\t\t}
\t\t\ti += 2
\t\tcase first >= 0xDC00 && first <= 0xDFFF:
\t\t\treturn nil, fmt.Errorf("malformed %s at byte %d: unpaired low surrogate", encoding.name, encoding.bomBytes+i)
\t\tdefault:
\t\t\tif err := appendRune(rune(first)); err != nil {
\t\t\t\treturn nil, err
\t\t\t}
\t\t}
\t}
\treturn out, nil
}

func minInt(a, b int) int {
\tif a < b {
\t\treturn a
\t}
\treturn b
}

// readSampleBytes reads up to n bytes from a file.
func readSampleBytes(f *os.File, n int) ([]byte, error) {
'''
if text.count(anchor) != 1:
    raise SystemExit(f"decode helper anchor count={text.count(anchor)}")
path.write_text(text.replace(anchor, insert))

Path("sdk/tools/sandbox/sandbox_read_unicode_test.go").write_text(r'''package sandbox

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"unicode/utf16"

	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

func encodeUTF16Test(order binary.ByteOrder, text string) []byte {
	bom := []byte{0xFF, 0xFE}
	if order == binary.BigEndian {
		bom = []byte{0xFE, 0xFF}
	}
	units := utf16.Encode([]rune(text))
	out := append([]byte(nil), bom...)
	for _, unit := range units {
		var buf [2]byte
		order.PutUint16(buf[:], unit)
		out = append(out, buf[:]...)
	}
	return out
}

func encodeUTF32Test(order binary.ByteOrder, text string) []byte {
	bom := []byte{0xFF, 0xFE, 0x00, 0x00}
	if order == binary.BigEndian {
		bom = []byte{0x00, 0x00, 0xFE, 0xFF}
	}
	out := append([]byte(nil), bom...)
	for _, r := range text {
		var buf [4]byte
		order.PutUint32(buf[:], uint32(r))
		out = append(out, buf[:]...)
	}
	return out
}

func readUnicodeFixture(t *testing.T, raw []byte, args map[string]any) (string, error) {
	t.Helper()
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "unicode.txt"), raw, 0o644); err != nil {
		t.Fatal(err)
	}
	s, err := New(root)
	if err != nil {
		t.Fatal(err)
	}
	deps := tools.NewContainer()
	tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })
	args["file_path"] = "unicode.txt"
	encoded, _ := json.Marshal(args)
	out, err := readTool().Execute(context.Background(), string(encoded), deps)
	return out.PlainText(), err
}

func TestReadDecodesUnicodeBOMVariants(t *testing.T) {
	fixtures := []struct {
		name string
		raw  []byte
	}{
		{"utf16le", encodeUTF16Test(binary.LittleEndian, "hi\nworld\n")},
		{"utf16be", encodeUTF16Test(binary.BigEndian, "hi\nworld\n")},
		{"utf32le", encodeUTF32Test(binary.LittleEndian, "hi\nworld\n")},
		{"utf32be", encodeUTF32Test(binary.BigEndian, "hi\nworld\n")},
	}
	for _, fixture := range fixtures {
		t.Run(fixture.name, func(t *testing.T) {
			got, err := readUnicodeFixture(t, fixture.raw, map[string]any{})
			if err != nil {
				t.Fatal(err)
			}
			if strings.ContainsRune(got, '\x00') || strings.Contains(got, "\ufeff") {
				t.Fatalf("decoded output still contains BOM/NUL bytes: %q", got)
			}
			if got != "   1  hi\n   2  world" {
				t.Fatalf("decoded output = %q", got)
			}
		})
	}
}

func TestReadUnicodeDecodingPreservesOffsetAndLimit(t *testing.T) {
	got, err := readUnicodeFixture(t, encodeUTF16Test(binary.LittleEndian, "one\ntwo\nthree\n"), map[string]any{"offset": 2, "limit": 1})
	if err != nil {
		t.Fatal(err)
	}
	if got != "   2  two" {
		t.Fatalf("offset/limit output = %q", got)
	}
}

func TestReadRejectsMalformedUnicodeSequences(t *testing.T) {
	fixtures := []struct {
		name string
		raw  []byte
	}{
		{"truncated-utf16", []byte{0xFF, 0xFE, 0x61}},
		{"unpaired-utf16", []byte{0xFF, 0xFE, 0x00, 0xD8}},
		{"truncated-utf32", []byte{0xFF, 0xFE, 0x00, 0x00, 0x61}},
		{"invalid-utf32", []byte{0x00, 0x00, 0xFE, 0xFF, 0x00, 0x11, 0x00, 0x00}},
		{"invalid-utf8", []byte{0xEF, 0xBB, 0xBF, 0xFF}},
	}
	for _, fixture := range fixtures {
		t.Run(fixture.name, func(t *testing.T) {
			got, err := readUnicodeFixture(t, fixture.raw, map[string]any{})
			if err == nil || !strings.Contains(strings.ToLower(err.Error()), "malformed") {
				t.Fatalf("error = %v, output = %q; want malformed encoding error", err, got)
			}
		})
	}
}
''')
