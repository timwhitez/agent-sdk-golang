package sandbox

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
