package sandbox

import (
	"bytes"
	"context"
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

func TestEditPreservesUnicodeBOMEncoding(t *testing.T) {
	encoders := map[string]func(string) []byte{
		"plain":   func(s string) []byte { return []byte(s) },
		"utf8":    func(s string) []byte { return append([]byte{0xef, 0xbb, 0xbf}, []byte(s)...) },
		"utf16le": func(s string) []byte { return encodeUTF16Test(binary.LittleEndian, s) },
		"utf16be": func(s string) []byte { return encodeUTF16Test(binary.BigEndian, s) },
		"utf32le": func(s string) []byte { return encodeUTF32Test(binary.LittleEndian, s) },
		"utf32be": func(s string) []byte { return encodeUTF32Test(binary.BigEndian, s) },
	}
	for _, tc := range []struct {
		tool tools.Tool
		args string
	}{
		{editTool(), `{"file_path":"values.txt","old_string":"1","new_string":"20界😀"}`},
		{multieditTool(), `{"file_path":"values.txt","edits":[{"old_string":"1","new_string":"20界😀"}]}`},
	} {
		for name, encode := range encoders {
			t.Run(tc.tool.Name+"/"+name, func(t *testing.T) {
				root := t.TempDir()
				path := filepath.Join(root, "values.txt")
				if err := os.WriteFile(path, encode("n=1\r\n"), 0600); err != nil {
					t.Fatal(err)
				}
				s, err := New(root)
				if err != nil {
					t.Fatal(err)
				}
				deps := tools.NewContainer()
				tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })
				tools.Provide(deps, ConfirmKey, func(context.Context) (Confirmer, error) { return allowConfirmer{}, nil })
				out, err := tc.tool.Execute(context.Background(), tc.args, deps)
				got, readErr := os.ReadFile(path)
				want := encode("n=20界😀\r\n")
				if err != nil || readErr != nil || !bytes.Equal(got, want) {
					t.Fatalf("err=%v readErr=%v output=%q bytes=%x want=%x", err, readErr, out.PlainText(), got, want)
				}
			})
		}
	}
}

func TestEditRejectsMalformedBOMBeforeConfirmation(t *testing.T) {
	for _, tc := range []struct {
		tool tools.Tool
		args string
	}{
		{editTool(), `{"file_path":"values.txt","old_string":"1","new_string":"20"}`},
		{multieditTool(), `{"file_path":"values.txt","edits":[{"old_string":"1","new_string":"20"}]}`},
	} {
		t.Run(tc.tool.Name, func(t *testing.T) {
			root := t.TempDir()
			path := filepath.Join(root, "values.txt")
			raw := []byte{0xff, 0xfe, '1'}
			if err := os.WriteFile(path, raw, 0600); err != nil {
				t.Fatal(err)
			}
			s, err := New(root)
			if err != nil {
				t.Fatal(err)
			}
			conf := &captureConfirmer{allow: true}
			deps := tools.NewContainer()
			tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })
			tools.Provide(deps, ConfirmKey, func(context.Context) (Confirmer, error) { return conf, nil })
			_, err = tc.tool.Execute(context.Background(), tc.args, deps)
			got, readErr := os.ReadFile(path)
			if err == nil || readErr != nil || !bytes.Equal(got, raw) || conf.action != "" {
				t.Fatalf("err=%v readErr=%v bytes=%x confirmation=%q", err, readErr, got, conf.action)
			}
		})
	}
}
