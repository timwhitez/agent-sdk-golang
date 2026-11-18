package tools

import (
	"encoding/json"
	"os"
	"os/exec"
	"testing"
	"time"
)

type s247Node struct {
	Name     string     `json:"name"`
	Children []s247Node `json:"children,omitempty"`
}

type s247Base struct {
	Path string `json:"path"`
	Dup  string `json:"dup"`
}

type S247Other struct {
	Dup   string `json:"dup"`
	Other string `json:"other,omitempty"`
}

type S247Exported struct {
	Exp string `json:"exp"`
}

type s247Embedded struct {
	s247Base
	*S247Other
	S247Exported `json:"tagged"`
	Mode         string    `json:"mode"`
	At           time.Time `json:"at"`
}

type s247Leaf struct {
	V string `json:"v"`
}

type s247Siblings struct {
	A s247Leaf `json:"a"`
	B s247Leaf `json:"b"`
}

func s247Props(t *testing.T, s map[string]any) map[string]any {
	t.Helper()
	p, _ := s["properties"].(map[string]any)
	return p
}

func TestSchemaForRecursiveTypeDoesNotCrash(t *testing.T) {
	if os.Getenv("S247_CHILD") == "1" {
		_ = SchemaFor[s247Node]()
		return
	}
	// A stack overflow is fatal, so probe in a child process.
	cmd := exec.Command(os.Args[0], "-test.run=^TestSchemaForRecursiveTypeDoesNotCrash$")
	cmd.Env = append(os.Environ(), "S247_CHILD=1")
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("SchemaFor on a recursive type crashed: %v: %.200s", err, out)
	}
	children := s247Props(t, SchemaFor[s247Node]())["children"].(map[string]any)
	if items, _ := children["items"].(map[string]any); len(items) != 0 {
		t.Fatalf("recursive reference should be an open schema, got %v", items)
	}
}

func TestSchemaForEmbeddedAndTime(t *testing.T) {
	s := SchemaFor[s247Embedded]()
	props := s247Props(t, s)
	for _, name := range []string{"path", "mode", "other", "tagged", "at"} {
		if _, ok := props[name]; !ok {
			t.Errorf("missing property %q in %v", name, props)
		}
	}
	if _, ok := props["dup"]; ok {
		t.Errorf("conflicting promoted field \"dup\" at equal depth must be dropped")
	}
	if _, ok := props["exp"]; ok {
		t.Errorf("tagged embedded struct must not be promoted")
	}
	at := props["at"].(map[string]any)
	if at["type"] != "string" || at["format"] != "date-time" {
		t.Errorf("time.Time schema = %v", at)
	}
	required := map[string]bool{}
	for _, r := range s["required"].([]any) {
		required[r.(string)] = true
	}
	if !required["path"] || required["other"] {
		t.Errorf("required = %v: value-embedded field is required, pointer-embedded optional", required)
	}
	// The schema must describe input the same type decodes.
	var out s247Embedded
	if err := json.Unmarshal([]byte(`{"path":"p","mode":"m","at":"2026-01-02T03:04:05Z","tagged":{"exp":"e"}}`), &out); err != nil || out.Path != "p" || out.At.IsZero() {
		t.Fatalf("decode = %+v, %v", out, err)
	}
}

func TestSchemaForSiblingStructsStayComplete(t *testing.T) {
	props := s247Props(t, SchemaFor[s247Siblings]())
	for _, name := range []string{"a", "b"} {
		if _, ok := s247Props(t, props[name].(map[string]any))["v"]; !ok {
			t.Errorf("sibling %s lost its fields: %v", name, props[name])
		}
	}
}

func TestSchemaForUnmarshalerTypes(t *testing.T) {
	type args struct {
		Raw   json.RawMessage `json:"raw"`
		Lvl   s247Level       `json:"lvl"`
		Items []s247Lenient   `json:"items"`
	}
	props := s247Props(t, SchemaFor[args]())
	if raw := props["raw"].(map[string]any); len(raw) != 0 {
		t.Errorf("non-struct json.Unmarshaler type = %v, want open schema", raw)
	}
	if lvl := props["lvl"].(map[string]any); lvl["type"] != "string" {
		t.Errorf("text unmarshaler type = %v, want string", lvl)
	}
	items := props["items"].(map[string]any)["items"].(map[string]any)
	if _, ok := s247Props(t, items)["label"]; !ok {
		t.Errorf("struct with UnmarshalJSON lost its field schema: %v", items)
	}
}

func TestSchemaForSkipsUnexportedEmbeddedPointer(t *testing.T) {
	type args struct {
		*s247Base
		Mode string `json:"mode"`
	}
	props := s247Props(t, SchemaFor[args]())
	if _, ok := props["path"]; ok {
		t.Errorf("unexported embedded pointer must not be promoted: %v", props)
	}
}

type s247Lenient struct {
	Label string `json:"label"`
}

func (l *s247Lenient) UnmarshalJSON(b []byte) error { return json.Unmarshal(b, &l.Label) }

type s247Level int

func (l *s247Level) UnmarshalText([]byte) error { *l = 1; return nil }
