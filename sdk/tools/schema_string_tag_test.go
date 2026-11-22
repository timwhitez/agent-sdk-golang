package tools

import (
	"context"
	"encoding/json"
	"reflect"
	"testing"
)

func TestSchemaQuotedIntMatchesDecoder(t *testing.T) {
	type args struct {
		Count int `json:"count,string"`
	}
	var seen []int
	tool := Func[args]("count", "", func(_ context.Context, a args, _ *Container) (any, error) {
		seen = append(seen, a.Count)
		return "ok", nil
	})
	prop := tool.Definition().Parameters["properties"].(map[string]any)["count"].(map[string]any)
	_, quotedErr := tool.Execute(context.Background(), `{"count":"7"}`, nil)
	_, numberErr := tool.Execute(context.Background(), `{"count":7}`, nil)
	if quotedErr != nil || numberErr == nil || len(seen) != 1 || seen[0] != 7 {
		t.Fatalf("bad control: seen=%v quotedErr=%v numberErr=%v", seen, quotedErr, numberErr)
	}
	if prop["type"] != "string" {
		t.Fatalf("advertised type=%v; advertised value 7 fails: %v; quoted value succeeds", prop["type"], numberErr)
	}
}

func TestSchemaStringTagSupportedKindsAndControls(t *testing.T) {
	type args struct {
		Signed   int64   `json:"signed,string"`
		Unsigned uint64  `json:"unsigned,omitempty,string"`
		Float    float64 `json:"float,string"`
		Bool     bool    `json:"bool,string"`
		Text     string  `json:"text,string"`
		Pointer  *int    `json:"pointer,string,omitempty"`
		Plain    int     `json:"plain"`
		Ignored  []int   `json:"ignored,string"`
	}
	n := 8
	want := args{Signed: -7, Unsigned: 7, Float: 1.25, Bool: true, Text: "hello", Pointer: &n, Plain: 9, Ignored: []int{1}}
	props := SchemaFor[args]()["properties"].(map[string]any)
	for _, name := range []string{"signed", "unsigned", "float", "bool", "text", "pointer"} {
		if typ := props[name].(map[string]any)["type"]; typ != "string" {
			t.Errorf("%s schema type = %v", name, typ)
		}
	}
	if props["plain"].(map[string]any)["type"] != "integer" || props["ignored"].(map[string]any)["type"] != "array" {
		t.Fatal("string option changed unsupported/untagged fields")
	}
	raw, err := json.Marshal(want)
	if err != nil {
		t.Fatal(err)
	}
	calls := 0
	tool := Func[args]("record", "", func(_ context.Context, got args, _ *Container) (any, error) {
		calls++
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("got %+v, want %+v", got, want)
		}
		return nil, nil
	})
	if _, err := tool.Execute(context.Background(), string(raw), nil); err != nil || calls != 1 {
		t.Fatalf("calls=%d err=%v input=%s", calls, err, raw)
	}
}
