package tools

import (
	"bytes"
	"context"
	"encoding/json"
	"math"
	"testing"
)

func TestRepairPreservesCanonicalIntegerValue(t *testing.T) {
	type Args struct {
		ID int64 `json:"id"`
	}
	for _, tc := range []struct {
		raw  string
		want int64
	}{
		{`{"id":9007199254740993}`, 9007199254740993},
		{`{"id":9007199254740993,"extra":true}`, 9007199254740993},
		{`{"extra":true,"id":9007199254740993}`, 9007199254740993},
		{`{"id":9223372036854775807,"extra":true}`, math.MaxInt64},
		{`{"id":-9007199254740993,"extra":true}`, -9007199254740993},
	} {
		t.Run(tc.raw, func(t *testing.T) {
			calls := 0
			var got int64
			tool := Func[Args]("numeric_fixture", "fixture", func(_ context.Context, args Args, _ *Container) (any, error) {
				calls++
				got = args.ID
				return "ok", nil
			})
			prepared, _ := tool.PrepareCall(tc.raw)
			view, ok := prepared.FinalArgs()
			if !ok {
				t.Fatal("final arguments unavailable")
			}
			var final Args
			if err := json.Unmarshal(view, &final); err != nil || final.ID != tc.want {
				t.Fatalf("final args=%s, err=%v, want id=%d", view, err, tc.want)
			}
			if _, err := prepared.Execute(context.Background(), NewContainer()); err != nil || calls != 1 || got != tc.want {
				t.Fatalf("calls=%d got=%d want=%d err=%v", calls, got, tc.want, err)
			}
		})
	}
}

func TestRepairPreservesNestedAndUnsignedNumbers(t *testing.T) {
	type Args struct {
		ID     uint64 `json:"id"`
		Nested struct {
			Value int64 `json:"value"`
		} `json:"nested"`
		List []struct {
			Value int64 `json:"value"`
		} `json:"list"`
		Values map[string]int64 `json:"values"`
	}
	const raw = `{"id":18446744073709551615,"nested":{"value":9007199254740993},"list":[{"value":-9007199254740993}],"values":{"target":9007199254740993},"extra":true}`
	tool := Func[Args]("numeric_fixture", "fixture", func(_ context.Context, args Args, _ *Container) (any, error) {
		if args.ID != math.MaxUint64 {
			t.Errorf("id=%d", args.ID)
		}
		if args.Nested.Value != 9007199254740993 || len(args.List) != 1 || args.List[0].Value != -9007199254740993 || args.Values["target"] != 9007199254740993 {
			t.Errorf("nested values changed: %+v", args)
		}
		return "ok", nil
	})
	prepared, _ := tool.PrepareCall(raw)
	view, ok := prepared.FinalArgs()
	if !ok {
		t.Fatal("final arguments unavailable")
	}
	if _, err := prepared.Execute(context.Background(), NewContainer()); err != nil {
		t.Fatal(err)
	}
	if !json.Valid(view) || !bytes.Contains(view, []byte(`"id":18446744073709551615`)) {
		t.Fatalf("final args=%s", view)
	}
}

func TestRepairPreservesNumberBesideAliasAndRejectsJSONTail(t *testing.T) {
	type Args struct {
		ID       int64  `json:"id"`
		FilePath string `json:"filePath"`
	}
	schema := SchemaFor[Args]()
	raw := []byte(`{"id":9007199254740993,"path":"a.txt"}  `)
	repaired, ok, err := repairJSONKeysBySchemaWithOptions(schema, raw, schemaRepairOptions{StripUnknown: true})
	if err != nil || !ok {
		t.Fatalf("repair ok=%v err=%v", ok, err)
	}
	var args Args
	if err := json.Unmarshal(repaired, &args); err != nil || args.ID != 9007199254740993 || args.FilePath != "a.txt" {
		t.Fatalf("repaired=%s args=%+v err=%v", repaired, args, err)
	}
	for _, bad := range []string{`{"id":9007199254740993,"extra":true}{}`, `{"id":9007199254740993,"extra":true} garbage`} {
		if b, ok, err := repairJSONKeysBySchemaWithOptions(schema, []byte(bad), schemaRepairOptions{StripUnknown: true}); b != nil || ok || err != nil {
			t.Fatalf("invalid tail accepted: %q repaired=%s ok=%v err=%v", bad, b, ok, err)
		}
	}
}
