package tools

import (
	"bytes"
	"context"
	"encoding/json"
	"math"
	"reflect"
	"strings"
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
		{`{"id":9007199254740991}`, 9007199254740991},
		{`{"id":9007199254740991,"extra":true}`, 9007199254740991},
		{`{"id":9007199254740992}`, 9007199254740992},
		{`{"id":9007199254740992,"extra":true}`, 9007199254740992},
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
	calls := 0
	tool := Func[Args]("numeric_fixture", "fixture", func(_ context.Context, args Args, _ *Container) (any, error) {
		calls++
		if args.ID != 9007199254740993 || args.FilePath != "a.txt" {
			t.Fatalf("alias repair changed typed execution: %+v", args)
		}
		return "ok", nil
	})
	if _, err := tool.Execute(context.Background(), string(raw), NewContainer()); err != nil || calls != 1 {
		t.Fatalf("alias repair calls=%d err=%v", calls, err)
	}
	repaired, ok, err := repairJSONKeysBySchemaWithOptions(schema, raw, schemaRepairOptions{StripUnknown: true})
	if err != nil || !ok {
		t.Fatalf("repair ok=%v err=%v", ok, err)
	}
	var args Args
	if err := json.Unmarshal(repaired, &args); err != nil || args.ID != 9007199254740993 || args.FilePath != "a.txt" {
		t.Fatalf("repaired=%s args=%+v err=%v", repaired, args, err)
	}
	for _, bad := range []string{`{`, `{"id":9007199254740993,"extra":true}{}`, `{"id":9007199254740993,"extra":true} garbage`} {
		if b, ok, err := repairJSONKeysBySchemaWithOptions(schema, []byte(bad), schemaRepairOptions{StripUnknown: true}); b != nil || ok || err != nil {
			t.Fatalf("invalid tail accepted: %q repaired=%s ok=%v err=%v", bad, b, ok, err)
		}
	}
}

func TestRepairRejectsOutOfRangeIntegerArguments(t *testing.T) {
	type Args struct {
		Signed   int64  `json:"signed"`
		Unsigned uint64 `json:"unsigned"`
	}
	for _, raw := range []string{
		`{"extra":true,"signed":9223372036854775808}`,
		`{"extra":true,"signed":-9223372036854775809}`,
		`{"extra":true,"unsigned":18446744073709551616}`,
		`{"extra":true,"unsigned":-1}`,
	} {
		t.Run(raw, func(t *testing.T) {
			calls := 0
			tool := Func[Args]("numeric_fixture", "fixture", func(context.Context, Args, *Container) (any, error) {
				calls++
				return "wrong", nil
			})
			prepared, _ := tool.PrepareCall(raw)
			if _, ok := prepared.FinalArgs(); ok {
				t.Fatal("out-of-range integer produced accepted final arguments")
			}
			ctx := WithToolResultMetadata(context.Background())
			if _, err := prepared.Execute(ctx, NewContainer()); err == nil || calls != 0 {
				t.Fatalf("out-of-range integer calls=%d err=%v", calls, err)
			}
			if kind, _ := ToolResultMetadataSnapshot(ctx)["args_repair_kind"].(string); strings.Contains(kind, "schema_key") {
				t.Fatalf("rejected integer published repair success: %q", kind)
			}
		})
	}
}

func TestRepairPreservesMixedScalarArguments(t *testing.T) {
	type Args struct {
		Integer int64   `json:"integer"`
		Float   float64 `json:"float"`
		Null    *int64  `json:"null"`
		Bool    bool    `json:"bool"`
		String  string  `json:"string"`
	}
	want := Args{Integer: 7, Float: 1.25, Bool: true, String: "42"}
	calls := 0
	tool := Func[Args]("numeric_fixture", "fixture", func(_ context.Context, args Args, _ *Container) (any, error) {
		calls++
		if !reflect.DeepEqual(args, want) {
			t.Fatalf("mixed scalar execution=%+v, want %+v", args, want)
		}
		return "ok", nil
	})
	prepared, _ := tool.PrepareCall(`{"extra":true,"integer":7,"float":1.25,"null":null,"bool":true,"string":"42"}`)
	view, ok := prepared.FinalArgs()
	var final Args
	if err := json.Unmarshal(view, &final); !ok || err != nil || !reflect.DeepEqual(final, want) {
		t.Fatalf("mixed scalar final args=%s ok=%v err=%v", view, ok, err)
	}
	ctx := WithToolResultMetadata(context.Background())
	if _, err := prepared.Execute(ctx, NewContainer()); err != nil || calls != 1 {
		t.Fatalf("mixed scalar calls=%d err=%v", calls, err)
	}
	if kind, _ := ToolResultMetadataSnapshot(ctx)["args_repair_kind"].(string); !strings.Contains(kind, "schema_key") {
		t.Fatal("mixed scalar fixture did not exercise schema repair")
	}
}
