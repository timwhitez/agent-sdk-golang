package llm

import (
	"encoding/json"
	"reflect"
	"testing"
)

type dynamicJSON int

func (dynamicJSON) MarshalJSON() ([]byte, error) { panic("must not execute serializer during binding") }

type dynamicPointer struct{ Value int }

func (*dynamicPointer) MarshalJSON() ([]byte, error) { panic("must not execute pointer serializer") }

type dynamicKey string

func (dynamicKey) MarshalText() ([]byte, error) { panic("must not execute map key serializer") }

type privateMutable struct{ values []int }

func TestCloneStaticJSONMapOwnsValuesWithoutRoundtrip(t *testing.T) {
	value := int64(9007199254740993)
	raw := json.RawMessage(`{"safe":true}`)
	source := map[string]any{"integer": value, "pointer": &value, "raw": raw, "slice": []map[string]int{{"x": 1}}, "empty": []string{}, "nil": []string(nil), "number": json.Number("9007199254740993")}
	owned, err := CloneStaticJSONMap(source)
	if err != nil || !reflect.DeepEqual(source, owned) {
		t.Fatal("value/type changed", err)
	}
	value = 1
	raw[0] = '['
	source["slice"].([]map[string]int)[0]["x"] = 9
	if *owned["pointer"].(*int64) != 9007199254740993 || owned["raw"].(json.RawMessage)[0] != '{' || owned["slice"].([]map[string]int)[0]["x"] != 1 {
		t.Fatal("clone aliases mutable input")
	}
	for _, source := range []map[string]any{nil, {}} {
		owned, err := CloneStaticJSONMap(source)
		if err != nil || !reflect.DeepEqual(source, owned) {
			t.Fatal("nil/empty shape changed")
		}
	}
}

func TestCloneStaticJSONMapRejectsDynamicAndUnboundedGraphs(t *testing.T) {
	cycle := map[string]any{}
	cycle["self"] = cycle
	branch := map[string]any{"leaf": 1}
	for i := 0; i < 20; i++ {
		branch = map[string]any{"a": branch, "b": branch}
	}
	for _, value := range []any{dynamicJSON(1), dynamicPointer{Value: 1}, &dynamicPointer{Value: 1}, map[dynamicKey]int{"x": 1}, privateMutable{values: []int{1}}, make(chan int), func() {}, cycle, branch, make([]int, 100001)} {
		if owned, err := CloneStaticJSONMap(map[string]any{"fixture": value}); err == nil || owned != nil {
			t.Fatalf("unsafe configuration accepted: %T", value)
		}
	}
	// Existing request cloning retains its old type-preserving behavior. The
	// stricter binding policy must not silently rewrite that public contract.
	legacy := map[string]any{"value": dynamicJSON(1), "pointer": dynamicPointer{Value: 2}}
	if cloned, err := cloneJSONMap(legacy); err != nil || !reflect.DeepEqual(cloned, legacy) {
		t.Fatal("legacy cloning policy changed", err)
	}
}
