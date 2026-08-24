package openai

import (
	"fmt"
	"reflect"
	"testing"
)

func TestMakeStrictSchemaSortsRequiredPropertiesRecursively(t *testing.T) {
	t.Parallel()

	expectedTop := []any{"alpha", "beta", "delta", "nested"}
	expectedNested := []any{"eta", "zeta"}

	for i := 0; i < 512; i++ {
		schema := map[string]any{
			"type": "object",
			"properties": map[string]any{
				"delta": map[string]any{"type": "string"},
				"alpha": map[string]any{"type": "integer"},
				"nested": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"zeta": map[string]any{"type": "string"},
						"eta":  map[string]any{"type": "boolean"},
					},
				},
				"beta": map[string]any{"type": "number"},
			},
		}

		strict := makeStrictSchema(schema)
		if got := strict["required"]; !reflect.DeepEqual(got, expectedTop) {
			t.Fatalf("iteration %d: top-level required = %#v, want %#v", i, got, expectedTop)
		}

		properties, ok := strict["properties"].(map[string]any)
		if !ok {
			t.Fatalf("iteration %d: properties type = %T", i, strict["properties"])
		}
		nested, ok := properties["nested"].(map[string]any)
		if !ok {
			t.Fatalf("iteration %d: nested schema type = %T", i, properties["nested"])
		}
		if got := nested["required"]; !reflect.DeepEqual(got, expectedNested) {
			t.Fatalf("iteration %d: nested required = %#v, want %#v", i, got, expectedNested)
		}

		encoded := fmt.Sprint(strict["required"], nested["required"])
		if encoded != "[alpha beta delta nested] [eta zeta]" {
			t.Fatalf("iteration %d: canonical order changed: %s", i, encoded)
		}
	}
}
