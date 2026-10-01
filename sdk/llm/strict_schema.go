package llm

import (
	"fmt"
	"slices"
	"sort"
)

// StrictSchemaCompatibility reports the first path that cannot be faithfully
// normalized by the SDK's OpenAI strict schema mapper. It does not mutate schema
// or validate the entire JSON Schema vocabulary. Open objects and unconstrained
// JSON values require non-strict mode; closing them would discard valid input.
func StrictSchemaCompatibility(schema map[string]any) error {
	return strictSchemaCompatibility(schema, "$", 0)
}

func strictSchemaCompatibility(schema map[string]any, path string, depth int) error {
	if depth >= 128 {
		return fmt.Errorf("%s: schema nesting exceeds 128 levels", path)
	}
	for _, key := range []string{"$ref", "$defs", "definitions", "anyOf", "oneOf", "allOf"} {
		if _, ok := schema[key]; ok {
			return fmt.Errorf("%s.%s: schema composition is unsupported by strict normalization", path, key)
		}
	}
	types := []string{}
	switch typ := schema["type"].(type) {
	case string:
		types = append(types, typ)
	case []string:
		types = typ
	case []any:
		for _, value := range typ {
			if typ, ok := value.(string); ok {
				types = append(types, typ)
			}
		}
	}
	if len(types) == 0 {
		return fmt.Errorf("%s: unconstrained JSON values require non-strict mode", path)
	}
	if slices.Contains(types, "object") {
		if allowed, ok := schema["additionalProperties"].(bool); !ok || allowed {
			return fmt.Errorf("%s.additionalProperties: open objects require non-strict mode", path)
		}
		if len(types) > 1 {
			return fmt.Errorf("%s.type: object unions are unsupported by strict normalization", path)
		}
		props, _ := schema["properties"].(map[string]any)
		names := make([]string, 0, len(props))
		for name := range props {
			names = append(names, name)
		}
		sort.Strings(names)
		for _, name := range names {
			prop, ok := props[name].(map[string]any)
			if !ok {
				return fmt.Errorf("%s.properties.%s: unsupported property schema", path, name)
			}
			if err := strictSchemaCompatibility(prop, path+".properties."+name, depth+1); err != nil {
				return err
			}
		}
	}
	if slices.Contains(types, "array") {
		items, ok := schema["items"].(map[string]any)
		if !ok {
			return fmt.Errorf("%s.items: unconstrained array items require non-strict mode", path)
		}
		return strictSchemaCompatibility(items, path+".items", depth+1)
	}
	return nil
}
