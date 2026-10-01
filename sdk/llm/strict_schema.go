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
	for _, key := range []string{"$defs", "definitions"} {
		if definitions, ok := schema[key].(map[string]any); ok {
			names := make([]string, 0, len(definitions))
			for name := range definitions {
				names = append(names, name)
			}
			sort.Strings(names)
			for _, name := range names {
				definition, ok := definitions[name].(map[string]any)
				if !ok {
					return fmt.Errorf("%s.%s.%s: unsupported definition schema", path, key, name)
				}
				if err := strictSchemaCompatibility(definition, path+"."+key+"."+name, depth+1); err != nil {
					return err
				}
			}
		}
	}
	constrained := false
	if ref, ok := schema["$ref"].(string); ok && ref != "" {
		constrained = true
	}
	for _, key := range []string{"anyOf", "oneOf", "allOf"} {
		if branches, ok := schema[key].([]any); ok && len(branches) > 0 {
			constrained = true
			for i, branch := range branches {
				child, ok := branch.(map[string]any)
				if !ok {
					return fmt.Errorf("%s.%s[%d]: unsupported branch schema", path, key, i)
				}
				if err := strictSchemaCompatibility(child, fmt.Sprintf("%s.%s[%d]", path, key, i), depth+1); err != nil {
					return err
				}
			}
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
	if len(types) == 0 && !constrained {
		return fmt.Errorf("%s: unconstrained JSON values require non-strict mode", path)
	}
	if slices.Contains(types, "object") {
		if allowed, ok := schema["additionalProperties"].(bool); !ok || allowed {
			return fmt.Errorf("%s.additionalProperties: open objects require non-strict mode", path)
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
