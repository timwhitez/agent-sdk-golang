package tools

import (
	"reflect"
	"strings"
)

// SchemaFor generates a JSON schema suitable for OpenAI/Anthropic tool calling.
// It is intentionally conservative and focuses on the shapes used by this repo.
func SchemaFor[T any]() map[string]any {
	var zero T
	rt := reflect.TypeOf(zero)
	if rt == nil {
		return map[string]any{"type": "object", "properties": map[string]any{}, "required": []any{}, "additionalProperties": false}
	}
	if rt.Kind() == reflect.Pointer {
		rt = rt.Elem()
	}
	if rt.Kind() != reflect.Struct {
		return map[string]any{"type": "object", "properties": map[string]any{"value": schemaForType(rt)}, "required": []any{"value"}, "additionalProperties": false}
	}
	props := map[string]any{}
	req := []any{}

	for i := 0; i < rt.NumField(); i++ {
		f := rt.Field(i)
		if f.PkgPath != "" { // unexported
			continue
		}
		name, omit := jsonFieldName(f)
		if name == "" {
			continue
		}
		ft := f.Type
		isPtr := ft.Kind() == reflect.Pointer
		if isPtr {
			ft = ft.Elem()
		}
		props[name] = schemaForType(ft)
		if !omit && !isPtr {
			req = append(req, name)
		}
	}

	return map[string]any{
		"type":                 "object",
		"properties":           props,
		"required":             req,
		"additionalProperties": false,
	}
}

func jsonFieldName(f reflect.StructField) (name string, omitempty bool) {
	tag := f.Tag.Get("json")
	if tag == "-" {
		return "", false
	}
	if tag != "" {
		parts := strings.Split(tag, ",")
		name = parts[0]
		for _, p := range parts[1:] {
			if p == "omitempty" {
				omitempty = true
				break
			}
		}
		if name == "" {
			name = lowerFirst(f.Name)
		}
		return name, omitempty
	}
	return lowerFirst(f.Name), false
}

func lowerFirst(s string) string {
	if s == "" {
		return ""
	}
	r := []rune(s)
	r[0] = []rune(strings.ToLower(string(r[0])))[0]
	return string(r)
}

func schemaForType(t reflect.Type) map[string]any {
	if t == nil {
		return map[string]any{"type": "string"}
	}
	if t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	switch t.Kind() {
	case reflect.String:
		return map[string]any{"type": "string"}
	case reflect.Bool:
		return map[string]any{"type": "boolean"}
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return map[string]any{"type": "integer"}
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		return map[string]any{"type": "integer"}
	case reflect.Float32, reflect.Float64:
		return map[string]any{"type": "number"}
	case reflect.Slice, reflect.Array:
		return map[string]any{"type": "array", "items": schemaForType(t.Elem())}
	case reflect.Map:
		// JSON only supports string keys.
		return map[string]any{"type": "object", "additionalProperties": schemaForType(t.Elem())}
	case reflect.Struct:
		// Special-cases
		if t.PkgPath() == "time" && t.Name() == "Duration" {
			return map[string]any{"type": "string"}
		}
		props := map[string]any{}
		req := []any{}
		for i := 0; i < t.NumField(); i++ {
			f := t.Field(i)
			if f.PkgPath != "" {
				continue
			}
			name, omit := jsonFieldName(f)
			if name == "" {
				continue
			}
			ft := f.Type
			isPtr := ft.Kind() == reflect.Pointer
			if isPtr {
				ft = ft.Elem()
			}
			props[name] = schemaForType(ft)
			if !omit && !isPtr {
				req = append(req, name)
			}
		}
		return map[string]any{"type": "object", "properties": props, "required": req, "additionalProperties": false}
	default:
		return map[string]any{"type": "string"}
	}
}
