package tools

import (
	"encoding"
	"encoding/json"
	"reflect"
	"strings"
	"time"
)

// SchemaFor generates a JSON schema suitable for OpenAI/Anthropic tool calling.
// It is intentionally conservative and focuses on the shapes used by this repo.
func SchemaFor[T any]() map[string]any {
	rt := reflect.TypeOf((*T)(nil))
	if rt == nil {
		return map[string]any{"type": "object", "properties": map[string]any{}, "required": []any{}, "additionalProperties": false}
	}
	rt = rt.Elem()
	if rt.Kind() == reflect.Pointer {
		rt = rt.Elem()
	}
	if rt.Kind() != reflect.Struct {
		return map[string]any{"type": "object", "properties": map[string]any{"value": schemaForType(rt, map[reflect.Type]bool{})}, "required": []any{"value"}, "additionalProperties": false}
	}
	return structSchema(rt, map[reflect.Type]bool{})
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

var (
	timeType            = reflect.TypeOf(time.Time{})
	jsonUnmarshalerType = reflect.TypeOf((*json.Unmarshaler)(nil)).Elem()
	textUnmarshalerType = reflect.TypeOf((*encoding.TextUnmarshaler)(nil)).Elem()
)

// customDecodedSchema describes types whose decoding is not driven by their Go
// shape: time.Time is an RFC 3339 string, text-only unmarshalers take a string,
// and non-struct json.Unmarshaler types (e.g. json.RawMessage) accept anything
// this reflection cannot infer. Structs with a custom UnmarshalJSON keep their
// reflected field schema: such decoders are typically lenient supersets of the
// struct shape, and dropping it would remove all guidance for the model.
func customDecodedSchema(t reflect.Type) (map[string]any, bool) {
	if t == timeType {
		return map[string]any{"type": "string", "format": "date-time"}, true
	}
	pt := reflect.PointerTo(t)
	if pt.Implements(jsonUnmarshalerType) {
		if t.Kind() == reflect.Struct {
			return nil, false
		}
		return map[string]any{}, true
	}
	if pt.Implements(textUnmarshalerType) {
		return map[string]any{"type": "string"}, true
	}
	return nil, false
}

// schemaField is one candidate JSON property gathered while flattening embedded structs.
type schemaField struct {
	name     string
	depth    int
	tagged   bool
	schema   map[string]any
	required bool
}

// structSchema builds the object schema for struct type t. path holds the struct
// types currently being expanded so a recursive type terminates.
func structSchema(t reflect.Type, path map[reflect.Type]bool) map[string]any {
	var fields []schemaField
	collectSchemaFields(t, 0, false, path, &fields)
	// Resolve name conflicts like encoding/json: the shallowest field wins, a
	// single tagged field breaks ties at equal depth, otherwise the name is dropped.
	props := map[string]any{}
	req := []any{}
	for i, f := range fields {
		best, dominant := f, true
		for j, o := range fields {
			if i == j || o.name != f.name {
				continue
			}
			switch {
			case o.depth < f.depth, o.depth == f.depth && o.tagged && !f.tagged, o.depth == f.depth && o.tagged == f.tagged:
				dominant = false
			}
		}
		if !dominant {
			continue
		}
		props[best.name] = best.schema
		if best.required {
			req = append(req, best.name)
		}
	}
	return map[string]any{"type": "object", "properties": props, "required": req, "additionalProperties": false}
}

func collectSchemaFields(t reflect.Type, depth int, optional bool, path map[reflect.Type]bool, out *[]schemaField) {
	if path[t] {
		return
	}
	path[t] = true
	defer delete(path, t)
	for i := 0; i < t.NumField(); i++ {
		f := t.Field(i)
		tag := f.Tag.Get("json")
		if tag == "-" {
			continue
		}
		tagName, _, _ := strings.Cut(tag, ",")
		ft := f.Type
		isPtr := ft.Kind() == reflect.Pointer
		if isPtr {
			ft = ft.Elem()
		}
		if f.Anonymous && tagName == "" && ft.Kind() == reflect.Struct {
			if isPtr && f.PkgPath != "" {
				continue // encoding/json cannot allocate an unexported embedded pointer
			}
			if _, custom := customDecodedSchema(ft); !custom {
				// Promoted like encoding/json, also from unexported embedded types.
				collectSchemaFields(ft, depth+1, optional || isPtr, path, out)
				continue
			}
		}
		if f.PkgPath != "" { // unexported
			continue
		}
		name, omit := jsonFieldName(f)
		if name == "" {
			continue
		}
		*out = append(*out, schemaField{name: name, depth: depth, tagged: tagName != "", schema: schemaForType(ft, path), required: !omit && !isPtr && !optional})
	}
}

func schemaForType(t reflect.Type, path map[reflect.Type]bool) map[string]any {
	if t == nil {
		return map[string]any{"type": "string"}
	}
	if t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	if schema, ok := customDecodedSchema(t); ok {
		return schema
	}
	switch t.Kind() {
	case reflect.String:
		return map[string]any{"type": "string"}
	case reflect.Bool:
		return map[string]any{"type": "boolean"}
	case reflect.Interface:
		// Allow any JSON value for interface{} / any fields.
		return map[string]any{}
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return map[string]any{"type": "integer"}
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		return map[string]any{"type": "integer"}
	case reflect.Float32, reflect.Float64:
		return map[string]any{"type": "number"}
	case reflect.Slice, reflect.Array:
		return map[string]any{"type": "array", "items": schemaForType(t.Elem(), path)}
	case reflect.Map:
		// JSON only supports string keys.
		return map[string]any{"type": "object", "additionalProperties": schemaForType(t.Elem(), path)}
	case reflect.Struct:
		if path[t] {
			// Recursive type: JSON schema here has no $ref, so accept any value
			// instead of expanding forever.
			return map[string]any{}
		}
		return structSchema(t, path)
	default:
		return map[string]any{"type": "string"}
	}
}
