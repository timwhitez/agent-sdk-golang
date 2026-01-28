package tools

import "testing"

type schemaArgs struct {
	Required string `json:"required"`
	Optional *int   `json:"optional,omitempty"`
	Omit     string `json:"omit,omitempty"`
}

func TestSchemaForRequiredFields(t *testing.T) {
	s := SchemaFor[schemaArgs]()
	req, ok := s["required"].([]any)
	if !ok {
		t.Fatalf("required not []any")
	}
	reqSet := map[string]bool{}
	for _, v := range req {
		if name, ok := v.(string); ok {
			reqSet[name] = true
		}
	}
	if !reqSet["required"] {
		t.Fatalf("expected 'required' to be required")
	}
	if reqSet["optional"] {
		t.Fatalf("did not expect 'optional' to be required")
	}
	if reqSet["omit"] {
		t.Fatalf("did not expect 'omit' to be required")
	}
}
