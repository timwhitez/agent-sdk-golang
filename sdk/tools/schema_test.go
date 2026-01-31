package tools

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

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

func TestToolExecuteRepairsNonJSONArgs(t *testing.T) {
	type tc struct {
		name     string
		tool     string
		input    string
		expected string
	}
	cases := []tc{
		{name: "ls plain", tool: "ls", input: "/tmp", expected: `{"path":"/tmp"}`},
		{name: "read plain", tool: "read", input: "a.txt", expected: `{"file_path":"a.txt"}`},
		{name: "bash plain", tool: "bash", input: "pwd", expected: `{"command":"pwd"}`},
		{name: "glob plain", tool: "glob", input: "**/*.go", expected: `{"pattern":"**/*.go"}`},
		{name: "grep plain", tool: "grep", input: "TODO", expected: `{"pattern":"TODO"}`},
		{name: "webfetch plain", tool: "webfetch", input: "https://example.com", expected: `{"url":"https://example.com"}`},
		{name: "apply_patch plain", tool: "apply_patch", input: "*** Begin Patch\n*** End Patch", expected: `{"patch":"*** Begin Patch\n*** End Patch"}`},
		{name: "empty args", tool: "ls", input: "", expected: `{}`},
		{name: "already json", tool: "ls", input: `{"path":"."}`, expected: `{"path":"."}`},
	}

	for _, c := range cases {
		c := c
		t.Run(c.name, func(t *testing.T) {
			tt := Tool{
				Name: c.tool,
				Handler: func(_ context.Context, raw json.RawMessage, _ *Container) (llm.Content, error) {
					return llm.TextContent(string(raw)), nil
				},
			}
			out, err := tt.Execute(context.Background(), c.input, NewContainer())
			if err != nil {
				t.Fatalf("execute: %v", err)
			}
			if out.PlainText() != c.expected {
				t.Fatalf("expected %q, got %q", c.expected, out.PlainText())
			}
		})
	}
}

func TestToolExecuteRepairsAliasKeys(t *testing.T) {
	type writeArgs struct {
		FilePath string `json:"file_path"`
		Content  string `json:"content"`
	}
	writeTool := Func[writeArgs]("write", "test", func(_ context.Context, a writeArgs, _ *Container) (any, error) {
		return a.FilePath + "|" + a.Content, nil
	})
	out, err := writeTool.Execute(context.Background(), `{"path":"notes.txt","contents":"hello"}`, NewContainer())
	if err != nil {
		t.Fatalf("execute alias: %v", err)
	}
	if out.PlainText() != "notes.txt|hello" {
		t.Fatalf("expected alias mapping, got %q", out.PlainText())
	}

	type patternArgs struct {
		Pattern string `json:"pattern"`
	}
	patternTool := Func[patternArgs]("grep", "test", func(_ context.Context, a patternArgs, _ *Container) (any, error) {
		return a.Pattern, nil
	})
	out, err = patternTool.Execute(context.Background(), `{"input":"TODO"}`, NewContainer())
	if err != nil {
		t.Fatalf("execute single-field alias: %v", err)
	}
	if out.PlainText() != "TODO" {
		t.Fatalf("expected single-field alias mapping, got %q", out.PlainText())
	}
}
