package tools

import (
	"encoding/json"
	"fmt"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

func SerializeResult(v any) (llm.Content, error) {
	if v == nil {
		return llm.TextContent(""), nil
	}
	switch x := v.(type) {
	case llm.Content:
		return x, nil
	case string:
		return llm.TextContent(x), nil
	case []byte:
		return llm.TextContent(string(x)), nil
	case []llm.ContentBlock:
		return llm.Content{Blocks: x}, nil
	default:
		b, err := json.Marshal(x)
		if err != nil {
			return llm.TextContent(fmt.Sprint(x)), nil
		}
		return llm.TextContent(string(b)), nil
	}
}
