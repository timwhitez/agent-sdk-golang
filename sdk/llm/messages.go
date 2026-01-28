package llm

func NewSystemMessage(text string) Message {
	return Message{Role: RoleSystem, Content: TextContent(text)}
}

func NewUserMessage(text string) Message {
	return Message{Role: RoleUser, Content: TextContent(text)}
}

func NewAssistantMessage(text string, toolCalls []ToolCall) Message {
	m := Message{Role: RoleAssistant, Content: TextContent(text)}
	if len(toolCalls) > 0 {
		m.ToolCalls = append([]ToolCall(nil), toolCalls...)
	}
	return m
}

func NewToolMessage(toolCallID, toolName string, content Content, isError bool) Message {
	return Message{
		Role:       RoleTool,
		ToolCallID: toolCallID,
		ToolName:   toolName,
		Content:    content,
		IsError:    isError,
	}
}
