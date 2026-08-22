package llm

// CloneMessages returns a deep copy of a conversation history. The returned
// graph owns every mutable slice and pointee carried by Message, so callers may
// modify it without changing the source history.
func CloneMessages(messages []Message) []Message {
	if messages == nil {
		return nil
	}
	out := make([]Message, len(messages))
	for i := range messages {
		out[i] = CloneMessage(messages[i])
	}
	return out
}

// CloneMessage returns a deep copy of one provider-neutral message.
func CloneMessage(message Message) Message {
	out := message
	out.Content = CloneContent(message.Content)
	if message.ToolCalls != nil {
		out.ToolCalls = make([]ToolCall, len(message.ToolCalls))
		for i := range message.ToolCalls {
			out.ToolCalls[i] = CloneToolCall(message.ToolCalls[i])
		}
	}
	return out
}

// CloneContent returns a deep copy of message content and its pointer-bearing
// blocks.
func CloneContent(content Content) Content {
	out := content
	if content.Blocks == nil {
		return out
	}
	out.Blocks = make([]ContentBlock, len(content.Blocks))
	for i, block := range content.Blocks {
		out.Blocks[i] = block
		if block.ImageURL != nil {
			image := *block.ImageURL
			out.Blocks[i].ImageURL = &image
		}
		if block.Source != nil {
			source := *block.Source
			out.Blocks[i].Source = &source
		}
	}
	return out
}

// CloneToolCall returns a deep copy of one tool call, including provider-owned
// opaque byte signatures.
func CloneToolCall(call ToolCall) ToolCall {
	out := call
	if call.ThoughtSig != nil {
		out.ThoughtSig = append([]byte(nil), call.ThoughtSig...)
	}
	return out
}
