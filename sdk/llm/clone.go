package llm

// CloneContent returns a deep copy of content and every mutable nested
// value. Strings are immutable values; pointer-backed blocks are copied
// so callers cannot mutate a retained conversation through aliases.
func CloneContent(content Content) Content {
	out := content
	if content.Blocks != nil {
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
	}
	return out
}

// CloneToolCalls returns a deep copy of tool calls, including opaque
// provider signatures that are stored as mutable byte slices.
func CloneToolCalls(calls []ToolCall) []ToolCall {
	if calls == nil {
		return nil
	}
	out := make([]ToolCall, len(calls))
	copy(out, calls)
	for i := range calls {
		if calls[i].ThoughtSig != nil {
			out[i].ThoughtSig = append([]byte{}, calls[i].ThoughtSig...)
		}
	}
	return out
}

// CloneMessage returns a deep copy suitable for crossing an ownership
// boundary between an Agent and its caller.
func CloneMessage(message Message) Message {
	out := message
	out.Content = CloneContent(message.Content)
	out.ToolCalls = CloneToolCalls(message.ToolCalls)
	return out
}

// CloneMessages preserves nil-versus-empty slice semantics while
// returning messages with no mutable aliases to the input.
func CloneMessages(messages []Message) []Message {
	if messages == nil {
		return nil
	}
	out := make([]Message, len(messages))
	for i, message := range messages {
		out[i] = CloneMessage(message)
	}
	return out
}
