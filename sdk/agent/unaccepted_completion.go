package agent

import "github.com/timwhitez/agent-sdk-golang/sdk/llm"

// appendUnacceptedCompletion retains observed partial content without accepting
// its tool calls. Opaque state can carry the provider's original call topology,
// so discarding any calls invalidates that whole state group just as abandoning
// a tool-call continuation does. State-only responses without calls remain
// replayable; the driver never interprets provider-owned payload fields.
func (a *Agent) appendUnacceptedCompletion(completion *llm.Completion) {
	if completion == nil {
		return
	}
	var content llm.Content
	if len(completion.ToolCalls) > 0 {
		content = llm.WithoutProviderState(completion.Content)
	} else {
		content = llm.CloneContent(completion.Content)
	}
	if content.IsEmpty() && !llm.HasProviderState(content) {
		return
	}
	a.appendMessages([]llm.Message{{Role: llm.RoleAssistant, Content: content}})
}
