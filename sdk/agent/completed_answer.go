package agent

import (
	"strings"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

// Preserve otherwise-empty whitespace fragments only for explicit continuation
// aggregation. PlainText's public meaningful-content classification stays intact.
func continuationText(content llm.Content) string {
	if text := content.PlainText(); text != "" {
		return text
	}
	if content.Text != "" {
		return content.Text
	}
	var fragments []string
	for _, block := range content.Blocks {
		if block.Type == "text" && block.Text != "" {
			fragments = append(fragments, block.Text)
		}
	}
	return strings.Join(fragments, "\n")
}

// Completion payloads are explicit content, not prose to classify as a generic
// acknowledgement. Preserve both pieces unless they are exactly the same.
func completedAnswer(answer, completion string) string {
	answer, completion = strings.TrimSpace(answer), strings.TrimSpace(completion)
	if answer == "" {
		return completion
	}
	if completion == "" || completion == answer {
		return answer
	}
	return answer + "\n\n" + completion
}
