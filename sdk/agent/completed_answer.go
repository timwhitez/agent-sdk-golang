package agent

import "strings"

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
