package compaction

import "regexp"

const (
	DefaultContextWindow  = 128_000
	DefaultThresholdRatio = 0.80
)

// DefaultSummaryPrompt is adapted from the Python SDK.
const DefaultSummaryPrompt = `You have been working on the task described above but have not yet completed it. Write a continuation summary that will allow you (or another instance of yourself) to resume work efficiently in a future context window where the conversation history will be replaced with this summary. Your summary should be structured, concise, and actionable. Include:

1. Task Overview
The user's core request and success criteria
Any clarifications or constraints they specified

2. Current State
What has been completed so far
Files created, modified, or analyzed (with paths if relevant)
Key outputs or artifacts produced

3. Important Discoveries
Technical constraints or requirements uncovered
Decisions made and their rationale
Errors encountered and how they were resolved
What approaches were tried that didn't work (and why)

4. Next Steps
Specific actions needed to complete the task
Any blockers or open questions to resolve
Priority order if multiple steps remain

5. Context to Preserve
User preferences or style requirements
Domain-specific details that aren't obvious
Any promises made to the user

Be concise but complete - err on the side of including information that would prevent duplicate work or repeated mistakes. Write in a way that enables immediate resumption of the task.

Wrap your summary in <summary></summary> tags.`

type Config struct {
	Enabled        bool
	ThresholdRatio float64
	SummaryPrompt  string
}

func DefaultConfig() Config {
	return Config{Enabled: true, ThresholdRatio: DefaultThresholdRatio, SummaryPrompt: DefaultSummaryPrompt}
}

type Result struct {
	Compacted      bool
	OriginalTokens int
	NewTokens      int
	Summary        string
}

var summaryRe = regexp.MustCompile(`(?s)<summary>(.*?)</summary>`)

func ExtractSummary(text string) string {
	m := summaryRe.FindStringSubmatch(text)
	if len(m) == 2 {
		return stringsTrim(m[1])
	}
	return stringsTrim(text)
}

func stringsTrim(s string) string {
	// avoid importing strings in multiple files; tiny helper
	for len(s) > 0 {
		r := s[0]
		if r == ' ' || r == '\n' || r == '\t' || r == '\r' {
			s = s[1:]
			continue
		}
		break
	}
	for len(s) > 0 {
		r := s[len(s)-1]
		if r == ' ' || r == '\n' || r == '\t' || r == '\r' {
			s = s[:len(s)-1]
			continue
		}
		break
	}
	return s
}
