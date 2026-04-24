package tools

import "strings"

var toolNamePrefixes = []string{
	"tool.",
	"tools.",
	"function.",
	"functions.",
	"tool:",
	"tools:",
	"function:",
	"functions:",
	"tool ",
	"tools ",
	"function ",
	"functions ",
}

// NormalizeToolName returns a normalized tool name for lookup/alias matching.
// It strips common prefixes, trims whitespace, and removes non-alphanumerics.
func NormalizeToolName(name string) string {
	name = strings.TrimSpace(name)
	if name == "" {
		return ""
	}
	lower := strings.ToLower(name)
	for {
		stripped := false
		for _, prefix := range toolNamePrefixes {
			if strings.HasPrefix(lower, prefix) {
				name = strings.TrimSpace(name[len(prefix):])
				lower = strings.ToLower(name)
				stripped = true
				break
			}
		}
		if !stripped {
			break
		}
	}
	return normalizeKeyNoDelims(name)
}
