package agent

import (
	"fmt"
	"strings"
)

func validCompactionSummary(body string) string {
	body = strings.TrimSpace(body)
	if body == "" {
		body = "UNKNOWN"
	}
	sections := []string{
		"Current Objective and Latest User Request",
		"Authoritative Current State",
		"Completed Work",
		"In-Progress and Remaining Work",
		"Exact External State",
		"Errors, Failed Attempts, and Successful Recovery",
		"Verification Already Run and Still Required",
		"Conflicts, Uncertainty, and Facts That Must Be Re-read",
	}
	var b strings.Builder
	b.WriteString("<summary>\n")
	for _, section := range sections {
		fmt.Fprintf(&b, "## %s\n%s\n\n", section, body)
	}
	b.WriteString("</summary>")
	return b.String()
}
