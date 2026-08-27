package compaction

import (
	"fmt"
	"regexp"
	"sort"
	"strings"
)

var requiredSummarySections = []string{
	"Current Objective and Latest User Request",
	"Authoritative Current State",
	"Completed Work",
	"In-Progress and Remaining Work",
	"Exact External State",
	"Errors, Failed Attempts, and Successful Recovery",
	"Verification Already Run and Still Required",
	"Conflicts, Uncertainty, and Facts That Must Be Re-read",
}

type summaryValidationError struct {
	reasons []string
}

func (e *summaryValidationError) Error() string {
	if e == nil || len(e.reasons) == 0 {
		return "summary validation failed"
	}
	return strings.Join(e.reasons, "; ")
}

func validateSummaryOutput(raw, material string) (string, error) {
	summary, reasons := validateSummaryEnvelope(raw)
	if summary != "" {
		reasons = append(reasons, validateRequiredSummarySections(summary)...)
		reasons = append(reasons, validateSummaryFactCoverage(summary, material)...)
		if strings.Contains(material, "Status: "+CheckpointStatusUnknown) && !strings.Contains(strings.ToUpper(summary), CheckpointStatusUnknown) {
			reasons = append(reasons, "host state is UNKNOWN but summary does not preserve UNKNOWN")
		}
	}
	if len(reasons) > 0 {
		sort.Strings(reasons)
		return "", &summaryValidationError{reasons: reasons}
	}
	return summary, nil
}

func validateSummaryFactCoverage(summary, material string) []string {
	reasons := []string{}
	latest := materialSectionBody(material, "Latest Real User Request")
	objective := materialSectionBody(summary, "Current Objective and Latest User Request")
	if strings.TrimSpace(latest) != "" && strings.ToUpper(strings.TrimSpace(latest)) != CheckpointStatusUnknown && summaryBodyIsOnlyUnknown(objective) {
		reasons = append(reasons, "latest user request was supplied but Current Objective and Latest User Request is UNKNOWN")
	}
	if strings.Contains(material, "## Host Checkpoint Context") && strings.Contains(material, "Status: "+CheckpointStatusVerified) {
		external := materialSectionBody(summary, "Exact External State")
		if summaryBodyIsOnlyUnknown(external) {
			reasons = append(reasons, "verified host state was supplied but Exact External State is UNKNOWN")
		}
	}
	return reasons
}

func materialSectionBody(text, title string) string {
	re := regexp.MustCompile(`(?m)^## ` + regexp.QuoteMeta(title) + `\r?$`)
	loc := re.FindStringIndex(text)
	if loc == nil {
		return ""
	}
	rest := text[loc[1]:]
	nextHeading := regexp.MustCompile(`(?m)^#{1,6}\s+.+$`).FindStringIndex(rest)
	if nextHeading != nil {
		rest = rest[:nextHeading[0]]
	}
	if end := strings.Index(rest, endUntrustedMaterial); end >= 0 {
		rest = rest[:end]
	}
	return strings.TrimSpace(rest)
}

func summaryBodyIsOnlyUnknown(body string) bool {
	body = strings.TrimSpace(body)
	body = strings.Trim(body, "-*.: `\t\r\n")
	return strings.EqualFold(body, CheckpointStatusUnknown)
}

func validateSummaryEnvelope(raw string) (string, []string) {
	raw = strings.TrimSpace(raw)
	captures := collectSummaryCaptures(raw)
	if len(captures) != 1 {
		return "", []string{fmt.Sprintf("expected exactly one summary block, got %d", len(captures))}
	}
	capture := captures[0]
	if strings.TrimSpace(raw[:capture.start]) != "" || strings.TrimSpace(raw[capture.end:]) != "" {
		return "", []string{"summary output contains text outside the summary block"}
	}
	summary := strings.TrimSpace(capture.content)
	if summary == "" {
		return "", []string{"summary block is empty"}
	}
	return summary, nil
}

func validateRequiredSummarySections(summary string) []string {
	type locatedSection struct {
		title string
		start int
		end   int
	}
	sections := make([]locatedSection, 0, len(requiredSummarySections))
	reasons := []string{}
	last := -1
	missing := 0
	for _, title := range requiredSummarySections {
		exactPattern := regexp.MustCompile(`(?m)^## ` + regexp.QuoteMeta(title) + `\r?$`)
		exactMatches := exactPattern.FindAllStringIndex(summary, -1)
		broadPattern := regexp.MustCompile(`(?mi)^#{1,6}[\t ]*` + regexp.QuoteMeta(title) + `[\t ]*\r?$`)
		broadMatches := broadPattern.FindAllStringIndex(summary, -1)
		if len(exactMatches) == 0 {
			missing++
			reasons = append(reasons, "missing required section: "+title)
			if len(broadMatches) > 0 {
				reasons = append(reasons, "required section must use exact level-2 heading: ## "+title)
			}
			continue
		}
		if len(exactMatches) > 1 {
			reasons = append(reasons, "duplicate required section: "+title)
		}
		if len(broadMatches) != len(exactMatches) {
			reasons = append(reasons, "required section must use exact level-2 heading: ## "+title)
		}
		loc := exactMatches[0]
		if loc[0] <= last {
			reasons = append(reasons, "required section out of order: "+title)
		}
		last = loc[0]
		sections = append(sections, locatedSection{title: title, start: loc[0], end: loc[1]})
	}
	if missing == len(requiredSummarySections) {
		reasons = append(reasons, "required sections must use exact Markdown heading lines (for example: ## Current Objective and Latest User Request)")
	}
	for i, section := range sections {
		bodyEnd := len(summary)
		if nextHeading := regexp.MustCompile(`(?m)^#{1,6}[\t ]+.+\r?$`).FindStringIndex(summary[section.end:]); nextHeading != nil {
			bodyEnd = section.end + nextHeading[0]
		} else if i+1 < len(sections) {
			bodyEnd = sections[i+1].start
		}
		if strings.TrimSpace(summary[section.end:bodyEnd]) == "" {
			reasons = append(reasons, "empty required section: "+section.title)
		}
	}
	return reasons
}
