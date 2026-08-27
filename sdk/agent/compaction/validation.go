package compaction

import (
	"encoding/json"
	"fmt"
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
		facts, err := summaryValidationFactsFromMaterial(material)
		if err != nil {
			reasons = append(reasons, "compaction material framing is invalid: "+err.Error())
		} else {
			reasons = append(reasons, validateSummaryFactCoverage(summary, facts)...)
			if facts.HostCheckpointStatus == CheckpointStatusUnknown && !strings.Contains(strings.ToUpper(summary), CheckpointStatusUnknown) {
				reasons = append(reasons, "host state is UNKNOWN but summary does not preserve UNKNOWN")
			}
		}
	}
	if len(reasons) > 0 {
		sort.Strings(reasons)
		return "", &summaryValidationError{reasons: reasons}
	}
	return summary, nil
}

type summaryValidationFacts struct {
	LatestRealUserRequest *string
	HostCheckpointStatus  string
}

func summaryValidationFactsFromMaterial(material string) (summaryValidationFacts, error) {
	envelope, framed, err := decodeCompactionMaterialEnvelope(material)
	if err != nil {
		return summaryValidationFacts{}, err
	}
	if framed {
		return summaryValidationFacts{
			LatestRealUserRequest: cloneStringPointer(envelope.LatestRealUserRequest),
			HostCheckpointStatus:  normalizeCompactionHostCheckpointStatus(envelope.HostCheckpointStatus),
		}, nil
	}
	latest := materialSectionBody(envelope.Material, "Latest Real User Request")
	return summaryValidationFacts{
		LatestRealUserRequest: &latest,
		HostCheckpointStatus:  legacyHostCheckpointStatus(envelope.Material),
	}, nil
}

func legacyHostCheckpointStatus(material string) string {
	if !strings.Contains(material, "## Host Checkpoint Context") {
		return ""
	}
	if strings.Contains(material, "Status: "+CheckpointStatusUnknown) {
		return CheckpointStatusUnknown
	}
	if strings.Contains(material, "Status: "+CheckpointStatusVerified) {
		return CheckpointStatusVerified
	}
	return ""
}

func validateSummaryFactCoverage(summary string, facts summaryValidationFacts) []string {
	reasons := []string{}
	latest := ""
	if facts.LatestRealUserRequest != nil {
		latest = *facts.LatestRealUserRequest
	}
	objective := materialSectionBody(summary, "Current Objective and Latest User Request")
	if strings.TrimSpace(latest) != "" && strings.ToUpper(strings.TrimSpace(latest)) != CheckpointStatusUnknown && summaryBodyIsOnlyUnknown(objective) {
		reasons = append(reasons, "latest user request was supplied but Current Objective and Latest User Request is UNKNOWN")
	}
	if facts.HostCheckpointStatus == CheckpointStatusVerified {
		external := materialSectionBody(summary, "Exact External State")
		if summaryBodyIsOnlyUnknown(external) {
			reasons = append(reasons, "verified host state was supplied but Exact External State is UNKNOWN")
		}
	}
	return reasons
}

func decodeSummaryValidationMaterial(material string) (string, error) {
	hasBegin := strings.Contains(material, beginUntrustedMaterial)
	hasEnd := strings.Contains(material, endUntrustedMaterial)
	if !hasBegin && !hasEnd {
		return material, nil
	}
	prefix := beginUntrustedMaterial + "\n"
	suffix := "\n" + endUntrustedMaterial
	if !strings.HasPrefix(material, prefix) || !strings.HasSuffix(material, suffix) {
		return "", fmt.Errorf("expected exact BEGIN/END three-line frame")
	}
	encoded := strings.TrimSuffix(strings.TrimPrefix(material, prefix), suffix)
	if strings.ContainsAny(encoded, "\r\n") {
		return "", fmt.Errorf("expected one JSON-string payload line")
	}
	var value any
	if err := json.Unmarshal([]byte(encoded), &value); err != nil {
		return "", fmt.Errorf("payload is not a valid JSON string: %w", err)
	}
	decoded, ok := value.(string)
	if !ok {
		return "", fmt.Errorf("payload JSON type must be string")
	}
	return decoded, nil
}

func materialSectionBody(text, title string) string {
	headings := scanMarkdownHeadings(text)
	sectionIndex := -1
	for i, heading := range headings {
		if heading.line == "## "+title {
			sectionIndex = i
			break
		}
	}
	if sectionIndex < 0 {
		return ""
	}
	section := headings[sectionIndex]
	bodyEnd := len(text)
	if sectionIndex+1 < len(headings) {
		bodyEnd = headings[sectionIndex+1].start
	}
	rest := text[section.end:bodyEnd]
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
	headings := scanMarkdownHeadings(summary)
	last := -1
	missing := 0
	for _, title := range requiredSummarySections {
		exactMatches := make([]markdownHeading, 0, 1)
		broadMatches := make([]markdownHeading, 0, 1)
		for _, heading := range headings {
			if strings.EqualFold(heading.title, title) {
				broadMatches = append(broadMatches, heading)
			}
			if heading.line == "## "+title {
				exactMatches = append(exactMatches, heading)
			}
		}
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
		if loc.start <= last {
			reasons = append(reasons, "required section out of order: "+title)
		}
		last = loc.start
		sections = append(sections, locatedSection{title: title, start: loc.start, end: loc.end})
	}
	if missing == len(requiredSummarySections) {
		reasons = append(reasons, "required sections must use exact Markdown heading lines (for example: ## Current Objective and Latest User Request)")
	}
	for _, section := range sections {
		bodyEnd := len(summary)
		for _, heading := range headings {
			if heading.start >= section.end {
				bodyEnd = heading.start
				break
			}
		}
		if strings.TrimSpace(summary[section.end:bodyEnd]) == "" {
			reasons = append(reasons, "empty required section: "+section.title)
		}
	}
	return reasons
}

type markdownHeading struct {
	line  string
	title string
	start int
	end   int
}

func scanMarkdownHeadings(text string) []markdownHeading {
	headings := []markdownHeading{}
	fenceChar := byte(0)
	fenceWidth := 0
	for start := 0; start <= len(text); {
		lineEnd := strings.IndexByte(text[start:], '\n')
		next := len(text) + 1
		if lineEnd < 0 {
			lineEnd = len(text)
		} else {
			lineEnd += start
			next = lineEnd + 1
		}
		line := strings.TrimSuffix(text[start:lineEnd], "\r")

		if marker, width, closing := markdownFenceMarker(line, fenceChar, fenceWidth); marker != 0 {
			if fenceChar == 0 && !closing {
				fenceChar = marker
				fenceWidth = width
			} else if fenceChar == marker && closing {
				fenceChar = 0
				fenceWidth = 0
			}
		} else if fenceChar == 0 {
			headingOffset := 0
			for headingOffset < len(line) && headingOffset < 4 && line[headingOffset] == ' ' {
				headingOffset++
			}
			if headingOffset > 3 || headingOffset >= len(line) || line[headingOffset] != '#' {
				if next > len(text) {
					break
				}
				start = next
				continue
			}
			level := 0
			for headingOffset+level < len(line) && level < 6 && line[headingOffset+level] == '#' {
				level++
			}
			if level > 0 {
				title := strings.TrimSpace(line[headingOffset+level:])
				headings = append(headings, markdownHeading{line: line, title: title, start: start, end: min(next, len(text))})
			}
		}

		if next > len(text) {
			break
		}
		start = next
	}
	return headings
}

func markdownFenceMarker(line string, active byte, activeWidth int) (marker byte, width int, closing bool) {
	indent := 0
	for indent < len(line) && indent < 4 && line[indent] == ' ' {
		indent++
	}
	if indent > 3 || indent >= len(line) {
		return 0, 0, false
	}
	marker = line[indent]
	if marker != '`' && marker != '~' {
		return 0, 0, false
	}
	for indent+width < len(line) && line[indent+width] == marker {
		width++
	}
	if width < 3 {
		return 0, 0, false
	}
	if active == 0 {
		return marker, width, false
	}
	if marker != active || width < activeWidth || !onlyASCIISpaceOrTab(line[indent+width:]) {
		return 0, 0, false
	}
	return marker, width, true
}

func onlyASCIISpaceOrTab(text string) bool {
	for i := 0; i < len(text); i++ {
		if text[i] != ' ' && text[i] != '\t' {
			return false
		}
	}
	return true
}
