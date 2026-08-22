package compaction

import (
	"strings"
)

const (
	recentUserMaterialTokenBudget  = 400
	keyUserMaterialTokenBudget     = 200
	keyEventMaterialTokenBudget    = 150
	systemContextTokenBudget       = 500
	previousSummaryTokenBudget     = 1000
	summaryDeltaTokenBudget        = 300
	toolSnapshotEntryTokenBudget   = 75
	assistantPreviewTokenBudget    = 55
	compactionTruncationMarker     = " [...truncated for compaction token budget...]"
	compactTruncationMarker        = " [truncated]"
	preservedIdentifierLabel       = "\nPreserved identifiers: "
	invalidUTF8OmissionPlaceholder = "[invalid UTF-8 omitted]"
)

type tokenEstimator func(string) int

func truncateTextToTokenBudget(text string, budget int, estimate tokenEstimator) string {
	if budget <= 0 {
		return ""
	}
	text = strings.TrimSpace(strings.ToValidUTF8(text, invalidUTF8OmissionPlaceholder))
	if text == "" {
		return ""
	}
	estimate = normalizedTokenEstimator(estimate)
	if estimate(text) <= budget {
		return text
	}

	marker := compactionTruncationMarker
	if estimate(marker) > budget {
		marker = compactTruncationMarker
	}
	if estimate(marker) > budget {
		return ""
	}

	identifierBlock := preservedIdentifierBlock(text, budget, marker, estimate)
	suffix := marker + identifierBlock
	if estimate(suffix) > budget {
		identifierBlock = ""
		suffix = marker
	}

	runes := []rune(text)
	lo, hi := 0, len(runes)
	best := suffix
	for lo <= hi {
		mid := (lo + hi) / 2
		head := strings.TrimRight(string(runes[:mid]), " \n\t")
		candidate := head + suffix
		if estimate(candidate) <= budget {
			best = candidate
			lo = mid + 1
		} else {
			hi = mid - 1
		}
	}
	return best
}

func preservedIdentifierBlock(text string, budget int, marker string, estimate tokenEstimator) string {
	identifierBudget := budget / 3
	if identifierBudget <= 0 {
		return ""
	}
	if identifierBudget > 128 {
		identifierBudget = 128
	}
	tokens := extractKeyTokens(text, 12)
	if len(tokens) == 0 {
		return ""
	}
	kept := make([]string, 0, len(tokens))
	for _, token := range tokens {
		candidateTokens := append(append([]string(nil), kept...), token)
		block := preservedIdentifierLabel + strings.Join(candidateTokens, " ")
		if estimate(block) > identifierBudget || estimate(marker+block) > budget {
			continue
		}
		kept = candidateTokens
	}
	if len(kept) == 0 {
		return ""
	}
	return preservedIdentifierLabel + strings.Join(kept, " ")
}

func normalizedTokenEstimator(estimate tokenEstimator) tokenEstimator {
	if estimate == nil {
		return approximateTextTokens
	}
	return func(text string) int {
		if strings.TrimSpace(text) == "" {
			return 0
		}
		if tokens := estimate(text); tokens > 0 {
			return tokens
		}
		return approximateTextTokens(text)
	}
}

func extractKeyTokens(text string, maxItems int) []string {
	raw := strings.FieldsFunc(text, func(r rune) bool {
		return r == ' ' || r == '\n' || r == '\t' || r == ',' || r == ';' || r == ':' || r == ')' || r == '(' || r == '[' || r == ']'
	})
	out := []string{}
	seen := map[string]struct{}{}
	for _, tok := range raw {
		tok = strings.Trim(tok, "\"'`")
		if tok == "" {
			continue
		}
		isKey := strings.Contains(tok, "/") || strings.Contains(tok, "\\") || strings.Contains(tok, ".") || strings.Contains(tok, "=") || strings.Contains(strings.ToLower(tok), "error")
		if !isKey {
			continue
		}
		if _, ok := seen[tok]; ok {
			continue
		}
		seen[tok] = struct{}{}
		out = append(out, tok)
		if len(out) >= maxItems {
			break
		}
	}
	return out
}
