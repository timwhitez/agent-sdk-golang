package anthropic

import (
	"fmt"
	"slices"
	"strconv"
	"strings"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

// ThinkingDisableMode describes how an Anthropic model disables extended
// thinking. Callers must not conflate omission, explicit disabled, and
// between_tools.
type ThinkingDisableMode string

const (
	// ThinkingDisableOmit disables thinking by omitting the thinking parameter
	// (models whose extended thinking is off by default).
	ThinkingDisableOmit ThinkingDisableMode = "omit"

	// ThinkingDisableExplicit disables thinking by sending {"type":"disabled"}
	// (default-on adaptive models that support full disable).
	ThinkingDisableExplicit ThinkingDisableMode = "disabled"

	// ThinkingDisableBetweenTools indicates the model only supports
	// {"type":"between_tools"}, which turns off pre-response thinking but may
	// still think between tool calls; it cannot guarantee full DisableThinking.
	ThinkingDisableBetweenTools ThinkingDisableMode = "between_tools"

	// ThinkingDisableUnsupported indicates the model does not support disabling
	// extended thinking.
	ThinkingDisableUnsupported ThinkingDisableMode = "unsupported"
)

var (
	effortsThroughHigh  = []string{"low", "medium", "high"}
	effortsThroughMax   = []string{"low", "medium", "high", "max"}
	effortsWithXHigh    = []string{"low", "medium", "high", "xhigh", "max"}
	disableUpToHighList = []string{"low", "medium", "high"}
)

// ModelCapabilities describes Anthropic thinking, effort, disable, and
// forced tool_choice support for a concrete model or gateway override.
type ModelCapabilities struct {
	// AdaptiveThinking reports whether the model uses adaptive thinking
	// (thinking.type="adaptive" + output_config.effort) rather than manual
	// budget_tokens by default.
	AdaptiveThinking bool

	// DefaultThinkingOn reports whether extended thinking is enabled by default
	// when the thinking parameter is omitted.
	DefaultThinkingOn bool

	// SupportedEfforts lists the valid output_config.effort values for this
	// model. An empty slice falls back to all standard wire effort levels
	// (low, medium, high, xhigh, max).
	SupportedEfforts []string

	// DisableMode describes how extended thinking can be disabled on this
	// model. Empty defaults to ThinkingDisableUnsupported when
	// DefaultThinkingOn is true, otherwise ThinkingDisableOmit.
	DisableMode ThinkingDisableMode

	// DisableAllowedEfforts, when non-empty, restricts which
	// output_config.effort values are legal alongside thinking.type="disabled"
	// (for example Claude Opus 5 and Haiku 5.5 allow disabled only at high and
	// below, not xhigh or max).
	DisableAllowedEfforts []string

	// ForcedToolChoiceWhenThinkingDisabled reports whether forced tool_choice
	// ("any" / specific tool) is supported once thinking is fully disabled.
	ForcedToolChoiceWhenThinkingDisabled bool
}

// Clone returns an owned copy of ModelCapabilities.
func (caps ModelCapabilities) Clone() ModelCapabilities {
	cloned := caps
	if caps.SupportedEfforts != nil {
		cloned.SupportedEfforts = append([]string(nil), caps.SupportedEfforts...)
	}
	if caps.DisableAllowedEfforts != nil {
		cloned.DisableAllowedEfforts = append([]string(nil), caps.DisableAllowedEfforts...)
	}
	return cloned
}

func (caps ModelCapabilities) effectiveDisableMode() ThinkingDisableMode {
	if caps.DisableMode != "" {
		return caps.DisableMode
	}
	if caps.DefaultThinkingOn {
		return ThinkingDisableUnsupported
	}
	return ThinkingDisableOmit
}

func (caps ModelCapabilities) effectiveSupportedEfforts() []string {
	if len(caps.SupportedEfforts) == 0 {
		return effortsWithXHigh
	}
	return caps.SupportedEfforts
}

// SupportsEffort reports whether the normalized effort level is supported by
// this model.
func (caps ModelCapabilities) SupportsEffort(effort string) bool {
	normalized := strings.ToLower(strings.TrimSpace(effort))
	if normalized == "" {
		return false
	}
	return slices.Contains(caps.effectiveSupportedEfforts(), normalized)
}

// SupportsFullDisable reports whether the model can guarantee that extended
// thinking is fully disabled for a request.
func (caps ModelCapabilities) SupportsFullDisable() bool {
	mode := caps.effectiveDisableMode()
	return mode == ThinkingDisableOmit || mode == ThinkingDisableExplicit
}

// SupportsBetweenTools reports whether the model supports
// thinking.type="between_tools".
func (caps ModelCapabilities) SupportsBetweenTools() bool {
	return caps.effectiveDisableMode() == ThinkingDisableBetweenTools
}

// SupportsDisableWithEffort reports whether effort is valid alongside explicit
// thinking.type="disabled".
func (caps ModelCapabilities) SupportsDisableWithEffort(effort string) bool {
	if !caps.SupportsFullDisable() {
		return false
	}
	normalized := strings.ToLower(strings.TrimSpace(effort))
	if normalized == "" {
		return true
	}
	if !caps.SupportsEffort(normalized) {
		return false
	}
	if len(caps.DisableAllowedEfforts) == 0 {
		return true
	}
	return slices.Contains(caps.DisableAllowedEfforts, normalized)
}

type claudeModelID struct {
	known  bool
	family string
	major  int
	minor  int
}

// CapabilitiesForModel returns the inferred Anthropic thinking capabilities for
// modelName. Callers using custom gateway aliases can override these via
// Client.Capabilities.
func CapabilitiesForModel(modelName string) ModelCapabilities {
	id := parseClaudeModelID(modelName)
	if !id.known {
		// Unknown/custom gateway alias: keep backward-compatible default-off
		// disable behavior and allow all official wire effort tiers unless an
		// explicit Client.Capabilities override is supplied.
		return ModelCapabilities{
			AdaptiveThinking:                     false,
			DefaultThinkingOn:                    false,
			SupportedEfforts:                     append([]string(nil), effortsWithXHigh...),
			DisableMode:                          ThinkingDisableOmit,
			ForcedToolChoiceWhenThinkingDisabled: true,
		}
	}

	if id.family == "mythos" || id.family == "fable" {
		return ModelCapabilities{
			AdaptiveThinking:                     true,
			DefaultThinkingOn:                    true,
			SupportedEfforts:                     append([]string(nil), effortsWithXHigh...),
			DisableMode:                          ThinkingDisableUnsupported,
			ForcedToolChoiceWhenThinkingDisabled: false,
		}
	}

	if id.major >= 5 {
		caps := ModelCapabilities{
			AdaptiveThinking:                     true,
			DefaultThinkingOn:                    true,
			SupportedEfforts:                     append([]string(nil), effortsWithXHigh...),
			DisableMode:                          ThinkingDisableUnsupported,
			ForcedToolChoiceWhenThinkingDisabled: false,
		}
		if id.major == 5 {
			switch {
			case id.family == "sonnet" && id.minor == 0:
				caps.DisableMode = ThinkingDisableExplicit
				caps.ForcedToolChoiceWhenThinkingDisabled = true
			case id.family == "opus" && id.minor == 0:
				caps.DisableMode = ThinkingDisableExplicit
				caps.DisableAllowedEfforts = append([]string(nil), disableUpToHighList...)
				caps.ForcedToolChoiceWhenThinkingDisabled = true
			case id.family == "haiku" && (id.minor == 0 || id.minor == 5):
				caps.DisableMode = ThinkingDisableExplicit
				caps.DisableAllowedEfforts = append([]string(nil), disableUpToHighList...)
				caps.ForcedToolChoiceWhenThinkingDisabled = true
			case id.family == "sonnet" && id.minor == 5:
				caps.DisableMode = ThinkingDisableBetweenTools
				caps.ForcedToolChoiceWhenThinkingDisabled = false
			case id.family == "opus" && id.minor == 5:
				caps.DisableMode = ThinkingDisableUnsupported
				caps.ForcedToolChoiceWhenThinkingDisabled = false
			}
		}
		return caps
	}

	if id.major == 4 {
		switch {
		case id.family == "opus" && (id.minor == 7 || id.minor == 8):
			return ModelCapabilities{
				AdaptiveThinking:                     true,
				DefaultThinkingOn:                    false,
				SupportedEfforts:                     append([]string(nil), effortsWithXHigh...),
				DisableMode:                          ThinkingDisableOmit,
				ForcedToolChoiceWhenThinkingDisabled: true,
			}
		case id.family == "opus" && id.minor == 5:
			return ModelCapabilities{
				AdaptiveThinking:                     false,
				DefaultThinkingOn:                    false,
				SupportedEfforts:                     append([]string(nil), effortsThroughHigh...),
				DisableMode:                          ThinkingDisableOmit,
				ForcedToolChoiceWhenThinkingDisabled: true,
			}
		default:
			// Claude Opus 4.6, Sonnet 4.6, and other Claude 4.x models do not
			// support native xhigh.
			return ModelCapabilities{
				AdaptiveThinking:                     false,
				DefaultThinkingOn:                    false,
				SupportedEfforts:                     append([]string(nil), effortsThroughMax...),
				DisableMode:                          ThinkingDisableOmit,
				ForcedToolChoiceWhenThinkingDisabled: true,
			}
		}
	}

	// Claude 3.x and older default-off models do not support xhigh.
	return ModelCapabilities{
		AdaptiveThinking:                     false,
		DefaultThinkingOn:                    false,
		SupportedEfforts:                     append([]string(nil), effortsThroughMax...),
		DisableMode:                          ThinkingDisableOmit,
		ForcedToolChoiceWhenThinkingDisabled: true,
	}
}

func parseClaudeModelID(modelName string) claudeModelID {
	normalized := strings.ToLower(strings.TrimSpace(modelName))
	if normalized == "" {
		return claudeModelID{}
	}
	tokens := strings.FieldsFunc(normalized, func(r rune) bool {
		return (r < 'a' || r > 'z') && (r < '0' || r > '9')
	})
	var matched claudeModelID
	for i := 0; i < len(tokens); i++ {
		if tokens[i] != "claude" {
			continue
		}
		if i+1 < len(tokens) && isASCIILetters(tokens[i+1]) {
			family := tokens[i+1]
			if family == "mythos" || family == "fable" {
				candidate := claudeModelID{known: true, family: family, major: 5, minor: 0}
				if i+2 < len(tokens) && len(tokens[i+2]) == 1 && isASCIIDigits(tokens[i+2]) {
					candidate.major = int(tokens[i+2][0] - '0')
					if i+3 < len(tokens) && isASCIIDigits(tokens[i+3]) && len(tokens[i+3]) <= 2 {
						if minor, err := strconv.Atoi(tokens[i+3]); err == nil {
							candidate.minor = minor
						}
					}
				}
				matched = candidate
				continue
			}
			if i+2 < len(tokens) && len(tokens[i+2]) == 1 && isASCIIDigits(tokens[i+2]) {
				candidate := claudeModelID{
					known:  true,
					family: family,
					major:  int(tokens[i+2][0] - '0'),
					minor:  0,
				}
				if i+3 < len(tokens) && isASCIIDigits(tokens[i+3]) && len(tokens[i+3]) <= 2 {
					if minor, err := strconv.Atoi(tokens[i+3]); err == nil {
						candidate.minor = minor
					}
				}
				matched = candidate
				continue
			}
			if i+2 < len(tokens) && len(tokens[i+2]) == 2 && isASCIIDigits(tokens[i+2]) && tokens[i+2][0] == '4' {
				matched = claudeModelID{
					known:  true,
					family: family,
					major:  4,
					minor:  int(tokens[i+2][1] - '0'),
				}
				continue
			}
		}
		if i+2 < len(tokens) && len(tokens[i+1]) == 1 && isASCIIDigits(tokens[i+1]) {
			major := int(tokens[i+1][0] - '0')
			if isASCIILetters(tokens[i+2]) {
				matched = claudeModelID{
					known:  true,
					family: tokens[i+2],
					major:  major,
					minor:  0,
				}
				continue
			}
			if i+3 < len(tokens) && isASCIIDigits(tokens[i+2]) && len(tokens[i+2]) <= 2 && isASCIILetters(tokens[i+3]) {
				if minor, err := strconv.Atoi(tokens[i+2]); err == nil {
					matched = claudeModelID{
						known:  true,
						family: tokens[i+3],
						major:  major,
						minor:  minor,
					}
					continue
				}
			}
		}
	}
	return matched
}

func isASCIILetters(s string) bool {
	if s == "" {
		return false
	}
	for _, r := range s {
		if r < 'a' || r > 'z' {
			return false
		}
	}
	return true
}

func isASCIIDigits(s string) bool {
	if s == "" {
		return false
	}
	for _, r := range s {
		if r < '0' || r > '9' {
			return false
		}
	}
	return true
}

func validateAnthropicEffort(modelName string, caps ModelCapabilities, effort string) (string, error) {
	normalized := strings.ToLower(strings.TrimSpace(effort))
	supported := caps.effectiveSupportedEfforts()
	switch normalized {
	case "low", "medium", "high", "xhigh", "max":
	default:
		return "", &llm.ThinkingCapabilityError{
			Provider: "anthropic",
			Model:    modelName,
			Control:  "effort",
			Value:    effort,
			Message:  fmt.Sprintf("unsupported adaptive thinking effort %q for model %q (supported: %s)", effort, modelName, strings.Join(supported, ", ")),
		}
	}
	if !caps.SupportsEffort(normalized) {
		return "", &llm.ThinkingCapabilityError{
			Provider: "anthropic",
			Model:    modelName,
			Control:  "effort",
			Value:    normalized,
			Message:  fmt.Sprintf("adaptive thinking effort %q is not supported by model %q (supported: %s)", normalized, modelName, strings.Join(supported, ", ")),
		}
	}
	return normalized, nil
}

func validateAnthropicDisableEffort(modelName string, caps ModelCapabilities, effort string) (string, error) {
	normalized, err := validateAnthropicEffort(modelName, caps, effort)
	if err != nil {
		return "", err
	}
	if !caps.SupportsDisableWithEffort(normalized) {
		return "", &llm.ThinkingCapabilityError{
			Provider: "anthropic",
			Model:    modelName,
			Control:  "disable_thinking",
			Value:    normalized,
			Message:  fmt.Sprintf("thinking mode \"disabled\" is incompatible with effort %q on model %q (allowed: %s)", normalized, modelName, strings.Join(caps.DisableAllowedEfforts, ", ")),
		}
	}
	return normalized, nil
}
