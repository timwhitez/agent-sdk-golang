package llm

import "fmt"

// ThinkingCapabilities describes request-level thinking control and forced
// tool_choice compatibility implemented by a concrete model client.
//
// Models that do not advertise ThinkingCapabilityProvider retain legacy
// behavior (callers may attempt DisableThinking and forced tool_choice).
// Concrete clients report exact model capabilities so callers such as the
// agent's RequireDone recovery loop avoid unsupported DisableThinking or
// forced tool_choice requests and avoid recording untruthful provenance.
type ThinkingCapabilities struct {
	// SupportsDisableThinking reports whether InvokeRequest.DisableThinking=true
	// can guarantee that extended thinking is fully disabled for a single call.
	SupportsDisableThinking bool

	// SupportsForcedToolChoice reports whether forced tool_choice ("required" or
	// a specific tool name) is valid on this model when thinking is disabled (or
	// when thinking is not active).
	SupportsForcedToolChoice bool

	// DisableThinkingReason optionally explains why full thinking disable is
	// unavailable on this model.
	DisableThinkingReason string
}

// ThinkingCapabilityProvider is an optional interface implemented by a
// ChatModel (or wrapper delegating to one) that can accurately report its
// thinking-disable and forced-tool capabilities.
//
// Return ok=false when the underlying model does not advertise capabilities;
// callers then retain legacy behavior.
type ThinkingCapabilityProvider interface {
	ThinkingCapabilities() (caps ThinkingCapabilities, ok bool)
}

// ModelThinkingCapabilities queries model for ThinkingCapabilityProvider.
// Returns ok=false when model is nil or does not advertise capabilities.
func ModelThinkingCapabilities(model ChatModel) (ThinkingCapabilities, bool) {
	if nilFrameModel(model) {
		return ThinkingCapabilities{}, false
	}
	provider, ok := model.(ThinkingCapabilityProvider)
	if !ok {
		return ThinkingCapabilities{}, false
	}
	return provider.ThinkingCapabilities()
}

// ThinkingCapabilityError reports a local capability mismatch before sending a
// provider request (for example an unsupported effort level, an unsupported
// DisableThinking request, or an incompatible thinking mode/effort pair).
type ThinkingCapabilityError struct {
	Provider string
	Model    string
	Control  string
	Value    string
	Message  string
}

func (e *ThinkingCapabilityError) Error() string {
	if e == nil {
		return "<nil>"
	}
	provider := e.Provider
	if provider == "" {
		provider = "llm"
	}
	if e.Message != "" {
		return fmt.Sprintf("%s: %s", provider, e.Message)
	}
	if e.Model != "" {
		return fmt.Sprintf("%s: unsupported thinking capability %s=%q for model %q", provider, e.Control, e.Value, e.Model)
	}
	return fmt.Sprintf("%s: unsupported thinking capability %s=%q", provider, e.Control, e.Value)
}
