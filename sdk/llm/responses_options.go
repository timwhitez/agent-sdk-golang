package llm

// ResponsesTextFormat describes a structured text output format (e.g., JSON schema).
type ResponsesTextFormat struct {
	Type   string         `json:"type,omitempty"`   // e.g. "json_schema"
	Strict bool           `json:"strict,omitempty"` // enforce schema strictly
	Schema map[string]any `json:"schema,omitempty"`
	Name   string         `json:"name,omitempty"`
}

// ResponsesTextControls configures verbosity/format for Responses API text output.
type ResponsesTextControls struct {
	Verbosity string               `json:"verbosity,omitempty"` // "low"|"medium"|"high"
	Format    *ResponsesTextFormat `json:"format,omitempty"`
}

// ResponsesReasoning configures reasoning options for the Responses API.
type ResponsesReasoning struct {
	Effort  string `json:"effort,omitempty"`
	Summary string `json:"summary,omitempty"`
}

// ResponsesOptions provides per-request options for OpenAI Responses API.
// Providers that don't support these options will ignore them.
type ResponsesOptions struct {
	// UseResponseItems enables the full Responses API input item format
	// (message/function_call/function_call_output).
	UseResponseItems *bool `json:"-"`
	// UseInstructions moves system messages to the top-level instructions field.
	UseInstructions *bool `json:"-"`
	// Instructions sets/overrides the top-level instructions field.
	Instructions string `json:"-"`

	ConversationID string `json:"conversation_id,omitempty"`
	PromptCacheKey string `json:"prompt_cache_key,omitempty"`
	Include        []string

	Text      *ResponsesTextControls `json:"text,omitempty"`
	Reasoning *ResponsesReasoning    `json:"reasoning,omitempty"`

	ParallelToolCalls *bool `json:"parallel_tool_calls,omitempty"`
	Store             *bool `json:"store,omitempty"`

	// Convenience: if Text is nil and Verbosity/OutputSchema are set, a text
	// controls object will be built automatically.
	Verbosity    string         `json:"-"`
	OutputSchema map[string]any `json:"-"`
}
