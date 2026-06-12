# Providers and Streaming

This document explains provider implementations, compatibility fallbacks,
stream normalization, and response metadata behavior.

## Shared LLM Contracts
- Provider-neutral interfaces: `ChatModel`, `StreamingChatModel` (`sdk/llm/model.go:8`, `sdk/llm/model.go:17`)
- Unified request envelope: `InvokeRequest` (messages, tools, tool choice, temperature, responses options) (`sdk/llm/model.go:82`)
- Unified stream event union: text/thinking/tool-call deltas, usage, done, response metadata, errors (`sdk/llm/model.go:24`, `sdk/llm/model.go:28`, `sdk/llm/model.go:33`, `sdk/llm/model.go:39`, `sdk/llm/model.go:50`, `sdk/llm/model.go:55`, `sdk/llm/model.go:62`, `sdk/llm/model.go:70`)
- Shared message and completion model: `ToolChoice`, `ToolCall`, `Message`, `Completion`, `Usage` (`sdk/llm/types.go:17`, `sdk/llm/types.go:37`, `sdk/llm/types.go:100`, `sdk/llm/types.go:133`, `sdk/llm/types.go:124`)

## Responses API Option Model
- `ResponsesOptions` carries response-item/instruction toggles, text formatting, reasoning controls, and behavior flags (`sdk/llm/responses_options.go:23`)
- Structured text controls: verbosity and output schema hooks (`sdk/llm/responses_options.go:3`, `sdk/llm/responses_options.go:11`)
- Reasoning controls: effort and summary mode (`sdk/llm/responses_options.go:17`)

## OpenAI Chat Completions
- Non-stream Chat parsing now synthesizes stable `tool_call_id` values (`call_N`) when compatible gateways omit them, preserving existing IDs and preventing invalid follow-up `tool` messages in multi-turn tool loops.
- Request building rejects invalid assistant tool-call/tool-result history locally with an `invalid tool history` error before sending to OpenAI-compatible endpoints.
- `ChatClient.ProviderLabel` can override the default `"openai"` provider label;
  `Provider()` and provider/rate-limit errors use the label when set.
- Compatibility downgrade retries on 400/422 by disabling unsupported fields (`sdk/llm/openai/chat.go:119`, `sdk/llm/openai/chat.go:128`, `sdk/llm/openai/chat.go:134`)
- Streaming parser emits normalized text/thinking/tool-call deltas plus usage and stop reason (`sdk/llm/openai/chat.go:285`, `sdk/llm/openai/chat.go:301`, `sdk/llm/openai/chat.go:313`, `sdk/llm/openai/chat.go:316`, `sdk/llm/openai/chat.go:322`)
- OpenAI Chat + Responses now share one retry policy normalizer (default attempts/base/max) and crypto-random 10% jitter source for backoff timing consistency (`sdk/llm/openai/retry_policy.go:9`, `sdk/llm/openai/chat.go:64`, `sdk/llm/openai/responses.go:62`). The default provider retryable status set is 401, 403, 408, 409, 425, 429, and any 5xx status, with explicit per-client override through `RetryableStatusCodes`.
- Stream decode failures now include provider, HTTP status, model, and endpoint context for easier multi-provider debugging (`sdk/llm/openai/chat.go:296`)
- `parallel_tool_calls` is omitted unless explicitly enabled on the chat client, improving compatibility with strict OpenAI-compatible gateways (`sdk/llm/openai/chat.go:783`)
- Strict tool schema conversion recurses through nested object properties and array item schemas, including arrays of objects and nested arrays, before Chat or Responses sends tool definitions to OpenAI-compatible gateways (`sdk/llm/openai/chat.go:1009`, `sdk/llm/openai/responses.go:1506`)
- Endpoint resolution now only treats `/api/vN` (numeric version segment) as pre-versioned, so paths like `/api/openai` or `/api/v2beta` still receive `/v1/` (`sdk/llm/openai/chat.go:445`, `sdk/llm/openai/chat.go:466`)
- Tool choice mapping supports auto/none/required and explicit forced function name (`sdk/llm/openai/chat.go:733`)
- Content serialization for chat requests is text-focused (`sdk/llm/openai/chat.go:791`)

## OpenAI Responses API
- Compatibility downgrade retries also strip unsupported extras and can force string `input.content` when content parts are rejected (`sdk/llm/openai/responses.go:126`, `sdk/llm/openai/responses.go:133`, `sdk/llm/openai/responses.go:139`, `sdk/llm/openai/responses.go:144`, `sdk/llm/openai/responses.go:151`)
- `ResponsesClient.ProviderLabel` mirrors Chat behavior so callers can preserve
  transport-specific labels such as `openai-responses` in diagnostics.
- Request building shares the same local tool-history validation as Chat, preventing orphan `function_call_output` items from being sent.
- Responses item mode serializes tool-error `function_call_output.output` as a string with an error marker. The API accepts strings or content-part arrays there, not structured `{content, success}` objects.
- Auto-compat mode uses staged fallback between full and legacy request shapes (`sdk/llm/openai/responses.go:77`, `sdk/llm/openai/responses.go:420`)
- Streaming parser supports output-text deltas, reasoning deltas, function-call argument deltas, usage, completion, and response metadata (`sdk/llm/openai/responses.go:568`, `sdk/llm/openai/responses.go:580`, `sdk/llm/openai/responses.go:628`, `sdk/llm/openai/responses.go:642`, `sdk/llm/openai/responses.go:651`, `sdk/llm/openai/responses.go:655`, `sdk/llm/openai/responses.go:660`, `sdk/llm/openai/responses.go:675`)
- Tool choice intentionally omits explicit `auto` for compatibility, while preserving `none`/`required`/forced tool modes (`sdk/llm/openai/responses.go:1085`)
- Parsed response IDs are attached to `Completion.ResponseID` (`sdk/llm/openai/responses.go:1272`)

## Anthropic Messages API
- Compatibility retries on 400/422 downgrade unsupported beta/thinking options, including one final-attempt retry and structured code/param detection (`sdk/llm/anthropic/client.go:125`, `sdk/llm/anthropic/client.go:135`, `sdk/llm/anthropic/client.go:215`, `sdk/llm/anthropic/client.go:244`, `sdk/llm/anthropic/client.go:621`, `sdk/llm/anthropic/client.go:631`)
- Backoff jitter now uses `crypto/rand` with a time-based fallback, avoiding deterministic jitter on older runtimes (`sdk/llm/anthropic/client.go:26`, `sdk/llm/anthropic/client.go:220`). The default retryable status set is 401, 403, 408, 409, 425, 429, and any 5xx status, with explicit per-client override through `RetryableStatusCodes`.
- Streaming parser maps content-block events into normalized text/thinking/tool deltas, preserves whitespace-only thinking deltas, emits response metadata from `message_start` plus `message_delta/message_stop` fallback IDs, and emits usage on message completion (`sdk/llm/anthropic/client.go:718`, `sdk/llm/anthropic/client.go:739`, `sdk/llm/anthropic/client.go:754`)
- SSE consumption now buffers malformed premature boundaries and surfaces malformed payload errors instead of silently dropping fragments (`sdk/llm/anthropic/client.go:757`)
- Non-positive numeric `Retry-After` values are ignored with a warning hook instead of failing silently (`sdk/llm/anthropic/client.go:422`)
- Tool choice mapping aligns `required` with Anthropic `any`, supports `auto`/`none`, and forced named tool mode (`sdk/llm/anthropic/client.go:887`)
- Tool-call ID normalization now logs both original and sanitized IDs when characters are rewritten for Anthropic compatibility (`sdk/llm/anthropic/client.go:521`)

## Agent-Side Stream Normalization
- `invokeCompletion` in the agent consumes provider-specific stream events, emits provider-agnostic agent events, and preserves stream response metadata (`sdk/agent/agent.go:597`, `sdk/agent/agent.go:616`, `sdk/agent/agent.go:644`)
- Agent-level invoke retry treats typed `RateLimitError`/`ProviderError`,
  transient network/timeout failures, and equivalent text-only gateway errors
  as retryable before visible output. Text-only status detection covers 401,
  403, 408, 409, 425, 429, and 5xx statuses so adapters that cannot preserve
  typed provider errors still get bounded backoff.
- Metadata-only stream events (`response_id`, usage, and done markers) are
  buffered until visible text/thinking/tool-call deltas. If a retryable
  provider error arrives first, the agent can retry without leaking failed
  attempt metadata or requiring the user to send a manual continuation.
- Response-level provider metadata is captured into `Completion.ResponseID` and surfaced to downstream consumers via `UsageEvent` / `AutoContinueEvent` / `FinalResponseEvent` (`sdk/agent/agent.go:255`, `sdk/agent/agent.go:285`, `sdk/agent/agent.go:348`, `sdk/agent/events.go:83`)
- `StreamRetryEvent` is non-terminal retry progress; the agent converts it to
  `WarnEvent{Kind:"rate_limit_retry"}` and continues consuming the same stream.

## Response ID Semantics
- `Completion.ResponseID` is optional at the shared type level (`sdk/llm/types.go:139`)
- OpenAI Responses populates IDs in both sync and streaming paths (`sdk/llm/openai/responses.go:651`, `sdk/llm/openai/responses.go:1272`)
- Anthropic populates IDs in both sync and streaming paths (`sdk/llm/anthropic/client.go:718`, `sdk/llm/anthropic/client.go:1219`)
- OpenAI Chat completions currently do not expose response IDs in parsed completion objects (`sdk/llm/openai/chat.go:904`)

## Usage and Optional Cost Calculation
- Providers populate `Usage`; downstream cost calculation is optional (`sdk/llm/types.go:124`, `sdk/tokens/cost.go:80`)
- Cost helper fetches LiteLLM pricing data, caches for 24h, warns on pricing-init/cache-read/cache-parse/cache-write/cost-calc failures while keeping non-fatal usage-only fallback, and tracks usage history (`sdk/tokens/cost.go:20`, `sdk/tokens/cost.go:22`, `sdk/tokens/cost.go:99`, `sdk/tokens/cost.go:108`, `sdk/tokens/cost.go:126`, `sdk/tokens/cost.go:173`, `sdk/tokens/cost.go:217`)
- Pricing lookup now supports case-insensitive names, common aliases, provider-prefix probing, and base-family fallback (choosing latest/date-tagged variants when exact names are unavailable) (`sdk/tokens/cost.go:292`, `sdk/tokens/cost.go:310`, `sdk/tokens/cost.go:392`)
- Cost math separates cached vs non-cached prompt tokens and output tokens (`sdk/tokens/cost.go:243`, `sdk/tokens/cost.go:255`, `sdk/tokens/cost.go:263`)
