# Providers and Streaming

This document explains provider implementations, compatibility fallbacks,
stream normalization, and response metadata behavior.

## Shared LLM Contracts
- Provider-neutral interfaces: `ChatModel`, `StreamingChatModel` (`sdk/llm/model.go:8`, `sdk/llm/model.go:17`)
- Unified request envelope: `InvokeRequest` (messages, tools, tool choice, temperature, responses options) (`sdk/llm/model.go:82`)
- Unified stream event union: text/thinking/tool-call deltas, usage, done, response metadata, errors (`sdk/llm/model.go:24`, `sdk/llm/model.go:28`, `sdk/llm/model.go:33`, `sdk/llm/model.go:39`, `sdk/llm/model.go:50`, `sdk/llm/model.go:55`, `sdk/llm/model.go:62`, `sdk/llm/model.go:70`)
- Shared message and completion model: `ToolChoice`, `ToolCall`, `Message`, `Completion`, `Usage` (`sdk/llm/types.go:17`, `sdk/llm/types.go:37`, `sdk/llm/types.go:100`, `sdk/llm/types.go:133`, `sdk/llm/types.go:124`). `Completion.Diagnostics` carries non-fatal provider diagnostics such as compatibility downgrades.

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
- Successful downgrade retries append `provider_compatibility_downgrade`
  diagnostics to the returned `Completion`, so agent surfaces can warn that
  reasoning/extra/thinking options were not honored.
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
- Successful downgrade retries append `provider_compatibility_downgrade`
  diagnostics to the returned `Completion`.
- `ResponsesClient.ProviderLabel` mirrors Chat behavior so callers can preserve
  transport-specific labels such as `openai-responses` in diagnostics.
- Request building shares the same local tool-history validation as Chat, preventing orphan `function_call_output` items from being sent.
- Responses item mode serializes tool-error `function_call_output.output` as a string with an error marker. The API accepts strings or content-part arrays there, not structured `{content, success}` objects.
- Auto-compat mode uses staged fallback between full and legacy request shapes (`sdk/llm/openai/responses.go:77`, `sdk/llm/openai/responses.go:420`)
- Streaming parser supports output-text deltas, reasoning deltas, function-call argument deltas, usage, completion, and response metadata (`sdk/llm/openai/responses.go:568`, `sdk/llm/openai/responses.go:580`, `sdk/llm/openai/responses.go:628`, `sdk/llm/openai/responses.go:642`, `sdk/llm/openai/responses.go:651`, `sdk/llm/openai/responses.go:655`, `sdk/llm/openai/responses.go:660`, `sdk/llm/openai/responses.go:675`)
- Buffered and streaming terminal parsing preserves refusal text and distinguishes `max_output_tokens` from `content_filter`. Failed, cancelled, queued, in-progress, root-error, and unsupported incomplete states fail closed with typed provider errors; streaming terminals emit response ID and usage before the terminal event.
- Tool choice intentionally omits explicit `auto` for compatibility, while preserving `none`/`required`/forced tool modes (`sdk/llm/openai/responses.go:1085`)
- Parsed response IDs are attached to `Completion.ResponseID` (`sdk/llm/openai/responses.go:1272`)
- Manual/stateless continuation preserves each returned Responses output item
  in opaque `Message.ProviderState`. The state is never included in `PlainText`
  or UI output; response-item-mode requests replay it in provider order before
  matching `function_call_output` items. Legacy message mode fails closed rather
  than silently dropping opaque state. Buffered and streaming clients use the
  same path, including `id`, `call_id`, `phase`, and `encrypted_content` fields.
- Opaque Responses state is bounded to 1,024 items and 8 MiB per response and
  per outgoing manual history. It is included in token/compaction estimates and
  fails closed when externally restored state is malformed or attached to a
  non-assistant message. Tool-pair repair and compaction mutations clear stale
  opaque items whenever their matching assistant message is changed.
- For provider-managed state, set `ResponsesOptions.PreviousResponseID` or
  `ConversationID` and send only new input. Those options are mutually exclusive
  with each other and with manually replayed `ProviderState`. For manual
  stateless reasoning (for example `store=false`), request
  `reasoning.encrypted_content` through `ResponsesOptions.Include`, then retain
  the returned `ProviderState` with the assistant message. OpenAI's Responses
  guide requires prior output items to be supplied again when callers manage
  context themselves.

## Anthropic Messages API
- Extended thinking supports both manual budgets (`thinking.type="enabled"` +
  `budget_tokens`) and adaptive mode (`thinking.type="adaptive"` + optional
  `output_config.effort`). `InvokeRequest.DisableThinking` suppresses both
  shapes for recovery calls.
- While thinking is active, `tool_choice` is limited to `auto` or `none`.
  Required/specific-tool conflicts return an actionable local error instead of
  silently downgrading the caller's choice; agent-owned require-done recovery
  sets `DisableThinking` before forcing a tool.
- Compatibility retries on 400/422 downgrade unsupported beta/thinking options, including one final-attempt retry and structured code/param detection (`sdk/llm/anthropic/client.go:125`, `sdk/llm/anthropic/client.go:135`, `sdk/llm/anthropic/client.go:215`, `sdk/llm/anthropic/client.go:244`, `sdk/llm/anthropic/client.go:621`, `sdk/llm/anthropic/client.go:631`)
- Successful beta/thinking downgrades append
  `provider_compatibility_downgrade` diagnostics to the returned
  `Completion`.
- Backoff jitter now uses `crypto/rand` with a time-based fallback, avoiding deterministic jitter on older runtimes (`sdk/llm/anthropic/client.go:26`, `sdk/llm/anthropic/client.go:220`). The default retryable status set is 401, 403, 408, 409, 425, 429, and any 5xx status, with explicit per-client override through `RetryableStatusCodes`.
- Streaming parser maps content-block events into normalized text/thinking/tool
  deltas, preserves whitespace-only thinking deltas, and carries Anthropic
  thinking block indices, opaque `signature_delta` data, and redacted-thinking
  data so the agent can persist replayable signed blocks. It also emits response
  metadata from `message_start` plus `message_delta/message_stop` fallback IDs,
  and emits usage on message completion.
- SSE consumption now buffers malformed premature boundaries and surfaces malformed payload errors instead of silently dropping fragments (`sdk/llm/anthropic/client.go:757`)
- Non-positive numeric `Retry-After` values are ignored with a warning hook instead of failing silently (`sdk/llm/anthropic/client.go:422`)
- Tool choice mapping aligns `required` with Anthropic `any`, supports `auto`/`none`, and forced named tool mode when thinking is disabled (`sdk/llm/anthropic/client.go:887`)
- Tool-call ID normalization now logs both original and sanitized IDs when characters are rewritten for Anthropic compatibility (`sdk/llm/anthropic/client.go:521`)

## Agent-Side Stream Normalization
- `invokeCompletion` in the agent consumes provider-specific stream events, emits provider-agnostic agent events, and preserves stream response metadata (`sdk/agent/agent.go:597`, `sdk/agent/agent.go:616`, `sdk/agent/agent.go:644`)
- Structured thinking stream metadata is aggregated into provider-neutral
  `ContentBlock{Type:"thinking", Thinking, Signature}` or
  `redacted_thinking` blocks. Anthropic then serializes those blocks before the
  associated assistant `tool_use` blocks on the next request; display-only
  thinking events from other providers remain in `Completion.Thinking` without
  changing their persisted content shape.
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
- Completion diagnostics are emitted as `WarnEvent`s. Diagnostics without an
  explicit kind use `provider_diagnostic`, while compatibility downgrades keep
  their provider-specific kind for downstream UIs and protocol adapters.

## Response ID Semantics
- `Completion.ResponseID` is optional at the shared type level (`sdk/llm/types.go:139`)
- OpenAI Responses populates IDs in both sync and streaming paths (`sdk/llm/openai/responses.go:651`, `sdk/llm/openai/responses.go:1272`)
- Anthropic populates IDs in both sync and streaming paths (`sdk/llm/anthropic/client.go:718`, `sdk/llm/anthropic/client.go:1219`)
- OpenAI Chat completions currently do not expose response IDs in parsed completion objects (`sdk/llm/openai/chat.go:904`)

## OpenAI redirect credential boundary

OpenAI Chat and Responses clients, in buffered and streaming modes, follow
only same-origin redirects. Origin includes scheme, host, and effective port;
HTTPS-to-HTTP downgrades are rejected before a redirected request is sent.
Caller-provided redirect callbacks run only for an initially allowed target,
and the target URL is checked again after the callback returns so it cannot
move bearer credentials onto another origin. Redirect-policy and malformed
Location failures are fail-fast and their diagnostics never echo URL
userinfo, path, or query data.

## Usage and Optional Cost Calculation
- Providers populate a versioned `Usage` contract; downstream cost calculation is optional (`sdk/llm/types.go`, `sdk/llm/usage.go`, `sdk/tokens/cost.go:80`).
- Under `prompt_tokens_semantics=total_input_v1`, `PromptTokens` is the complete effective input size. `PromptCachedTokens`, `PromptCacheCreationTokens`, `PromptImageTokens`, and `PromptUncachedTokens` are breakdown fields and consumers must not add them to `PromptTokens` again.
- Anthropic normalizes both streaming and non-streaming usage as `input_tokens + cache_read_input_tokens + cache_creation_input_tokens`. OpenAI Chat keeps `prompt_tokens` unchanged because cached tokens are already a subset; OpenAI Responses treats `input_tokens` the same way and preserves cached/image details as subsets.
- `prompt_tokens_valid`, `prompt_tokens_source`, and `prompt_tokens_semantics` describe quality. Sources are `provider`, `estimate`, `missing`, or `legacy_or_unknown`. `ProviderPromptTokens` and `ProviderTotalTokens` retain raw gateway values when the agent substitutes an estimate.
- When a compaction watermark decision adds a local history-growth estimate to an exact provider count, the decision total is an estimate rather than a measurement. The agent emits one `compaction_decision_estimate_mixed` warning carrying the exact provider base, the estimated delta, and the resulting decision value, so the estimator-versus-provider drift is observable for calibration.
- If a provider returns a usage object with prompt tokens missing/zero for a non-empty request, the agent uses `EstimateMessagesTokens`, emits one `provider_usage_prompt_tokens_missing` warning per query, and sends the effective estimate to context, compaction, and budget consumers. A completely absent usage object remains absent; the SDK does not invent a usage event.
- Positive custom/legacy usage remains usable but is explicitly labeled `legacy_or_unknown`, so adapters can render it as approximate instead of claiming v1 provider precision.
- Cost helper fetches LiteLLM pricing data, caches for 24h, warns on pricing-init/cache-read/cache-parse/cache-write/cost-calc failures while keeping non-fatal usage-only fallback, and tracks usage history (`sdk/tokens/cost.go:20`, `sdk/tokens/cost.go:22`, `sdk/tokens/cost.go:99`, `sdk/tokens/cost.go:108`, `sdk/tokens/cost.go:126`, `sdk/tokens/cost.go:173`, `sdk/tokens/cost.go:217`)
- Pricing lookup now supports case-insensitive names, common aliases, provider-prefix probing, and base-family fallback (choosing latest/date-tagged variants when exact names are unavailable) (`sdk/tokens/cost.go:292`, `sdk/tokens/cost.go:310`, `sdk/tokens/cost.go:392`)
- Cost math separates cached vs non-cached prompt tokens and output tokens (`sdk/tokens/cost.go:243`, `sdk/tokens/cost.go:255`, `sdk/tokens/cost.go:263`)

### Anthropic redirect credential boundary

Anthropic buffered and streaming clients follow only same-origin redirects.
Cross-origin targets and HTTPS-to-non-HTTPS downgrades are rejected before a
redirected request is sent, so `x-api-key` and request content never leave the
configured API origin. A caller-provided `CheckRedirect` callback is composed
for allowed same-origin redirects and cannot mutate the request onto another
origin without a second policy check.
