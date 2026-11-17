package openai

import (
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

const promptCacheKeyRejection = `{"error":{"message":"Unrecognized request argument supplied: prompt_cache_key","type":"invalid_request_error"}}`

func promptCacheKeyRequest(key string) llm.InvokeRequest {
	req := toolChoiceRequest("")
	if key != "" {
		req.Responses = &llm.ResponsesOptions{PromptCacheKey: key}
	}
	return req
}

func bodyKeysWithout(body map[string]any, drop string) []string {
	var keys []string
	for k := range body {
		if k != drop {
			keys = append(keys, k)
		}
	}
	return keys
}

// A gateway that rejects prompt_cache_key gets one resend without it; the
// legacy input stage is not entered and the caller's options stay untouched.
func TestResponsesPromptCacheKeyDowngrade(t *testing.T) {
	t.Parallel()
	for _, stream := range []bool{false, true} {
		stream := stream
		name := "buffered"
		if stream {
			name = "stream"
		}
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			run := func(key string, reject func(map[string]any) (int, string, bool), opts compatClientOptions, req llm.InvokeRequest) (*toolChoiceGateway, *warningRecorder, []llm.Diagnostic, string, error) {
				gw := &toolChoiceGateway{api: "responses", reject: reject}
				server := httptest.NewServer(gw)
				defer server.Close()
				warn := &warningRecorder{}
				opts.maxRetries = 1
				text, diags, err := invokeCompatClient(t, "responses", stream, server.URL, warn, opts, req)
				return gw, warn, diags, text, err
			}
			rejectKey := rejectWhenKey("prompt_cache_key", promptCacheKeyRejection)

			t.Run("rejected", func(t *testing.T) {
				req := promptCacheKeyRequest("cache-1")
				gw, warn, diags, text, err := run("cache-1", rejectKey, compatClientOptions{}, req)
				if err != nil || text != "ok" {
					t.Fatalf("text=%q err=%v, want ok", text, err)
				}
				bodies := gw.snapshot()
				if len(bodies) != 2 {
					t.Fatalf("requests = %d, want 2", len(bodies))
				}
				if bodies[0]["prompt_cache_key"] != "cache-1" {
					t.Fatalf("first request prompt_cache_key = %v", bodies[0]["prompt_cache_key"])
				}
				if _, ok := bodies[1]["prompt_cache_key"]; ok {
					t.Fatal("second request still carries prompt_cache_key")
				}
				// Only the cache key changed: still the full (non-legacy) input stage.
				if got, want := bodyKeysWithout(bodies[1], ""), bodyKeysWithout(bodies[0], "prompt_cache_key"); !sameKeys(got, want) {
					t.Fatalf("second request keys %v, want %v", got, want)
				}
				if warn.count("prompt_cache_key") != 1 || warn.count("legacy chat-compatible input") != 0 {
					t.Fatalf("warnings = %q", warn.msgs)
				}
				if !stream {
					n := 0
					for _, d := range diags {
						if d.Kind == "provider_compatibility_downgrade" {
							n++
							if !strings.Contains(d.Message, "prompt_cache_key") {
								t.Fatalf("diagnostic = %q", d.Message)
							}
						}
					}
					if n != 1 {
						t.Fatalf("downgrade diagnostics = %d, want 1 (%#v)", n, diags)
					}
				}
				if req.Responses.PromptCacheKey != "cache-1" {
					t.Fatal("caller's ResponsesOptions was mutated")
				}
			})

			t.Run("accepted", func(t *testing.T) {
				gw, warn, _, text, err := run("cache-1", func(map[string]any) (int, string, bool) { return 0, "", false }, compatClientOptions{}, promptCacheKeyRequest("cache-1"))
				if err != nil || text != "ok" {
					t.Fatalf("text=%q err=%v", text, err)
				}
				bodies := gw.snapshot()
				if len(bodies) != 1 || bodies[0]["prompt_cache_key"] != "cache-1" || len(warn.msgs) != 0 {
					t.Fatalf("bodies=%d warnings=%q", len(bodies), warn.msgs)
				}
			})

			t.Run("no_key_unaffected", func(t *testing.T) {
				// The gateway message names prompt_cache_key but none was sent:
				// existing behavior (legacy input stage) is unchanged.
				calls := 0
				gw, warn, _, _, _ := run("", func(map[string]any) (int, string, bool) {
					calls++
					if calls == 1 {
						return http.StatusBadRequest, promptCacheKeyRejection, true
					}
					return 0, "", false
				}, compatClientOptions{}, promptCacheKeyRequest(""))
				if len(gw.snapshot()) != 2 || warn.count("legacy chat-compatible input") != 1 || warn.count("rejected prompt_cache_key") != 0 {
					t.Fatalf("requests=%d warnings=%q", len(gw.snapshot()), warn.msgs)
				}
			})

			t.Run("composes_with_reasoning_effort", func(t *testing.T) {
				reject := func(body map[string]any) (int, string, bool) {
					if _, ok := body["prompt_cache_key"]; ok {
						return http.StatusBadRequest, promptCacheKeyRejection, true
					}
					if _, ok := body["reasoning"]; ok {
						return http.StatusBadRequest, `{"error":{"message":"unknown parameter: reasoning_effort"}}`, true
					}
					return 0, "", false
				}
				gw, warn, _, text, err := run("cache-1", reject, compatClientOptions{reasoningEffort: "low"}, promptCacheKeyRequest("cache-1"))
				if err != nil || text != "ok" || len(gw.snapshot()) != 3 {
					t.Fatalf("text=%q err=%v requests=%d", text, err, len(gw.snapshot()))
				}
				if warn.count("prompt_cache_key") != 1 || warn.count("reasoning_effort") != 1 || warn.count("legacy") != 0 {
					t.Fatalf("warnings = %q", warn.msgs)
				}
			})

			t.Run("persistent_rejection_does_not_loop", func(t *testing.T) {
				// The key is dropped once; a gateway that keeps rejecting is then
				// handled by the pre-existing legacy-input stage and finally fails.
				gw, warn, _, _, err := run("cache-1", func(map[string]any) (int, string, bool) {
					return http.StatusBadRequest, promptCacheKeyRejection, true
				}, compatClientOptions{}, promptCacheKeyRequest("cache-1"))
				if err == nil || len(gw.snapshot()) != 3 || warn.count("rejected prompt_cache_key") != 1 {
					t.Fatalf("err=%v requests=%d warnings=%q", err, len(gw.snapshot()), warn.msgs)
				}
			})
		})
	}
}

func sameKeys(a, b []string) bool {
	m := map[string]int{}
	for _, k := range a {
		m[k]++
	}
	for _, k := range b {
		m[k]--
	}
	return reflect.DeepEqual(filterZero(m), map[string]int{})
}

func filterZero(m map[string]int) map[string]int {
	for k, v := range m {
		if v == 0 {
			delete(m, k)
		}
	}
	return m
}
