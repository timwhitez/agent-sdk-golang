package openai

import (
	"context"
	"errors"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

func openAIRedirectTestRequest() llm.InvokeRequest {
	return llm.InvokeRequest{Messages: []llm.Message{llm.NewUserMessage("ping")}}
}

func invokeOpenAIRedirectTest(t *testing.T, provider string, streaming bool, baseURL, apiKey string) error {
	return invokeOpenAIRedirectTestWithRetries(t, provider, streaming, baseURL, apiKey, 1)
}

func invokeOpenAIRedirectTestWithRetries(t *testing.T, provider string, streaming bool, baseURL, apiKey string, maxRetries int) error {
	t.Helper()
	request := openAIRedirectTestRequest()
	var model llm.ChatModel
	switch provider {
	case "chat":
		model = &ChatClient{HTTPClient: &http.Client{Timeout: 2 * time.Second}, BaseURL: baseURL, APIKey: apiKey, ModelName: "test-model", MaxRetries: maxRetries, RetryBaseDelay: time.Nanosecond, RetryMaxDelay: time.Nanosecond}
	case "responses":
		model = &ResponsesClient{HTTPClient: &http.Client{Timeout: 2 * time.Second}, BaseURL: baseURL, APIKey: apiKey, ModelName: "test-model", MaxRetries: maxRetries, RetryBaseDelay: time.Nanosecond, RetryMaxDelay: time.Nanosecond}
	default:
		t.Fatalf("unknown provider %q", provider)
	}
	if !streaming {
		_, err := model.Invoke(context.Background(), request)
		return err
	}
	streamingModel, ok := model.(llm.StreamingChatModel)
	if !ok {
		t.Fatalf("provider %q is not streaming", provider)
	}
	events, err := streamingModel.InvokeStream(context.Background(), request)
	if err != nil {
		return err
	}
	for event := range events {
		if failure, ok := event.(llm.StreamErrorEvent); ok {
			return failure.AsError()
		}
	}
	return nil
}

func TestOpenAIClientsRedactMalformedLocationAndDoNotRetry(t *testing.T) {
	for _, provider := range []string{"chat", "responses"} {
		for _, streaming := range []bool{false, true} {
			provider := provider
			streaming := streaming
			t.Run(provider+map[bool]string{false: "_buffered", true: "_streaming"}[streaming], func(t *testing.T) {
				const secret = "malformed-location-secret"
				var sourceCalls atomic.Int32
				redirector := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
					sourceCalls.Add(1)
					w.Header().Set("Location", "https://audit-user:audit-password@connection.invalid/%zz?token="+secret)
					w.WriteHeader(http.StatusTemporaryRedirect)
				}))
				defer redirector.Close()

				err := invokeOpenAIRedirectTestWithRetries(t, provider, streaming, redirector.URL, secret, 3)
				if err == nil {
					t.Fatal("malformed redirect unexpectedly succeeded")
				}
				for _, value := range []string{secret, "audit-user", "audit-password", "connection.invalid", "/%zz", "token="} {
					if strings.Contains(err.Error(), value) {
						t.Fatalf("malformed redirect error exposed %q: %v", value, err)
					}
				}
				if got := sourceCalls.Load(); got != 1 {
					t.Fatalf("malformed redirect source received %d requests, want fail-fast 1", got)
				}
			})
		}
	}
}

func TestOpenAIClientsDoNotRetryPolicyErrorWithRetryKeyword(t *testing.T) {
	for _, provider := range []string{"chat", "responses"} {
		for _, streaming := range []bool{false, true} {
			provider := provider
			streaming := streaming
			t.Run(provider+map[bool]string{false: "_buffered", true: "_streaming"}[streaming], func(t *testing.T) {
				var sourceCalls atomic.Int32
				redirector := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
					sourceCalls.Add(1)
					w.Header().Set("Location", "http://connection.invalid/private")
					w.WriteHeader(http.StatusTemporaryRedirect)
				}))
				defer redirector.Close()

				err := invokeOpenAIRedirectTestWithRetries(t, provider, streaming, redirector.URL, "policy-secret", 3)
				if err == nil || !strings.Contains(err.Error(), "cross-origin redirect") {
					t.Fatalf("policy redirect error = %v", err)
				}
				if isRetryableNetErr(err) {
					t.Fatalf("redirect policy error was classified retryable: %v", err)
				}
				if got := sourceCalls.Load(); got != 1 {
					t.Fatalf("policy redirect source received %d requests, want fail-fast 1", got)
				}
			})
		}
	}
}

func TestOpenAIClientsRejectSameOriginRedirectURLCredentials(t *testing.T) {
	for _, provider := range []string{"chat", "responses"} {
		for _, streaming := range []bool{false, true} {
			provider := provider
			streaming := streaming
			t.Run(provider+map[bool]string{false: "_buffered", true: "_streaming"}[streaming], func(t *testing.T) {
				var sourceCalls atomic.Int32
				var redirectedCalls atomic.Int32
				var server *httptest.Server
				server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					sourceCalls.Add(1)
					if r.URL.Path == "/redirected" {
						redirectedCalls.Add(1)
						w.WriteHeader(http.StatusInternalServerError)
						return
					}
					target := strings.Replace(server.URL, "://", "://audit-user:audit-password@", 1) + "/redirected?token=audit-query-secret"
					w.Header().Set("Location", target)
					w.WriteHeader(http.StatusTemporaryRedirect)
				}))
				defer server.Close()

				err := invokeOpenAIRedirectTestWithRetries(t, provider, streaming, server.URL, "bearer-secret", 3)
				if err == nil || !strings.Contains(err.Error(), "embedded URL credentials") {
					t.Fatalf("redirect credential error = %v", err)
				}
				for _, value := range []string{"audit-user", "audit-password", "audit-query-secret", "/redirected"} {
					if strings.Contains(err.Error(), value) {
						t.Fatalf("redirect credential error exposed %q: %v", value, err)
					}
				}
				if sourceCalls.Load() != 1 || redirectedCalls.Load() != 0 {
					t.Fatalf("source/redirected calls = %d/%d, want 1/0", sourceCalls.Load(), redirectedCalls.Load())
				}
			})
		}
	}
}

func TestSanitizeOpenAIHTTPErrorPreservesSafeRetrySemantics(t *testing.T) {
	t.Parallel()
	const secret = "network-error-secret"
	tests := []struct {
		name        string
		cause       error
		wantRetry   bool
		wantTimeout bool
	}{
		{name: "timeout", cause: secretOpenAITimeoutError{message: secret}, wantRetry: true, wantTimeout: true},
		{name: "network operation", cause: &net.OpError{Op: "dial", Net: "tcp", Err: errors.New(secret)}, wantRetry: true},
		{name: "permanent network error", cause: secretOpenAIPermanentNetError{message: secret}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := sanitizeOpenAIHTTPError(&url.Error{
				Op:  "Post",
				URL: "https://api.example.test/private?token=" + secret,
				Err: tc.cause,
			})
			var sanitized *url.Error
			if !errors.As(err, &sanitized) {
				t.Fatalf("sanitized error = %T, want *url.Error", err)
			}
			if strings.Contains(sanitized.Error(), secret) || strings.Contains(sanitized.Error(), "/private") {
				t.Fatalf("sanitized error leaked request details: %q", sanitized.Error())
			}
			if got := isRetryableNetErr(sanitized); got != tc.wantRetry {
				t.Fatalf("retryable = %t, want %t: %v", got, tc.wantRetry, sanitized)
			}
			if got := sanitized.Timeout(); got != tc.wantTimeout {
				t.Fatalf("timeout = %t, want %t: %v", got, tc.wantTimeout, sanitized)
			}
		})
	}
}

type secretOpenAITimeoutError struct {
	message string
}

func (e secretOpenAITimeoutError) Error() string { return e.message }
func (secretOpenAITimeoutError) Timeout() bool   { return true }
func (secretOpenAITimeoutError) Temporary() bool { return true }

type secretOpenAIPermanentNetError struct {
	message string
}

func (e secretOpenAIPermanentNetError) Error() string { return e.message }
func (secretOpenAIPermanentNetError) Timeout() bool   { return false }
func (secretOpenAIPermanentNetError) Temporary() bool { return false }

func TestOpenAIClientsRejectCrossOriginRedirectWithoutLeakingBearerToken(t *testing.T) {
	for _, provider := range []string{"chat", "responses"} {
		for _, streaming := range []bool{false, true} {
			provider := provider
			streaming := streaming
			t.Run(provider+map[bool]string{false: "_buffered", true: "_streaming"}[streaming], func(t *testing.T) {
				const secret = "redirect-bearer-sentinel"
				var destinationCalls atomic.Int32
				destination := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					destinationCalls.Add(1)
					if got := r.Header.Get("Authorization"); got != "" {
						t.Errorf("redirect destination received Authorization %q", got)
					}
					w.WriteHeader(http.StatusInternalServerError)
				}))
				defer destination.Close()
				redirectTarget := strings.Replace(destination.URL, "://", "://audit-user:audit-password@", 1) + "/private?token=audit-query-secret"

				redirector := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					http.Redirect(w, r, redirectTarget, http.StatusTemporaryRedirect)
				}))
				defer redirector.Close()

				err := invokeOpenAIRedirectTest(t, provider, streaming, redirector.URL, secret)
				if err == nil || !strings.Contains(err.Error(), "cross-origin redirect") {
					t.Fatalf("redirect error = %v, want cross-origin rejection", err)
				}
				for _, credential := range []string{secret, "audit-user", "audit-password", "audit-query-secret", "/private"} {
					if strings.Contains(err.Error(), credential) {
						t.Fatalf("redirect error exposed %q: %v", credential, err)
					}
				}
				if got := destinationCalls.Load(); got != 0 {
					t.Fatalf("redirect destination received %d request(s)", got)
				}
			})
		}
	}
}

func TestOpenAIClientsAllowSameOriginRedirect(t *testing.T) {
	for _, provider := range []string{"chat", "responses"} {
		for _, streaming := range []bool{false, true} {
			provider := provider
			streaming := streaming
			t.Run(provider+map[bool]string{false: "_buffered", true: "_streaming"}[streaming], func(t *testing.T) {
				const secret = "same-origin-secret"
				var server *httptest.Server
				server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					if r.URL.Path != "/redirected" {
						http.Redirect(w, r, server.URL+"/redirected", http.StatusTemporaryRedirect)
						return
					}
					if got := r.Header.Get("Authorization"); got != "Bearer "+secret {
						t.Errorf("same-origin Authorization = %q", got)
					}
					if streaming {
						w.Header().Set("Content-Type", "text/event-stream")
						if provider == "chat" {
							_, _ = io.WriteString(w, "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n")
						} else {
							_, _ = io.WriteString(w, "data: {\"type\":\"response.output_text.delta\",\"delta\":\"ok\"}\n\ndata: [DONE]\n\n")
						}
						return
					}
					w.Header().Set("Content-Type", "application/json")
					if provider == "chat" {
						_, _ = io.WriteString(w, `{"id":"chat_1","choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`)
					} else {
						_, _ = io.WriteString(w, `{"id":"resp_1","status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"ok"}]}]}`)
					}
				}))
				defer server.Close()

				if err := invokeOpenAIRedirectTest(t, provider, streaming, server.URL, secret); err != nil {
					t.Fatalf("same-origin redirect: %v", err)
				}
			})
		}
	}
}

func TestRedirectSafeOpenAIHTTPClientComposesAndRechecksCallerCallback(t *testing.T) {
	origin, _ := url.Parse("https://api.example.test/v1/responses")
	sameOrigin, _ := url.Parse("https://api.example.test/v2/responses")
	foreign, _ := url.Parse("https://attacker.example.test/steal")

	called := false
	base := &http.Client{CheckRedirect: func(req *http.Request, _ []*http.Request) error {
		called = true
		req.URL = foreign
		return nil
	}}
	safe := redirectSafeOpenAIHTTPClient(base)
	request := &http.Request{URL: sameOrigin}
	err := safe.CheckRedirect(request, []*http.Request{{URL: origin}})
	if !called {
		t.Fatal("caller CheckRedirect was not invoked for same-origin target")
	}
	if err == nil || !strings.Contains(err.Error(), "cross-origin redirect") {
		t.Fatalf("post-callback error = %v, want cross-origin rejection", err)
	}
	if base.CheckRedirect == nil {
		t.Fatal("base client was mutated")
	}
}

func TestRedirectSafeOpenAIHTTPClientRejectsHTTPSDowngrade(t *testing.T) {
	origin, _ := url.Parse("https://api.example.test/v1/responses")
	downgrade, _ := url.Parse("http://api.example.test/v1/responses")
	safe := redirectSafeOpenAIHTTPClient(&http.Client{})
	err := safe.CheckRedirect(&http.Request{URL: downgrade}, []*http.Request{{URL: origin}})
	if err == nil || !strings.Contains(err.Error(), "HTTPS redirect downgrade") {
		t.Fatalf("downgrade error = %v", err)
	}
}

func TestRedirectSafeOpenAIHTTPClientPreservesCallerStopDecision(t *testing.T) {
	origin, _ := url.Parse("https://api.example.test/v1/responses")
	target, _ := url.Parse("https://api.example.test/v2/responses")
	safe := redirectSafeOpenAIHTTPClient(&http.Client{CheckRedirect: func(*http.Request, []*http.Request) error {
		return http.ErrUseLastResponse
	}})
	err := safe.CheckRedirect(&http.Request{URL: target}, []*http.Request{{URL: origin}})
	if !errors.Is(err, http.ErrUseLastResponse) {
		t.Fatalf("caller redirect decision = %v, want ErrUseLastResponse", err)
	}
}

func TestStreamHTTPClientRetainsOpenAIRedirectPolicy(t *testing.T) {
	origin, _ := url.Parse("https://api.example.test/v1/responses")
	foreign, _ := url.Parse("https://api.example.test:8443/v1/responses")
	base := redirectSafeOpenAIHTTPClient(&http.Client{Timeout: time.Second})
	stream := streamHTTPClient(base)
	if stream.Timeout != 0 {
		t.Fatalf("stream timeout = %s, want zero", stream.Timeout)
	}
	if stream.CheckRedirect == nil {
		t.Fatal("stream client dropped redirect policy")
	}
	err := stream.CheckRedirect(&http.Request{URL: foreign}, []*http.Request{{URL: origin}})
	if err == nil || !strings.Contains(err.Error(), "cross-origin redirect") {
		t.Fatalf("stream redirect error = %v", err)
	}
}
