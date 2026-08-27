package openai

import (
	"context"
	"errors"
	"io"
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
	t.Helper()
	request := openAIRedirectTestRequest()
	var model llm.ChatModel
	switch provider {
	case "chat":
		model = &ChatClient{HTTPClient: &http.Client{Timeout: 2 * time.Second}, BaseURL: baseURL, APIKey: apiKey, ModelName: "test-model", MaxRetries: 1}
	case "responses":
		model = &ResponsesClient{HTTPClient: &http.Client{Timeout: 2 * time.Second}, BaseURL: baseURL, APIKey: apiKey, ModelName: "test-model", MaxRetries: 1}
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
