package anthropic

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

func redirectTestRequest() llm.InvokeRequest {
	return llm.InvokeRequest{Messages: []llm.Message{llm.NewUserMessage("ping")}}
}

func TestAnthropicInvokeRejectsCrossOriginRedirectWithoutLeakingAPIKey(t *testing.T) {
	var destinationCalls atomic.Int32
	leaked := make(chan string, 1)
	destination := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		destinationCalls.Add(1)
		leaked <- r.Header.Get("X-Api-Key")
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer destination.Close()

	redirector := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, destination.URL, http.StatusTemporaryRedirect)
	}))
	defer redirector.Close()

	client := &Client{
		HTTPClient: &http.Client{Timeout: 2 * time.Second},
		BaseURL:    redirector.URL,
		APIKey:     "secret-key",
		ModelName:  "test-model",
		MaxTokens:  16,
		MaxRetries: 1,
	}
	if _, err := client.Invoke(context.Background(), redirectTestRequest()); err == nil || !strings.Contains(err.Error(), "cross-origin redirect") {
		t.Fatalf("Invoke error = %v, want cross-origin redirect rejection", err)
	}
	if got := destinationCalls.Load(); got != 0 {
		t.Fatalf("redirect destination received %d request(s)", got)
	}
	select {
	case value := <-leaked:
		t.Fatalf("redirect destination received x-api-key %q", value)
	default:
	}
}

func TestAnthropicInvokeStreamRejectsCrossOriginRedirectWithoutLeakingAPIKey(t *testing.T) {
	var destinationCalls atomic.Int32
	destination := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		destinationCalls.Add(1)
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer destination.Close()

	redirector := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, destination.URL, http.StatusTemporaryRedirect)
	}))
	defer redirector.Close()

	client := &Client{
		HTTPClient: &http.Client{Timeout: 2 * time.Second},
		BaseURL:    redirector.URL,
		APIKey:     "secret-key",
		ModelName:  "test-model",
		MaxTokens:  16,
		MaxRetries: 1,
	}
	events, err := client.InvokeStream(context.Background(), redirectTestRequest())
	if err != nil {
		t.Fatalf("InvokeStream setup: %v", err)
	}
	found := false
	for event := range events {
		if failure, ok := event.(llm.StreamErrorEvent); ok && failure.Err != nil && strings.Contains(failure.Err.Error(), "cross-origin redirect") {
			found = true
		}
	}
	if !found {
		t.Fatal("stream did not emit cross-origin redirect error")
	}
	if got := destinationCalls.Load(); got != 0 {
		t.Fatalf("stream redirect destination received %d request(s)", got)
	}
}

func TestAnthropicInvokeAllowsSameOriginRedirect(t *testing.T) {
	receivedAPIKey := make(chan string, 1)
	var server *httptest.Server
	server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/redirected" {
			http.Redirect(w, r, server.URL+"/redirected", http.StatusTemporaryRedirect)
			return
		}
		receivedAPIKey <- r.Header.Get("X-Api-Key")
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"text","text":"ok"}],"model":"test-model","stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)
	}))
	defer server.Close()

	client := &Client{
		HTTPClient: &http.Client{Timeout: 2 * time.Second},
		BaseURL:    server.URL,
		APIKey:     "secret-key",
		ModelName:  "test-model",
		MaxTokens:  16,
		MaxRetries: 1,
	}
	completion, err := client.Invoke(context.Background(), redirectTestRequest())
	if err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	if got := completion.PlainText(); got != "ok" {
		t.Fatalf("completion text = %q, want ok", got)
	}
	select {
	case got := <-receivedAPIKey:
		if got != "secret-key" {
			t.Fatalf("same-origin x-api-key = %q, want secret-key", got)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("same-origin redirect destination was not reached")
	}
}

func TestRedirectSafeHTTPClientComposesAndRechecksCallerCallback(t *testing.T) {
	origin, _ := url.Parse("https://api.example.test/v1/messages")
	sameOrigin, _ := url.Parse("https://api.example.test/v2/messages")
	foreign, _ := url.Parse("https://attacker.example.test/steal")

	called := false
	base := &http.Client{CheckRedirect: func(req *http.Request, _ []*http.Request) error {
		called = true
		req.URL = foreign
		return nil
	}}
	safe := redirectSafeHTTPClient(base)
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

func TestRedirectSafeHTTPClientRejectsHTTPSDowngrade(t *testing.T) {
	origin, _ := url.Parse("https://api.example.test/v1/messages")
	downgrade, _ := url.Parse("http://api.example.test/v1/messages")
	safe := redirectSafeHTTPClient(&http.Client{})
	err := safe.CheckRedirect(&http.Request{URL: downgrade}, []*http.Request{{URL: origin}})
	if err == nil || !strings.Contains(err.Error(), "HTTPS redirect downgrade") {
		t.Fatalf("downgrade error = %v", err)
	}
}

func TestRedirectSafeHTTPClientPreservesCallerStopDecision(t *testing.T) {
	origin, _ := url.Parse("https://api.example.test/v1/messages")
	target, _ := url.Parse("https://api.example.test/v2/messages")
	safe := redirectSafeHTTPClient(&http.Client{CheckRedirect: func(*http.Request, []*http.Request) error {
		return http.ErrUseLastResponse
	}})
	err := safe.CheckRedirect(&http.Request{URL: target}, []*http.Request{{URL: origin}})
	if !errors.Is(err, http.ErrUseLastResponse) {
		t.Fatalf("caller redirect decision = %v, want ErrUseLastResponse", err)
	}
}
