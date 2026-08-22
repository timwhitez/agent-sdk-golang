from pathlib import Path

chat = Path("sdk/llm/openai/chat.go")
text = chat.read_text()
old = '''func defaultRetryableStatus(code int) bool {
\tswitch code {
\tcase 401, 403, 408, 409, 425, 429:
\t\treturn true
\tdefault:
\t\treturn code >= 500 && code <= 599
\t}
}
'''
new = '''func defaultRetryableStatus(code int) bool {
\tswitch code {
\t// Authentication and authorization failures are permanent for a client
\t// whose credentials do not change between attempts. Integrations with an
\t// explicit refresh mechanism can opt in through RetryableStatusCodes.
\tcase 408, 409, 425, 429:
\t\treturn true
\tdefault:
\t\treturn code >= 500 && code <= 599
\t}
}
'''
if text.count(old) != 1:
    raise SystemExit(f"OpenAI retry status anchor count={text.count(old)}")
chat.write_text(text.replace(old, new))

agent = Path("sdk/agent/agent.go")
text = agent.read_text()
old = '''func retryableProviderStatus(status int) bool {
\tswitch status {
\tcase 401, 403, 408, 409, 425, 429:
\t\treturn true
\tdefault:
\t\treturn status >= 500 && status <= 599
\t}
}
'''
new = '''func retryableProviderStatus(status int) bool {
\tswitch status {
\tcase 408, 409, 425, 429:
\t\treturn true
\tdefault:
\t\treturn status >= 500 && status <= 599
\t}
}
'''
if text.count(old) != 1:
    raise SystemExit(f"Agent retry status anchor count={text.count(old)}")
agent.write_text(text.replace(old, new))

Path("sdk/llm/openai/auth_retry_test.go").write_text(r'''package openai

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

func authFailureServer(status int, requests *atomic.Int32) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests.Add(1)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_, _ = fmt.Fprint(w, `{"error":{"message":"denied"}}`)
	}))
}

func TestOpenAIAuthFailuresAreNotRetriedByDefault(t *testing.T) {
	for _, status := range []int{http.StatusUnauthorized, http.StatusForbidden} {
		status := status
		t.Run(fmt.Sprint(status), func(t *testing.T) {
			for _, provider := range []string{"chat", "responses"} {
				provider := provider
				t.Run(provider, func(t *testing.T) {
					var requests atomic.Int32
					server := authFailureServer(status, &requests)
					defer server.Close()
					var err error
					switch provider {
					case "chat":
						client := &ChatClient{BaseURL: server.URL, ModelName: "test-model"}
						_, err = client.Invoke(context.Background(), llm.InvokeRequest{Messages: []llm.Message{llm.NewUserMessage("hello")}})
					case "responses":
						client := &ResponsesClient{BaseURL: server.URL, ModelName: "test-model"}
						_, err = client.Invoke(context.Background(), llm.InvokeRequest{Messages: []llm.Message{llm.NewUserMessage("hello")}})
					}
					if err == nil {
						t.Fatal("expected authentication failure")
					}
					if got := requests.Load(); got != 1 {
						t.Fatalf("HTTP requests = %d, want 1", got)
					}
				})
			}
		})
	}
}

func TestOpenAIAuthRetryCanBeExplicitlyEnabled(t *testing.T) {
	var requests atomic.Int32
	server := authFailureServer(http.StatusUnauthorized, &requests)
	defer server.Close()
	client := &ChatClient{
		BaseURL:             server.URL,
		ModelName:           "test-model",
		MaxRetries:          2,
		RetryableStatusCodes: map[int]struct{}{http.StatusUnauthorized: {}},
	}
	_, err := client.Invoke(context.Background(), llm.InvokeRequest{Messages: []llm.Message{llm.NewUserMessage("hello")}})
	if err == nil {
		t.Fatal("expected authentication failure")
	}
	if got := requests.Load(); got != 2 {
		t.Fatalf("HTTP requests = %d, want explicit two attempts", got)
	}
}
''')

Path("sdk/agent/agent_auth_retry_test.go").write_text(r'''package agent

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm/openai"
)

func TestAgentDoesNotRetryOpenAIAuthFailures(t *testing.T) {
	for _, status := range []int{http.StatusUnauthorized, http.StatusForbidden} {
		status := status
		t.Run(fmt.Sprint(status), func(t *testing.T) {
			var requests atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				requests.Add(1)
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(status)
				_, _ = fmt.Fprint(w, `{"error":{"message":"denied"}}`)
			}))
			defer server.Close()
			client := &openai.ChatClient{BaseURL: server.URL, ModelName: "test-model"}
			agent, err := New(Config{LLM: client})
			if err != nil {
				t.Fatal(err)
			}
			_, err = agent.Query(context.Background(), "hello")
			if err == nil {
				t.Fatal("expected authentication failure")
			}
			if got := requests.Load(); got != 1 {
				t.Fatalf("HTTP requests = %d, want one provider/framework attempt", got)
			}
		})
	}
}
''')
