package anthropic

import (
	"context"
	"io"
	"net/http"
	"reflect"
	"strings"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

func TestFrameBindingOwnsClientConfiguration(t *testing.T) {
	temperature, topP, seed, budget := 0.2, 0.7, 7, 1500
	client := &Client{ModelName: "old", APIKey: "fixture", Temperature: &temperature, TopP: &topP, Seed: &seed, ThinkingBudgetTokens: &budget, Beta: []string{"old-beta"}, RetryableStatusCodes: map[int]struct{}{500: {}}}
	model, known, err := llm.BindFrameModel(context.Background(), client)
	if err != nil || !known {
		t.Fatal(known, err)
	}
	bound := model.(*Client)
	if !reflect.DeepEqual(client, bound) {
		t.Fatal("binding changed configuration values")
	}
	client.ModelName, client.APIKey = "new", "new-fixture"
	temperature, topP, seed, budget = 0.8, 0.9, 8, 3000
	client.Beta[0] = "new-beta"
	delete(client.RetryableStatusCodes, 500)
	if bound.ModelName != "old" || bound.APIKey != "fixture" || *bound.Temperature != 0.2 || *bound.TopP != 0.7 || *bound.Seed != 7 || *bound.ThinkingBudgetTokens != 1500 || bound.Beta[0] != "old-beta" || len(bound.RetryableStatusCodes) != 1 {
		t.Fatal("binding retained mutable source configuration")
	}
	for _, empty := range []bool{false, true} {
		c := &Client{}
		if empty {
			c.Beta = []string{}
			c.RetryableStatusCodes = map[int]struct{}{}
		}
		m, _, err := c.BindFrameModel(context.Background())
		if err != nil || !reflect.DeepEqual(c, m) {
			t.Fatal("nil/empty changed", err)
		}
	}
}

func TestFrameBindingPreservesBufferedAndStreamWire(t *testing.T) {
	for _, stream := range []bool{false, true} {
		var requests, headers []string
		httpClient := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
			body, _ := io.ReadAll(r.Body)
			requests = append(requests, string(body))
			headers = append(headers, r.Header.Get("anthropic-beta"))
			return httpResponse(401, `{"error":{"message":"fixture"}}`, r), nil
		})}
		temperature := 0.2
		client := &Client{HTTPClient: httpClient, BaseURL: "https://fixture.invalid", ModelName: "old", Temperature: &temperature, Beta: []string{"old-beta"}, MaxRetries: 1}
		invoke := func(model llm.ChatModel) {
			request := llm.InvokeRequest{Messages: []llm.Message{llm.NewUserMessage("fixture")}}
			if stream {
				ch, err := model.(llm.StreamingChatModel).InvokeStream(context.Background(), request)
				if err == nil {
					for range ch {
					}
				}
			} else {
				_, _ = model.Invoke(context.Background(), request)
			}
		}
		invoke(client)
		bound, known, err := llm.BindFrameModel(context.Background(), client)
		if err != nil || !known {
			t.Fatal(err)
		}
		client.ModelName, client.Beta[0], temperature = "new", "new-beta", 0.9
		invoke(bound)
		if len(requests) != 2 || requests[0] != requests[1] || headers[0] != headers[1] || !strings.Contains(requests[1], `"model":"old"`) || !strings.Contains(requests[1], `"temperature":0.2`) {
			t.Fatal("bound wire changed with source", requests, headers)
		}
		if bound.(*Client).HTTPClient != httpClient {
			t.Fatal("transport handle unexpectedly copied")
		}
	}
}

func BenchmarkFrameModelBinding(b *testing.B) {
	temperature := 0.2
	client := &Client{ModelName: "fixture", Temperature: &temperature, Beta: []string{"a", "b"}, RetryableStatusCodes: map[int]struct{}{500: {}, 502: {}}}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if _, known, err := llm.BindFrameModel(context.Background(), client); err != nil || !known {
			b.Fatal(known, err)
		}
	}
}
