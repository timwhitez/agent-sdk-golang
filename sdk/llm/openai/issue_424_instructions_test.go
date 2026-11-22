package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"reflect"
	"strings"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

type issue424InstructionsTransport struct {
	body   string
	stream bool
}

func (r *issue424InstructionsTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	b, err := io.ReadAll(req.Body)
	if err != nil {
		return nil, err
	}
	r.body = string(b)
	data := `{"id":"issue424","status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"ok"}]}]}`
	if r.stream {
		data = "data: {\"type\":\"response.completed\",\"response\":" + data + "}\n\n"
	}
	return &http.Response{StatusCode: 200, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(data)), Request: req}, nil
}

func TestIssue424ResponsesInstructionsMovesSystemMessages(t *testing.T) {
	for _, useItems := range []bool{false, true} {
		for _, stream := range []bool{false, true} {
			for _, setting := range []string{"true", "false", "nil"} {
				t.Run(fmt.Sprintf("items=%v/stream=%v/instructions=%s", useItems, stream, setting), func(t *testing.T) {
					transport := &issue424InstructionsTransport{stream: stream}
					client := &ResponsesClient{ModelName: "issue424-model", MaxRetries: 1, HTTPClient: &http.Client{Transport: transport}}
					var useInstructions *bool
					if setting != "nil" {
						value := setting == "true"
						useInstructions = &value
					}
					request := llm.InvokeRequest{Messages: []llm.Message{
						{Role: llm.RoleSystem, Content: llm.TextContent("first system")},
						llm.NewUserMessage("hello"),
						{Role: llm.RoleSystem, Content: llm.TextContent("second system")},
					}, Responses: &llm.ResponsesOptions{UseResponseItems: &useItems, UseInstructions: useInstructions}}
					before, _ := json.Marshal(request.Messages)
					if stream {
						events, err := client.InvokeStream(context.Background(), request)
						if err != nil {
							t.Fatal(err)
						}
						for event := range events {
							if e, ok := event.(llm.StreamErrorEvent); ok {
								t.Fatal(e.AsError())
							}
						}
					} else if _, err := client.Invoke(context.Background(), request); err != nil {
						t.Fatal(err)
					}
					after, _ := json.Marshal(request.Messages)
					if !reflect.DeepEqual(before, after) {
						t.Fatal("input messages were mutated")
					}
					var payload struct {
						Instructions string `json:"instructions"`
						Input        []struct {
							Role string `json:"role"`
						} `json:"input"`
					}
					if err := json.Unmarshal([]byte(transport.body), &payload); err != nil {
						t.Fatal(err)
					}
					wantSystems := 2
					if setting == "true" {
						wantSystems = 0
						if payload.Instructions != "first system\n\nsecond system" {
							t.Fatalf("instructions=%q", payload.Instructions)
						}
					} else if payload.Instructions != "" {
						t.Fatalf("unexpected instructions=%q", payload.Instructions)
					}
					systems := 0
					for _, item := range payload.Input {
						if item.Role == "system" {
							systems++
						}
					}
					if systems != wantSystems {
						t.Fatalf("system messages=%d, want=%d", systems, wantSystems)
					}
				})
			}
		}
	}
}
