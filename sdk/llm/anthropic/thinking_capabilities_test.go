package anthropic

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

func captureAnthropicInvokePayload(t *testing.T, client *Client, req llm.InvokeRequest) map[string]any {
	t.Helper()
	var rawBody []byte
	c := *client
	c.BaseURL = "https://example.invalid"
	c.MaxRetries = 1
	c.HTTPClient = &http.Client{
		Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
			var err error
			rawBody, err = io.ReadAll(r.Body)
			_ = r.Body.Close()
			if err != nil {
				return nil, err
			}
			body := `{"id":"msg_cap","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`
			return httpResponse(http.StatusOK, body, r), nil
		}),
	}
	if _, err := c.Invoke(context.Background(), req); err != nil {
		t.Fatalf("Invoke(%s): %v", c.ModelName, err)
	}
	var payload map[string]any
	if err := json.Unmarshal(rawBody, &payload); err != nil {
		t.Fatalf("unmarshal captured payload: %v", err)
	}
	return payload
}

func captureAnthropicStreamPayload(t *testing.T, client *Client, req llm.InvokeRequest) map[string]any {
	t.Helper()
	var rawBody []byte
	c := *client
	c.BaseURL = "https://example.invalid"
	c.MaxRetries = 1
	c.HTTPClient = &http.Client{
		Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
			var err error
			rawBody, err = io.ReadAll(r.Body)
			_ = r.Body.Close()
			if err != nil {
				return nil, err
			}
			streamBody := strings.Join([]string{
				`data: {"type":"message_start","message":{"id":"msg_stream_cap","usage":{"input_tokens":1}}}`,
				"",
				`data: {"type":"content_block_delta","index":0,"delta":{"text":"ok"}}`,
				"",
				`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}`,
				"",
				`data: {"type":"message_stop"}`,
				"",
			}, "\n")
			return httpResponse(http.StatusOK, streamBody, r), nil
		}),
	}
	events, err := c.InvokeStream(context.Background(), req)
	if err != nil {
		t.Fatalf("InvokeStream(%s): %v", c.ModelName, err)
	}
	for ev := range events {
		if errEv, ok := ev.(llm.StreamErrorEvent); ok {
			t.Fatalf("InvokeStream(%s) error event: %v", c.ModelName, errEv.AsError())
		}
	}
	var payload map[string]any
	if err := json.Unmarshal(rawBody, &payload); err != nil {
		t.Fatalf("unmarshal captured stream payload: %v", err)
	}
	return payload
}

func TestAdaptiveEffortXHighAndStandardLevelsRequestCapture(t *testing.T) {
	t.Parallel()

	userReq := llm.InvokeRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: llm.TextContent("hi")}},
	}

	xhighModels := []string{
		"claude-opus-4-7",
		"claude-opus-4.7-latest",
		"claude-opus-4-8",
		"claude-opus-4.8-20260715",
		"claude-opus-5",
		"claude-opus-5-5",
		"anthropic/claude-opus-5-5-20261101",
		"claude-sonnet-5",
		"claude-sonnet-5-5",
		"claude-haiku-5-5",
		"claude-fable-5-1",
		"claude-mythos-preview",
	}
	for _, model := range xhighModels {
		model := model
		t.Run("xhigh/"+model, func(t *testing.T) {
			t.Parallel()
			client := &Client{
				ModelName:      model,
				ThinkingMode:   "adaptive",
				ThinkingEffort: "xhigh",
			}
			for _, stream := range []bool{false, true} {
				var payload map[string]any
				if stream {
					payload = captureAnthropicStreamPayload(t, client, userReq)
				} else {
					payload = captureAnthropicInvokePayload(t, client, userReq)
				}
				thinking, _ := payload["thinking"].(map[string]any)
				if thinking["type"] != "adaptive" {
					t.Fatalf("stream=%v thinking = %#v, want adaptive", stream, payload["thinking"])
				}
				outputConfig, _ := payload["output_config"].(map[string]any)
				if outputConfig["effort"] != "xhigh" {
					t.Fatalf("stream=%v output_config = %#v, want effort=xhigh verbatim", stream, payload["output_config"])
				}
			}
		})
	}

	// Original 4 effort levels and uppercase normalization.
	for _, tc := range []struct {
		input string
		want  string
	}{
		{"low", "low"},
		{"medium", "medium"},
		{"high", "high"},
		{"XHIGH", "xhigh"},
		{"max", "max"},
	} {
		t.Run("level/"+tc.input, func(t *testing.T) {
			client := &Client{
				ModelName:      "claude-opus-5-5",
				ThinkingMode:   "adaptive",
				ThinkingEffort: tc.input,
			}
			payload := captureAnthropicInvokePayload(t, client, userReq)
			outputConfig, _ := payload["output_config"].(map[string]any)
			if outputConfig["effort"] != tc.want {
				t.Fatalf("effort = %#v, want %q", outputConfig["effort"], tc.want)
			}
		})
	}

	// No explicit effort omits output_config.
	t.Run("no_explicit_effort", func(t *testing.T) {
		client := &Client{
			ModelName:    "claude-opus-4-8",
			ThinkingMode: "adaptive",
		}
		payload := captureAnthropicInvokePayload(t, client, userReq)
		thinking, _ := payload["thinking"].(map[string]any)
		if thinking["type"] != "adaptive" {
			t.Fatalf("thinking = %#v, want adaptive", payload["thinking"])
		}
		if _, exists := payload["output_config"]; exists {
			t.Fatalf("output_config should be omitted when effort is empty, got %#v", payload["output_config"])
		}
	})
}

func TestAdaptiveEffortModelValidationRejectsUnsupportedXHigh(t *testing.T) {
	t.Parallel()

	userReq := llm.InvokeRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: llm.TextContent("hi")}},
	}

	// Models that do not support xhigh must reject it before sending HTTP.
	for _, tc := range []struct {
		model     string
		badEffort string
		okEffort  string
	}{
		{model: "claude-opus-4-6", badEffort: "xhigh", okEffort: "max"},
		{model: "anthropic/claude-opus-4-6-20260205", badEffort: "xhigh", okEffort: "max"},
		{model: "claude-sonnet-4-6", badEffort: "xhigh", okEffort: "max"},
		{model: "claude-opus-4-5", badEffort: "xhigh", okEffort: "high"},
		{model: "claude-opus-4-5", badEffort: "max", okEffort: "high"},
		{model: "claude-sonnet-4-5-20250929", badEffort: "xhigh", okEffort: "high"},
		{model: "claude-3-7-sonnet-latest", badEffort: "xhigh", okEffort: "high"},
	} {
		t.Run(tc.model+"/"+tc.badEffort, func(t *testing.T) {
			called := false
			client := &Client{
				BaseURL:        "https://example.invalid",
				ModelName:      tc.model,
				ThinkingMode:   "adaptive",
				ThinkingEffort: tc.badEffort,
				MaxRetries:     1,
				HTTPClient: &http.Client{
					Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
						called = true
						return nil, errors.New("unexpected network call")
					}),
				},
			}
			_, err := client.Invoke(context.Background(), userReq)
			if err == nil {
				t.Fatalf("expected capability error for model=%q effort=%q", tc.model, tc.badEffort)
			}
			if called {
				t.Fatalf("network call should not be attempted when effort %q is unsupported", tc.badEffort)
			}
			var capErr *llm.ThinkingCapabilityError
			if !errors.As(err, &capErr) {
				t.Fatalf("expected *llm.ThinkingCapabilityError, got %T (%v)", err, err)
			}
			if capErr.Control != "effort" || capErr.Value != tc.badEffort {
				t.Fatalf("capErr = %+v, want control=effort value=%q", capErr, tc.badEffort)
			}

			// Supported level on the same model still passes cleanly.
			client.ThinkingEffort = tc.okEffort
			payload := captureAnthropicInvokePayload(t, client, userReq)
			outputConfig, _ := payload["output_config"].(map[string]any)
			if outputConfig["effort"] != tc.okEffort {
				t.Fatalf("effort = %#v, want %q", outputConfig["effort"], tc.okEffort)
			}
		})
	}

	// Explicit Capabilities override controls custom gateway models and is cloned by BindFrameModel.
	t.Run("custom_gateway_override_and_bind_clone", func(t *testing.T) {
		override := &ModelCapabilities{
			AdaptiveThinking:                     true,
			SupportedEfforts:                     []string{"low", "medium", "high"},
			DisableMode:                          ThinkingDisableOmit,
			ForcedToolChoiceWhenThinkingDisabled: true,
		}
		client := &Client{
			ModelName:      "corp-gateway-alias",
			ThinkingMode:   "adaptive",
			ThinkingEffort: "xhigh",
			Capabilities:   override,
		}
		boundModel, ok, err := client.BindFrameModel(context.Background())
		if err != nil || !ok {
			t.Fatalf("BindFrameModel: ok=%v err=%v", ok, err)
		}
		// Mutate source override after binding; bound client must keep original snapshot.
		override.SupportedEfforts = append(override.SupportedEfforts, "xhigh")
		boundClient := boundModel.(*Client)
		if _, err := boundClient.buildRequestWithThinking(userReq, boundClient.configuredThinking()); err == nil {
			t.Fatalf("bound client should still reject xhigh after source capabilities mutated")
		}
	})
}

func TestDisableThinkingByModelCapabilityAndNoConfigMutation(t *testing.T) {
	t.Parallel()

	tool := llm.ToolDefinition{Name: "done", Description: "finish", Parameters: map[string]any{"type": "object"}}
	reqWithTools := func(tc llm.ToolChoice, disable bool) llm.InvokeRequest {
		return llm.InvokeRequest{
			Messages:        []llm.Message{{Role: llm.RoleUser, Content: llm.TextContent("hi")}},
			Tools:           []llm.ToolDefinition{tool},
			ToolChoice:      tc,
			DisableThinking: disable,
		}
	}

	// 1. Default-on & disableable: claude-sonnet-5 sends explicit {"type":"disabled"} and does not mutate client config.
	t.Run("claude-sonnet-5_explicit_disabled_and_no_mutation", func(t *testing.T) {
		var bodies []map[string]any
		client := &Client{
			BaseURL:        "https://example.invalid",
			ModelName:      "claude-sonnet-5",
			ThinkingMode:   "adaptive",
			ThinkingEffort: "xhigh",
			MaxRetries:     1,
			HTTPClient: &http.Client{
				Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
					data, _ := io.ReadAll(r.Body)
					_ = r.Body.Close()
					var payload map[string]any
					if err := json.Unmarshal(data, &payload); err != nil {
						t.Fatalf("unmarshal: %v", err)
					}
					bodies = append(bodies, payload)
					body := `{"id":"msg_s5","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`
					return httpResponse(http.StatusOK, body, r), nil
				}),
			},
		}

		// Single DisableThinking=true call with forced tool_choice.
		if _, err := client.Invoke(context.Background(), reqWithTools("required", true)); err != nil {
			t.Fatalf("Invoke DisableThinking=true: %v", err)
		}
		// Subsequent normal call must retain adaptive + xhigh.
		if _, err := client.Invoke(context.Background(), reqWithTools("auto", false)); err != nil {
			t.Fatalf("Invoke normal: %v", err)
		}
		if len(bodies) != 2 {
			t.Fatalf("captured %d requests, want 2", len(bodies))
		}
		thinking0, _ := bodies[0]["thinking"].(map[string]any)
		if thinking0["type"] != "disabled" {
			t.Fatalf("DisableThinking=true on claude-sonnet-5 sent thinking=%#v, want type=disabled", bodies[0]["thinking"])
		}
		if _, hasOC := bodies[0]["output_config"]; hasOC {
			t.Fatalf("DisableThinking=true should omit output_config, got %#v", bodies[0]["output_config"])
		}
		tc0, _ := bodies[0]["tool_choice"].(map[string]any)
		if tc0["type"] != "any" {
			t.Fatalf("tool_choice = %#v, want any", bodies[0]["tool_choice"])
		}

		thinking1, _ := bodies[1]["thinking"].(map[string]any)
		if thinking1["type"] != "adaptive" {
			t.Fatalf("subsequent call thinking = %#v, want adaptive", bodies[1]["thinking"])
		}
		oc1, _ := bodies[1]["output_config"].(map[string]any)
		if oc1["effort"] != "xhigh" {
			t.Fatalf("subsequent call output_config = %#v, want effort=xhigh", bodies[1]["output_config"])
		}
		if client.ThinkingMode != "adaptive" || client.ThinkingEffort != "xhigh" {
			t.Fatalf("client config mutated: mode=%q effort=%q", client.ThinkingMode, client.ThinkingEffort)
		}
	})

	// 2. Default-on & disableable within effort range: claude-opus-5 and claude-haiku-5-5.
	for _, model := range []string{"claude-opus-5", "claude-haiku-5-5"} {
		model := model
		t.Run(model+"_disable_and_effort_range", func(t *testing.T) {
			// Single-call DisableThinking=true omits output_config even if normal effort is xhigh/max.
			client := &Client{
				ModelName:      model,
				ThinkingMode:   "adaptive",
				ThinkingEffort: "xhigh",
			}
			payload := captureAnthropicInvokePayload(t, client, reqWithTools("required", true))
			thinking, _ := payload["thinking"].(map[string]any)
			if thinking["type"] != "disabled" {
				t.Fatalf("thinking = %#v, want disabled", payload["thinking"])
			}
			if _, hasOC := payload["output_config"]; hasOC {
				t.Fatalf("DisableThinking=true must omit output_config so effort never contradicts disabled: %#v", payload["output_config"])
			}

			// Explicit ThinkingMode="disabled" with allowed effort (high) succeeds.
			disabledHigh := &Client{
				ModelName:      model,
				ThinkingMode:   "disabled",
				ThinkingEffort: "high",
			}
			payload = captureAnthropicInvokePayload(t, disabledHigh, reqWithTools("auto", false))
			thinking, _ = payload["thinking"].(map[string]any)
			if thinking["type"] != "disabled" {
				t.Fatalf("explicit disabled thinking = %#v, want disabled", payload["thinking"])
			}
			oc, _ := payload["output_config"].(map[string]any)
			if oc["effort"] != "high" {
				t.Fatalf("explicit disabled output_config = %#v, want effort=high", payload["output_config"])
			}

			// Explicit ThinkingMode="disabled" with xhigh or max must be rejected.
			for _, bad := range []string{"xhigh", "max"} {
				disabledBad := &Client{
					ModelName:      model,
					ThinkingMode:   "disabled",
					ThinkingEffort: bad,
				}
				_, err := disabledBad.buildRequestWithThinking(reqWithTools("auto", false), disabledBad.configuredThinking())
				var capErr *llm.ThinkingCapabilityError
				if !errors.As(err, &capErr) {
					t.Fatalf("model=%s disabled+effort=%s: expected ThinkingCapabilityError, got %v", model, bad, err)
				}
			}
		})
	}

	// 3. Default-off legacy models omit thinking on DisableThinking=true.
	for _, model := range []string{"claude-3-7-sonnet-latest", "claude-sonnet-4-20250514", "claude-opus-4-6", "claude-opus-4-8"} {
		t.Run("legacy_default_off/"+model, func(t *testing.T) {
			budget := 8000
			client := &Client{
				ModelName:            model,
				ThinkingBudgetTokens: &budget,
			}
			payload := captureAnthropicInvokePayload(t, client, reqWithTools("required", true))
			if _, hasThinking := payload["thinking"]; hasThinking {
				t.Fatalf("default-off model %s should omit thinking when DisableThinking=true, got %#v", model, payload["thinking"])
			}
			if _, hasOC := payload["output_config"]; hasOC {
				t.Fatalf("default-off model %s should omit output_config when DisableThinking=true, got %#v", model, payload["output_config"])
			}
		})
	}

	// 4. Non-disableable models reject DisableThinking=true before HTTP.
	for _, tc := range []struct {
		model               string
		wantSubstring       string
		supportsBetweenTool bool
	}{
		{model: "claude-opus-5-5", wantSubstring: "cannot be disabled", supportsBetweenTool: false},
		{model: "claude-sonnet-5-5", wantSubstring: "between_tools", supportsBetweenTool: true},
		{model: "claude-fable-5-1", wantSubstring: "cannot be disabled", supportsBetweenTool: false},
		{model: "claude-mythos-preview", wantSubstring: "cannot be disabled", supportsBetweenTool: false},
	} {
		t.Run("non_disableable/"+tc.model, func(t *testing.T) {
			called := false
			client := &Client{
				BaseURL:        "https://example.invalid",
				ModelName:      tc.model,
				ThinkingMode:   "adaptive",
				ThinkingEffort: "xhigh",
				MaxRetries:     1,
				HTTPClient: &http.Client{
					Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
						called = true
						return nil, errors.New("unexpected network call")
					}),
				},
			}
			_, err := client.Invoke(context.Background(), reqWithTools("auto", true))
			if err == nil {
				t.Fatalf("expected capability error for DisableThinking=true on %s", tc.model)
			}
			if called {
				t.Fatalf("unexpected network call on unsupported DisableThinking for %s", tc.model)
			}
			var capErr *llm.ThinkingCapabilityError
			if !errors.As(err, &capErr) {
				t.Fatalf("expected *llm.ThinkingCapabilityError, got %T (%v)", err, err)
			}
			if !strings.Contains(err.Error(), tc.wantSubstring) {
				t.Fatalf("error %q missing %q", err.Error(), tc.wantSubstring)
			}

			// ThinkingCapabilities must also report false for both controls.
			caps, ok := llm.ModelThinkingCapabilities(client)
			if !ok || caps.SupportsDisableThinking || caps.SupportsForcedToolChoice {
				t.Fatalf("ThinkingCapabilities(%s) = %+v, ok=%v; want both false", tc.model, caps, ok)
			}

			// between_tools is distinct from disabled: accepted on claude-sonnet-5-5 only.
			btClient := &Client{
				ModelName:      tc.model,
				ThinkingMode:   "between_tools",
				ThinkingEffort: "high",
			}
			if tc.supportsBetweenTool {
				payload := captureAnthropicInvokePayload(t, btClient, reqWithTools("auto", false))
				thinking, _ := payload["thinking"].(map[string]any)
				if thinking["type"] != "between_tools" {
					t.Fatalf("between_tools payload thinking = %#v", payload["thinking"])
				}
			} else {
				if _, err := btClient.buildRequestWithThinking(reqWithTools("auto", false), btClient.configuredThinking()); err == nil {
					t.Fatalf("expected between_tools error on %s", tc.model)
				}
			}
		})
	}

	// 5. Default-on model with thinking omitted rejects forced tool_choice unless DisableThinking=true.
	t.Run("default_on_omitted_thinking_rejects_forced_tool_choice", func(t *testing.T) {
		sonnet5 := &Client{ModelName: "claude-sonnet-5"}
		if _, err := sonnet5.buildRequestWithThinking(reqWithTools("required", false), nil); err == nil || !strings.Contains(err.Error(), "DisableThinking") {
			t.Fatalf("expected DisableThinking conflict error on claude-sonnet-5 with omitted thinking, got %v", err)
		}
		opus55 := &Client{ModelName: "claude-opus-5-5"}
		_, err := opus55.buildRequestWithThinking(reqWithTools("required", false), nil)
		var capErr *llm.ThinkingCapabilityError
		if !errors.As(err, &capErr) || capErr.Control != "tool_choice" {
			t.Fatalf("expected tool_choice ThinkingCapabilityError on claude-opus-5-5, got %v", err)
		}
	})
}
