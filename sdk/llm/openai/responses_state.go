package openai

import (
	"bytes"
	"encoding/json"
	"fmt"
	"reflect"
	"strings"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

const (
	responsesStateProvider       = "openai-responses"
	responsesOutputItemStateKind = "response.output_item.v1"
	maxResponsesOutputItems      = 1024
	maxResponsesStateBytes       = 8 * 1024 * 1024
)

func responsesProviderStateFromResponseJSON(data []byte) ([]llm.ProviderState, error) {
	var root map[string]json.RawMessage
	if err := json.Unmarshal(data, &root); err != nil {
		return nil, err
	}
	rawOutput, ok := root["output"]
	if !ok || bytes.Equal(bytes.TrimSpace(rawOutput), []byte("null")) {
		return nil, nil
	}
	var items []json.RawMessage
	if err := json.Unmarshal(rawOutput, &items); err != nil {
		return nil, fmt.Errorf("openai responses: output must be an array: %w", err)
	}
	return responsesProviderStateFromRawItems(items)
}

func responsesProviderStateFromStreamEvent(data []byte) ([]llm.ProviderState, error) {
	var event map[string]json.RawMessage
	if err := json.Unmarshal(data, &event); err != nil {
		return nil, err
	}
	response, ok := event["response"]
	if !ok || len(bytes.TrimSpace(response)) == 0 || bytes.Equal(bytes.TrimSpace(response), []byte("null")) {
		return responsesProviderStateFromResponseJSON(data)
	}
	return responsesProviderStateFromResponseJSON(response)
}

func responsesProviderStateFromRawStreamItem(data []byte) (int, llm.ProviderState, bool, error) {
	var event struct {
		OutputIndex *int            `json:"output_index"`
		Item        json.RawMessage `json:"item"`
	}
	if err := json.Unmarshal(data, &event); err != nil {
		return 0, llm.ProviderState{}, false, err
	}
	if event.OutputIndex == nil {
		return 0, llm.ProviderState{}, false, fmt.Errorf("openai responses: output_item.done event has no output_index")
	}
	if len(bytes.TrimSpace(event.Item)) == 0 {
		return 0, llm.ProviderState{}, false, fmt.Errorf("openai responses: output_item.done event has no item")
	}
	if *event.OutputIndex < 0 {
		return 0, llm.ProviderState{}, false, fmt.Errorf("openai responses: output_index must be non-negative")
	}
	if *event.OutputIndex >= maxResponsesOutputItems {
		return 0, llm.ProviderState{}, false, fmt.Errorf("openai responses: output_index %d exceeds limit %d", *event.OutputIndex, maxResponsesOutputItems-1)
	}
	states, err := responsesProviderStateFromRawItems([]json.RawMessage{event.Item})
	if err != nil {
		return 0, llm.ProviderState{}, false, err
	}
	return *event.OutputIndex, states[0], true, nil
}

func responsesProviderStateFromRawItems(items []json.RawMessage) ([]llm.ProviderState, error) {
	if len(items) > maxResponsesOutputItems {
		return nil, fmt.Errorf("openai responses: output item count %d exceeds limit %d", len(items), maxResponsesOutputItems)
	}
	states := make([]llm.ProviderState, 0, len(items))
	totalBytes := 0
	for index, raw := range items {
		raw = bytes.TrimSpace(raw)
		if len(raw) == 0 || !json.Valid(raw) {
			return nil, fmt.Errorf("openai responses: output item %d is not valid JSON", index)
		}
		var item map[string]json.RawMessage
		if err := json.Unmarshal(raw, &item); err != nil || item == nil {
			return nil, fmt.Errorf("openai responses: output item %d must be an object", index)
		}
		var itemType string
		if err := json.Unmarshal(item["type"], &itemType); err != nil || strings.TrimSpace(itemType) == "" {
			return nil, fmt.Errorf("openai responses: output item %d has no valid type", index)
		}
		if err := validateResponsesOutputItem(index, itemType, item); err != nil {
			return nil, err
		}
		totalBytes += len(raw)
		if totalBytes > maxResponsesStateBytes {
			return nil, fmt.Errorf("openai responses: opaque output state exceeds %d bytes", maxResponsesStateBytes)
		}
		states = append(states, llm.ProviderState{
			Provider: responsesStateProvider,
			Kind:     responsesOutputItemStateKind,
			Data:     append([]byte(nil), raw...),
		})
	}
	return states, nil
}

func validateResponsesOutputItem(index int, itemType string, item map[string]json.RawMessage) error {
	itemType = strings.TrimSpace(itemType)
	if responsesOutputItemTypeIsInputOnly(itemType) {
		return fmt.Errorf("openai responses: output item %d uses input-only type %q", index, itemType)
	}
	if itemType != "message" {
		return nil
	}
	if rawRole, present := item["role"]; present {
		var role string
		if err := json.Unmarshal(rawRole, &role); err != nil || role != "assistant" {
			return fmt.Errorf("openai responses: output message item %d must have role assistant when role is present", index)
		}
	}
	var content []json.RawMessage
	if err := json.Unmarshal(item["content"], &content); err != nil {
		return fmt.Errorf("openai responses: output message item %d must have a content array", index)
	}
	for partIndex, rawPart := range content {
		var part map[string]json.RawMessage
		if err := json.Unmarshal(rawPart, &part); err != nil || part == nil {
			return fmt.Errorf("openai responses: output message item %d content %d must be an object", index, partIndex)
		}
		var partType string
		if err := json.Unmarshal(part["type"], &partType); err != nil || strings.TrimSpace(partType) == "" {
			return fmt.Errorf("openai responses: output message item %d content %d has no valid type", index, partIndex)
		}
		if strings.HasPrefix(strings.TrimSpace(partType), "input_") {
			return fmt.Errorf("openai responses: output message item %d content %d uses input-only type %q", index, partIndex, strings.TrimSpace(partType))
		}
	}
	return nil
}

func responsesOutputItemTypeIsInputOnly(itemType string) bool {
	itemType = strings.TrimSpace(itemType)
	if strings.HasPrefix(itemType, "input_") {
		return true
	}
	switch itemType {
	case "function_call_output", "item_reference":
		return true
	default:
		return false
	}
}

func responsesOutputItemsFromMessage(message llm.Message) ([]json.RawMessage, bool, error) {
	providerState, err := llm.ProviderStateFromContent(message.Content)
	if err != nil {
		return nil, true, err
	}
	items := make([]json.RawMessage, 0, len(providerState))
	totalBytes := 0
	for _, state := range providerState {
		if state.Provider != responsesStateProvider || state.Kind != responsesOutputItemStateKind {
			continue
		}
		if len(items) >= maxResponsesOutputItems {
			return nil, true, fmt.Errorf("openai responses: opaque output item count exceeds %d", maxResponsesOutputItems)
		}
		validated, err := responsesProviderStateFromRawItems([]json.RawMessage{state.Data})
		if err != nil {
			return nil, true, err
		}
		totalBytes += len(validated[0].Data)
		if totalBytes > maxResponsesStateBytes {
			return nil, true, fmt.Errorf("openai responses: opaque output state exceeds %d bytes", maxResponsesStateBytes)
		}
		items = append(items, append([]byte(nil), validated[0].Data...))
	}
	return items, len(items) > 0, nil
}

func responsesMessagesContainOutputState(messages []llm.Message) (bool, error) {
	for _, message := range messages {
		stateItems, err := llm.ProviderStateFromContent(message.Content)
		if err != nil {
			return true, err
		}
		for _, state := range stateItems {
			if state.Provider == responsesStateProvider && state.Kind == responsesOutputItemStateKind {
				return true, nil
			}
		}
	}
	return false, nil
}

func orderedResponsesStreamState(items map[int]llm.ProviderState) ([]llm.ProviderState, error) {
	if len(items) == 0 {
		return nil, nil
	}
	if len(items) > maxResponsesOutputItems {
		return nil, fmt.Errorf("openai responses: output item count %d exceeds limit %d", len(items), maxResponsesOutputItems)
	}
	state := make([]llm.ProviderState, 0, len(items))
	totalBytes := 0
	for index := 0; index < len(items); index++ {
		item, ok := items[index]
		if !ok {
			return nil, fmt.Errorf("openai responses: streamed output indexes are not contiguous at index %d", index)
		}
		totalBytes += len(item.Data)
		if totalBytes > maxResponsesStateBytes {
			return nil, fmt.Errorf("openai responses: opaque output state exceeds %d bytes", maxResponsesStateBytes)
		}
		state = append(state, item)
	}
	return llm.CloneProviderState(state), nil
}

func responsesJSONEqual(left, right []byte) bool {
	var leftValue any
	var rightValue any
	if json.Unmarshal(left, &leftValue) != nil || json.Unmarshal(right, &rightValue) != nil {
		return false
	}
	return reflect.DeepEqual(leftValue, rightValue)
}
