package llm

import (
	"encoding/json"
	"fmt"
	"strings"
)

const providerStateBlockType = "_sdk_provider_state.v1"

// IsProviderStateBlock reports whether a content block is the SDK-owned opaque
// provider-history carrier. Provider serializers other than the owning adapter
// must ignore these blocks.
func IsProviderStateBlock(block ContentBlock) bool {
	return block.Type == providerStateBlockType
}

// WithProviderState replaces the opaque provider history attached to content.
// State is encoded in an existing ContentBlock so adding this capability does
// not change the public Message or Completion struct shape.
func WithProviderState(content Content, state []ProviderState) (Content, error) {
	out := WithoutProviderState(content)
	if len(state) == 0 {
		return out, nil
	}
	cloned := CloneProviderState(state)
	for index, item := range cloned {
		if strings.TrimSpace(item.Provider) == "" || strings.TrimSpace(item.Kind) == "" {
			return content, fmt.Errorf("llm: provider state %d has no provider or kind", index)
		}
		if len(item.Data) == 0 || !json.Valid(item.Data) {
			return content, fmt.Errorf("llm: provider state %d has invalid JSON data", index)
		}
	}
	encoded, err := json.Marshal(cloned)
	if err != nil {
		return content, fmt.Errorf("llm: encode provider state: %w", err)
	}
	out.Blocks = append(out.Blocks, ContentBlock{Type: providerStateBlockType, Data: string(encoded)})
	return out, nil
}

// ProviderStateFromContent returns a deep copy of opaque provider history.
func ProviderStateFromContent(content Content) ([]ProviderState, error) {
	var encoded string
	found := false
	for _, block := range content.Blocks {
		if !IsProviderStateBlock(block) {
			continue
		}
		if found {
			return nil, fmt.Errorf("llm: content contains duplicate provider-state blocks")
		}
		if block.Text != "" || block.ImageURL != nil || block.Source != nil || block.Thinking != "" || block.Signature != "" {
			return nil, fmt.Errorf("llm: provider-state block contains visible or foreign fields")
		}
		encoded = block.Data
		found = true
	}
	if !found {
		return nil, nil
	}
	var state []ProviderState
	if err := json.Unmarshal([]byte(encoded), &state); err != nil {
		return nil, fmt.Errorf("llm: decode provider state: %w", err)
	}
	if len(state) == 0 {
		return nil, fmt.Errorf("llm: provider-state block is empty")
	}
	for index, item := range state {
		if strings.TrimSpace(item.Provider) == "" || strings.TrimSpace(item.Kind) == "" {
			return nil, fmt.Errorf("llm: provider state %d has no provider or kind", index)
		}
		if len(item.Data) == 0 || !json.Valid(item.Data) {
			return nil, fmt.Errorf("llm: provider state %d has invalid JSON data", index)
		}
	}
	return CloneProviderState(state), nil
}

// HasProviderState reports whether content carries an opaque state block. It
// intentionally does not treat malformed state as absent; adapters still
// validate it through ProviderStateFromContent and fail closed.
func HasProviderState(content Content) bool {
	for _, block := range content.Blocks {
		if IsProviderStateBlock(block) {
			return true
		}
	}
	return false
}

// WithoutProviderState removes all opaque state blocks while preserving other
// content and slice ownership.
func WithoutProviderState(content Content) Content {
	out := CloneContent(content)
	if len(out.Blocks) == 0 {
		return out
	}
	kept := out.Blocks[:0]
	for _, block := range out.Blocks {
		if IsProviderStateBlock(block) {
			continue
		}
		kept = append(kept, block)
	}
	if len(kept) == 0 {
		out.Blocks = nil
	} else {
		out.Blocks = kept
	}
	return out
}
