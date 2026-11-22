package openai

import (
	"encoding/json"
	"fmt"
)

// Each content kind owns an independent tracker. Text and refusal events share
// the public delta type, but their provider snapshots describe distinct parts.
type responsesTextParts struct {
	kind        string
	text        map[responsesTextPartKey]string
	itemIDs     map[int]string
	itemIndices map[string]int
}

type responsesTextPartKey struct {
	itemID                    string
	outputIndex, contentIndex int
}

func newResponsesTextParts(kind string) *responsesTextParts {
	return &responsesTextParts{
		kind:        kind,
		text:        map[responsesTextPartKey]string{},
		itemIDs:     map[int]string{},
		itemIndices: map[string]int{},
	}
}

func (p *responsesTextParts) alignKey(key responsesTextPartKey) responsesTextPartKey {
	// A legacy single-part event may have no identity at all. Bind it when its
	// first indexed/identified snapshot becomes available, even after reasoning.
	anonymous := responsesTextPartKey{outputIndex: -1, contentIndex: key.contentIndex}
	if len(p.text) == 1 {
		if prior, exists := p.text[anonymous]; exists {
			p.text[key] = prior
			delete(p.text, anonymous)
		}
	}
	return key
}

// bindItem records only an explicit item ID/index pair. Item headers may have
// no content yet, but their declared aliases already apply to later parts.
func (p *responsesTextParts) bindItem(itemID string, outputIndex any) error {
	index, hasIndex := responsesStreamIndex(outputIndex)
	if itemID == "" || !hasIndex {
		return nil
	}
	if index < 0 {
		return fmt.Errorf("Responses %s item identity conflicts with negative output index", p.kind)
	}
	if existing, ok := p.itemIDs[index]; ok {
		if existing != itemID {
			return fmt.Errorf("Responses %s item identity conflicts with output index %d", p.kind, index)
		}
		return nil
	}
	if existing, ok := p.itemIndices[itemID]; ok && existing != index {
		return fmt.Errorf("Responses %s output index conflicts with the announced item identity", p.kind)
	}
	// A single unbound indexed prefix can be moved to its declared ID. Two
	// independently emitted nonempty prefixes cannot be silently overwritten
	// or reordered now that they are claimed to describe the same part.
	for key, prior := range p.text {
		if key.itemID != "" || key.outputIndex != index {
			continue
		}
		identified := responsesTextPartKey{itemID: itemID, outputIndex: -1, contentIndex: key.contentIndex}
		if prior != "" && p.text[identified] != "" {
			return fmt.Errorf("Responses %s item identity conflicts with separately emitted part prefixes", p.kind)
		}
	}
	p.itemIDs[index] = itemID
	p.itemIndices[itemID] = index
	for key, prior := range p.text {
		if key.itemID != "" || key.outputIndex != index {
			continue
		}
		identified := responsesTextPartKey{itemID: itemID, outputIndex: -1, contentIndex: key.contentIndex}
		if p.text[identified] == "" {
			p.text[identified] = prior
		}
		delete(p.text, key)
	}
	return nil
}

func (p *responsesTextParts) key(itemID string, outputIndex any, contentIndex int) (responsesTextPartKey, error) {
	if err := p.bindItem(itemID, outputIndex); err != nil {
		return responsesTextPartKey{}, err
	}
	index, hasIndex := responsesStreamIndex(outputIndex)
	if !hasIndex && itemID == "" {
		if len(p.text) == 1 {
			for key := range p.text {
				if key.contentIndex == contentIndex {
					return key, nil
				}
			}
		}
		return responsesTextPartKey{outputIndex: -1, contentIndex: contentIndex}, nil
	}
	if itemID == "" && hasIndex {
		itemID = p.itemIDs[index]
	}
	if itemID != "" {
		key := responsesTextPartKey{itemID: itemID, outputIndex: -1, contentIndex: contentIndex}
		return p.alignKey(key), nil
	}
	return p.alignKey(responsesTextPartKey{outputIndex: index, contentIndex: contentIndex}), nil
}

// A top-level compatibility snapshot has no part identity. It can complete a
// single part, including one with a nonzero content index, but cannot be mapped
// onto one of several named parts without inventing an identity.
func (p *responsesTextParts) singleKey() (responsesTextPartKey, bool) {
	if len(p.text) > 1 {
		return responsesTextPartKey{}, false
	}
	for key := range p.text {
		return key, true
	}
	return responsesTextPartKey{outputIndex: -1}, true
}

// Streaming must preserve whitespace-only content as well as trailing spaces.
// The buffered helpers deliberately normalize presentation, so do not use them
// to reconcile an already-emitted exact prefix.
func responsesStreamContentText(content map[string]any) (string, bool) {
	for _, key := range []string{"text", "refusal", "content"} {
		if value, ok := content[key].(string); ok {
			return value, true
		}
	}
	return "", false
}

func responsesStreamIndex(v any) (int, bool) {
	switch x := v.(type) {
	case float64:
		return int(x), true
	case int:
		return x, true
	case int64:
		return int(x), true
	case json.Number:
		if i, err := x.Int64(); err == nil {
			return int(i), true
		}
		f, err := x.Float64()
		return int(f), err == nil
	default:
		return 0, false
	}
}
