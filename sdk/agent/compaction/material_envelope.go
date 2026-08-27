package compaction

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"strings"

	"github.com/timwhitez/agent-sdk-golang/sdk/agent/messageorigin"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

const compactionMaterialSchemaV1 = "goode.compaction.material.v1"

type realUserAnchors struct {
	First  *string
	Latest *string
}

type compactionMaterialEnvelope struct {
	Schema                string  `json:"schema"`
	FirstRealUserRequest  *string `json:"first_real_user_request,omitempty"`
	LatestRealUserRequest *string `json:"latest_real_user_request,omitempty"`
	HostCheckpointStatus  string  `json:"host_checkpoint_status,omitempty"`
	Material              string  `json:"material"`
}

func collectRealUserAnchors(messages []llm.Message, estimate tokenEstimator) realUserAnchors {
	first := -1
	latest := -1
	for i, message := range messages {
		if !messageorigin.IsRealUserMessage(message) {
			continue
		}
		if first < 0 {
			first = i
		}
		latest = i
	}
	if first < 0 {
		return realUserAnchors{}
	}
	firstText := truncateCompactionMaterialTextWithEstimator(messages[first].Content.PlainText(), recentUserMaterialTokenBudget, estimate)
	latestText := truncateCompactionMaterialTextWithEstimator(messages[latest].Content.PlainText(), recentUserMaterialTokenBudget, estimate)
	return realUserAnchors{First: &firstText, Latest: &latestText}
}

func wrapCompactionMaterial(material string, anchors realUserAnchors, hostCheckpointStatus string) string {
	material = strings.TrimSpace(material)
	if material == "" {
		material = fallbackSummaryContext
	}
	envelope := compactionMaterialEnvelope{
		Schema:                compactionMaterialSchemaV1,
		FirstRealUserRequest:  cloneStringPointer(anchors.First),
		LatestRealUserRequest: cloneStringPointer(anchors.Latest),
		HostCheckpointStatus:  normalizeCompactionHostCheckpointStatus(hostCheckpointStatus),
		Material:              material,
	}
	encoded, _ := json.Marshal(envelope)
	return wrapUntrustedMaterial(string(encoded))
}

func decodeCompactionMaterialEnvelope(material string) (compactionMaterialEnvelope, bool, error) {
	framed := strings.Contains(material, beginUntrustedMaterial) || strings.Contains(material, endUntrustedMaterial)
	decoded, err := decodeSummaryValidationMaterial(material)
	if err != nil {
		return compactionMaterialEnvelope{}, framed, err
	}
	if !framed {
		return compactionMaterialEnvelope{Material: decoded}, false, nil
	}
	if err := validateCompactionEnvelopeObject(decoded); err != nil {
		return compactionMaterialEnvelope{}, true, err
	}
	decoder := json.NewDecoder(bytes.NewBufferString(decoded))
	decoder.DisallowUnknownFields()
	var envelope compactionMaterialEnvelope
	if err := decoder.Decode(&envelope); err != nil {
		return compactionMaterialEnvelope{}, true, fmt.Errorf("payload is not a valid %s object: %w", compactionMaterialSchemaV1, err)
	}
	if err := ensureJSONDecoderEOF(decoder); err != nil {
		return compactionMaterialEnvelope{}, true, err
	}
	if envelope.Schema != compactionMaterialSchemaV1 {
		return compactionMaterialEnvelope{}, true, fmt.Errorf("unsupported compaction material schema %q", envelope.Schema)
	}
	if (envelope.FirstRealUserRequest == nil) != (envelope.LatestRealUserRequest == nil) {
		return compactionMaterialEnvelope{}, true, fmt.Errorf("first/latest real-user anchors must either both be present or both be absent")
	}
	if !validCompactionHostCheckpointStatus(envelope.HostCheckpointStatus) {
		return compactionMaterialEnvelope{}, true, fmt.Errorf("invalid host checkpoint status %q", envelope.HostCheckpointStatus)
	}
	if strings.TrimSpace(envelope.Material) == "" {
		return compactionMaterialEnvelope{}, true, fmt.Errorf("compaction material body is empty")
	}
	return envelope, true, nil
}

func validateCompactionEnvelopeObject(data string) error {
	decoder := json.NewDecoder(strings.NewReader(data))
	start, err := decoder.Token()
	if err != nil {
		return fmt.Errorf("payload is not a JSON object: %w", err)
	}
	if delimiter, ok := start.(json.Delim); !ok || delimiter != '{' {
		return fmt.Errorf("payload must be a JSON object")
	}
	allowed := map[string]struct{}{
		"schema":                   {},
		"first_real_user_request":  {},
		"latest_real_user_request": {},
		"host_checkpoint_status":   {},
		"material":                 {},
	}
	seen := map[string]struct{}{}
	for decoder.More() {
		token, err := decoder.Token()
		if err != nil {
			return fmt.Errorf("read compaction material key: %w", err)
		}
		key, ok := token.(string)
		if !ok {
			return fmt.Errorf("compaction material object key must be a string")
		}
		if _, allowedKey := allowed[key]; !allowedKey {
			return fmt.Errorf("unknown compaction material field %q", key)
		}
		if _, duplicate := seen[key]; duplicate {
			return fmt.Errorf("duplicate compaction material field %q", key)
		}
		seen[key] = struct{}{}
		var value json.RawMessage
		if err := decoder.Decode(&value); err != nil {
			return fmt.Errorf("decode compaction material field %q: %w", key, err)
		}
		if bytes.Equal(bytes.TrimSpace(value), []byte("null")) {
			return fmt.Errorf("compaction material field %q must not be null", key)
		}
		var stringValue string
		if err := json.Unmarshal(value, &stringValue); err != nil {
			return fmt.Errorf("compaction material field %q must be a string: %w", key, err)
		}
		if key == "host_checkpoint_status" && stringValue == "" {
			return fmt.Errorf("compaction material field %q must not be empty when present", key)
		}
	}
	if _, err := decoder.Token(); err != nil {
		return fmt.Errorf("close compaction material object: %w", err)
	}
	if err := ensureJSONDecoderEOF(decoder); err != nil {
		return err
	}
	for _, required := range []string{"schema", "material"} {
		if _, ok := seen[required]; !ok {
			return fmt.Errorf("missing required compaction material field %q", required)
		}
	}
	return nil
}

func normalizeCompactionHostCheckpointStatus(status string) string {
	status = strings.ToUpper(strings.TrimSpace(status))
	if validCompactionHostCheckpointStatus(status) {
		return status
	}
	return CheckpointStatusUnknown
}

func validCompactionHostCheckpointStatus(status string) bool {
	switch status {
	case "", CheckpointStatusVerified, CheckpointStatusUnverified, CheckpointStatusUnknown:
		return true
	default:
		return false
	}
}

func ensureJSONDecoderEOF(decoder *json.Decoder) error {
	var trailing any
	if err := decoder.Decode(&trailing); err == io.EOF {
		return nil
	} else if err != nil {
		return fmt.Errorf("invalid trailing compaction material data: %w", err)
	}
	return fmt.Errorf("unexpected trailing compaction material value")
}

func cloneStringPointer(value *string) *string {
	if value == nil {
		return nil
	}
	cloned := *value
	return &cloned
}
