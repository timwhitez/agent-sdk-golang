package compaction

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"strings"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/artifact"
)

const (
	LedgerSchemaVersion          = 1
	CompactionSummaryMessageName = compactionSummaryMessageName
)

type LedgerStore interface {
	Load(ctx context.Context, sessionID string) (*Ledger, error)
	Save(ctx context.Context, sessionID string, ledger *Ledger) error
}

type Ledger struct {
	SchemaVersion int                 `json:"schema_version"`
	SessionID     string              `json:"session_id"`
	UpdatedAt     time.Time           `json:"updated_at,omitempty"`
	ContextWindow int                 `json:"context_window,omitempty"`
	PolicyHash    string              `json:"policy_hash,omitempty"`
	Summary       *LedgerSummary      `json:"summary,omitempty"`
	Replacements  []LedgerReplacement `json:"replacements,omitempty"`
}

type LedgerSummary struct {
	Version         int    `json:"version,omitempty"`
	MessageName     string `json:"message_name,omitempty"`
	SummaryHash     string `json:"summary_hash,omitempty"`
	CoveredStartKey string `json:"covered_start_key,omitempty"`
	CoveredEndKey   string `json:"covered_end_key,omitempty"`
	CheckpointID    string `json:"checkpoint_id,omitempty"`
	SourceSnapshot  string `json:"source_snapshot,omitempty"`
}

type LedgerReplacement struct {
	MessageKey        string             `json:"message_key"`
	PartKey           string             `json:"part_key"`
	Role              string             `json:"role,omitempty"`
	ToolName          string             `json:"tool_name,omitempty"`
	Tier              string             `json:"tier,omitempty"`
	OriginalHash      string             `json:"original_hash,omitempty"`
	ReplacementHash   string             `json:"replacement_hash,omitempty"`
	ReplacementText   string             `json:"replacement_text,omitempty"`
	CanonicalArtifact *artifact.Manifest `json:"canonical_artifact,omitempty"`
	// FullArtifact is legacy path-only compatibility data. It is never a
	// verified canonical artifact slot and is mutually exclusive with
	// CanonicalArtifact.
	FullArtifact          string    `json:"full_artifact,omitempty"`
	ParentReplacementHash string    `json:"parent_replacement_hash,omitempty"`
	CreatedAt             time.Time `json:"created_at,omitempty"`

	OriginalText string `json:"-"`
}

type MessageKeyInput struct {
	Role           string
	ToolCallID     string
	ToolName       string
	OriginalText   string
	FirstSeenIndex int
}

func NewLedger(sessionID string) *Ledger {
	return &Ledger{
		SchemaVersion: LedgerSchemaVersion,
		SessionID:     strings.TrimSpace(sessionID),
	}
}

func (l *Ledger) Clone() *Ledger {
	if l == nil {
		return nil
	}
	out := *l
	if l.Summary != nil {
		summary := *l.Summary
		out.Summary = &summary
	}
	if len(l.Replacements) > 0 {
		out.Replacements = append([]LedgerReplacement(nil), l.Replacements...)
		for i := range out.Replacements {
			if l.Replacements[i].CanonicalArtifact != nil {
				manifest := l.Replacements[i].CanonicalArtifact.Clone()
				out.Replacements[i].CanonicalArtifact = &manifest
			}
		}
	}
	return &out
}

func (l *Ledger) Validate(sessionID string) error {
	if l == nil {
		return fmt.Errorf("ledger is nil")
	}
	if l.SchemaVersion != LedgerSchemaVersion {
		return fmt.Errorf("unsupported ledger schema_version %d", l.SchemaVersion)
	}
	wantSession := strings.TrimSpace(sessionID)
	if gotSession := strings.TrimSpace(l.SessionID); gotSession != "" && wantSession != "" && gotSession != wantSession {
		return fmt.Errorf("ledger session_id %q does not match %q", gotSession, wantSession)
	}
	if strings.TrimSpace(l.SessionID) == "" {
		l.SessionID = wantSession
	}
	seen := map[string]struct{}{}
	for i, repl := range l.Replacements {
		id := strings.TrimSpace(repl.MessageKey) + "\x00" + strings.TrimSpace(repl.PartKey)
		if strings.TrimSpace(repl.MessageKey) == "" || strings.TrimSpace(repl.PartKey) == "" {
			return fmt.Errorf("replacement %d missing message_key or part_key", i)
		}
		if _, ok := seen[id]; ok {
			return fmt.Errorf("duplicate replacement identity message_key=%q part_key=%q", repl.MessageKey, repl.PartKey)
		}
		seen[id] = struct{}{}
		if repl.OriginalHash != "" && repl.OriginalText != "" && repl.OriginalHash != ContentHash(repl.OriginalText) {
			return fmt.Errorf("replacement %d original hash mismatch", i)
		}
		if repl.ReplacementHash != "" && repl.ReplacementHash != ContentHash(repl.ReplacementText) {
			return fmt.Errorf("replacement %d replacement hash mismatch", i)
		}
		if repl.CanonicalArtifact != nil {
			if strings.TrimSpace(repl.FullArtifact) != "" {
				return fmt.Errorf("replacement %d canonical_artifact and legacy full_artifact are mutually exclusive", i)
			}
			if err := validateLedgerCanonicalArtifact(*repl.CanonicalArtifact); err != nil {
				return fmt.Errorf("replacement %d canonical_artifact invalid: %w", i, err)
			}
			text := repl.ReplacementText
			if strings.TrimSpace(text) == "" {
				return fmt.Errorf("replacement %d canonical replacement text is required", i)
			}
			if !strings.Contains(text, repl.CanonicalArtifact.ObjectRef) {
				return fmt.Errorf("replacement %d text omits canonical object_ref", i)
			}
			if !strings.Contains(text, repl.CanonicalArtifact.Recovery.Tool) {
				return fmt.Errorf("replacement %d text omits canonical recovery tool", i)
			}
		}
	}
	if l.Summary != nil && l.Summary.SummaryHash != "" && l.Summary.MessageName == "" {
		return fmt.Errorf("summary message_name is required when summary_hash is set")
	}
	return nil
}

func validateLedgerCanonicalArtifact(manifest artifact.Manifest) error {
	if err := manifest.Validate(); err != nil {
		return err
	}
	if !manifest.Complete || !manifest.Recoverable {
		return fmt.Errorf("manifest must be complete and recoverable")
	}
	if manifest.ObjectKind == artifact.ObjectKindProviderVisibleView {
		return fmt.Errorf("provider-visible view cannot populate a verified canonical source slot")
	}
	if manifest.Retention.Class != artifact.RetentionDurable || manifest.Retention.ExpiresAt != nil {
		return fmt.Errorf("manifest retention must be durable without expires_at")
	}
	if manifest.ObjectMeasurement.Bytes == nil || strings.TrimSpace(manifest.ObjectMeasurement.SHA256) == "" {
		return fmt.Errorf("manifest must contain measured bytes and sha256")
	}
	return nil
}

func StableMessageKey(in MessageKeyInput) string {
	role := normalizeKeyPart(in.Role, "unknown")
	toolCallID := normalizeKeyPart(in.ToolCallID, "-")
	toolName := normalizeKeyPart(in.ToolName, "-")
	return fmt.Sprintf("%s/%s/%s/%s/idx:%d", role, toolCallID, toolName, ContentHash(in.OriginalText), in.FirstSeenIndex)
}

func ContentHash(text string) string {
	normalized := normalizeHashText(text)
	sum := sha256.Sum256([]byte(normalized))
	return "sha256:" + hex.EncodeToString(sum[:])
}

func normalizeKeyPart(value, fallback string) string {
	value = strings.TrimSpace(value)
	if value == "" {
		return fallback
	}
	value = strings.ReplaceAll(value, "/", "_")
	return value
}

func normalizeHashText(text string) string {
	text = strings.ReplaceAll(text, "\r\n", "\n")
	text = strings.ReplaceAll(text, "\r", "\n")
	lines := strings.Split(text, "\n")
	for i, line := range lines {
		lines[i] = strings.TrimRight(line, " \t")
	}
	return strings.TrimSpace(strings.Join(lines, "\n"))
}
