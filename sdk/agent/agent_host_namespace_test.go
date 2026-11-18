package agent

import (
	"strings"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/agent/compaction"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

func TestAgentRejectsInvalidHostNamespace(t *testing.T) {
	_, err := New(Config{LLM: noopCompactionUpdateModel{}, Compaction: &compaction.Config{Enabled: true, HostNamespace: "x -->"}})
	if err == nil || !strings.Contains(err.Error(), "host namespace") {
		t.Fatalf("New with invalid host namespace: err=%v", err)
	}

	var warnings []string
	ag, err := New(Config{
		LLM:        noopCompactionUpdateModel{},
		Compaction: &compaction.Config{Enabled: true, ContextWindow: 1000, HostNamespace: "acme"},
		Warningf:   func(format string, args ...any) { warnings = append(warnings, format) },
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	ag.UpdateCompactionConfig(&compaction.Config{Enabled: true, ContextWindow: 2000, HostNamespace: "Bad"})
	if ag.compactor == nil || ag.compactor.ContextWindow != 1000 || ag.compactor.Config.HostNamespace != "acme" {
		t.Fatalf("invalid update replaced the compactor: %#v", ag.compactor)
	}
	if len(warnings) != 1 || !strings.Contains(warnings[0], "UpdateCompactionConfig rejected") {
		t.Fatalf("warnings = %#v", warnings)
	}
}

func TestEmergencyTrimProtectedUsesHostNamespace(t *testing.T) {
	note := llm.Message{Role: llm.RoleUser, Name: "acme_internal_note", Content: llm.TextContent("host note")}
	if !emergencyTrimProtected("", note) || emergencyTrimProtected("acme", note) {
		t.Fatal("acme-internal note must be droppable only under the acme namespace")
	}
	legacy := llm.Message{Role: llm.RoleUser, Name: "goode_internal_note", Content: llm.TextContent("host note")}
	if emergencyTrimProtected("", legacy) || emergencyTrimProtected("acme", legacy) {
		t.Fatal("legacy goode-internal note must stay droppable under every namespace")
	}
}
