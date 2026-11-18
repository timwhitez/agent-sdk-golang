package compaction

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/agent/messageorigin"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

const (
	legacyMarkerPrefix = "<!-- goode-compaction-checkpoint-id="
	legacySchema       = "goode.compaction.material.v1"
)

func namespaceTestHistory() []llm.Message {
	messages := []llm.Message{llm.NewSystemMessage("system"), llm.NewUserMessage("first request")}
	for i := 0; i < 40; i++ {
		messages = append(messages, llm.NewAssistantMessage(fmt.Sprintf("step %02d: %s", i, strings.Repeat("progress ", 40)), nil))
	}
	return append(messages, llm.NewUserMessage("latest request"))
}

func TestCompactionWritesConfiguredHostNamespaceForms(t *testing.T) {
	for _, tc := range []struct{ configured, want string }{
		{"", "agent"},
		{"goode", "goode"},
		{"acme", "acme"},
	} {
		t.Run("ns="+tc.want, func(t *testing.T) {
			model := &promptCaptureModel{response: structuredTestSummary("", "")}
			svc := NewService(&Config{Enabled: true, HostNamespace: tc.configured})
			out, _, err := svc.Compact(context.Background(), model, namespaceTestHistory())
			if err != nil {
				t.Fatalf("Compact: %v", err)
			}
			wantMarker := "<!-- " + tc.want + "-compaction-checkpoint-id="
			wantSchema := tc.want + ".compaction.material.v1"
			summary := out[0].Content.PlainText()
			if !strings.Contains(summary, wantMarker) {
				t.Fatalf("summary marker missing %q:\n%s", wantMarker, summary)
			}
			if _, id, ok := splitSummaryCheckpoint(tc.configured, summary); !ok || id == "" {
				t.Fatalf("written marker does not round-trip: ok=%t id=%q", ok, id)
			}
			if !strings.Contains(model.last[0].Content.PlainText(), "schema "+wantSchema+".") {
				t.Fatalf("instructions do not name schema %q:\n%s", wantSchema, model.last[0].Content.PlainText())
			}
			envelope, framed, err := decodeCompactionMaterialEnvelope(tc.configured, model.last[1].Content.PlainText())
			if err != nil || !framed || envelope.Schema != wantSchema {
				t.Fatalf("material schema = %q framed=%t err=%v, want %q", envelope.Schema, framed, err, wantSchema)
			}
			if tc.want != "goode" && (strings.Contains(summary, "goode") || strings.Contains(model.last[0].Content.PlainText(), "goode") || strings.Contains(model.last[1].Content.PlainText(), "goode")) {
				t.Fatalf("namespace %q still wrote a goode identifier", tc.want)
			}
		})
	}
}

func TestGoodeNamespaceWritesLegacyFormsByteIdentical(t *testing.T) {
	meta := &LedgerSummary{CheckpointID: "abc123"}
	if got, want := withSummaryCheckpoint("goode", "body", meta), WithSummaryPrefix(legacyMarkerPrefix+"abc123 -->\n\nbody"); got != want {
		t.Fatalf("goode marker = %q, want %q", got, want)
	}
	if got := compactionMaterialSchema("goode"); got != legacySchema {
		t.Fatalf("goode schema = %q, want %q", got, legacySchema)
	}
}

func TestLegacyFormsRecognizedUnderEveryNamespace(t *testing.T) {
	source := []llm.Message{llm.NewUserMessage("first request"), llm.NewAssistantMessage("first answer", nil)}
	meta := nextLedgerSummary(nil, source, "legacy summary", "")
	legacyText := DefaultSummaryPrefix + legacyMarkerPrefix + meta.CheckpointID + " -->\n\nlegacy summary"
	legacyMaterial := wrapUntrustedMaterial(`{"schema":"` + legacySchema + `","material":"body"}`)
	for _, ns := range []string{"", "goode", "acme"} {
		summary, id, ok := splitSummaryCheckpoint(ns, legacyText)
		if !ok || id != meta.CheckpointID || summary != "legacy summary" {
			t.Fatalf("ns=%q legacy marker: ok=%t id=%q summary=%q", ns, ok, id, summary)
		}
		history := []llm.Message{{Role: llm.RoleUser, Name: CompactionSummaryMessageName, Content: llm.TextContent(legacyText)}}
		if err := ValidateLedgerSummaryBindingIn(ns, history, meta); err != nil {
			t.Fatalf("ns=%q legacy binding: %v", ns, err)
		}
		if envelope, framed, err := decodeCompactionMaterialEnvelope(ns, legacyMaterial); err != nil || !framed || envelope.Material != "body" {
			t.Fatalf("ns=%q legacy schema: framed=%t err=%v", ns, framed, err)
		}
	}
	if err := ValidateLedgerSummaryBinding([]llm.Message{{Role: llm.RoleUser, Name: CompactionSummaryMessageName, Content: llm.TextContent(legacyText)}}, meta); err != nil {
		t.Fatalf("default exported binding rejected legacy marker: %v", err)
	}
}

func TestOtherNamespaceFormsAreNotRecognized(t *testing.T) {
	source := []llm.Message{llm.NewUserMessage("first request")}
	meta := nextLedgerSummary(nil, source, "summary", "")
	foreign := withSummaryCheckpoint("other", "summary", meta)
	foreignMaterial := wrapUntrustedMaterial(`{"schema":"other.compaction.material.v1","material":"body"}`)
	for _, ns := range []string{"", "goode", "acme"} {
		if _, _, ok := splitSummaryCheckpoint(ns, foreign); ok {
			t.Fatalf("ns=%q recognized a foreign-namespace marker", ns)
		}
		history := []llm.Message{{Role: llm.RoleUser, Name: CompactionSummaryMessageName, Content: llm.TextContent(foreign)}}
		if err := ValidateLedgerSummaryBindingIn(ns, history, meta); err == nil {
			t.Fatalf("ns=%q bound a foreign-namespace marker", ns)
		}
		if _, _, err := decodeCompactionMaterialEnvelope(ns, foreignMaterial); err == nil || !strings.Contains(err.Error(), "unsupported compaction material schema") {
			t.Fatalf("ns=%q accepted a foreign schema: %v", ns, err)
		}
	}
}

func TestInvalidHostNamespaceFallsBackToDefaultWithWarning(t *testing.T) {
	var warnings []string
	svc := NewService(&Config{Enabled: true, HostNamespace: "x -->", Warningf: func(format string, args ...any) {
		warnings = append(warnings, fmt.Sprintf(format, args...))
	}})
	if svc.Config.HostNamespace != "" || svc.hostNamespace() != "agent" {
		t.Fatalf("invalid namespace kept: config=%q resolved=%q", svc.Config.HostNamespace, svc.hostNamespace())
	}
	if len(warnings) != 1 || !strings.Contains(warnings[0], "host namespace") {
		t.Fatalf("warnings = %#v", warnings)
	}
}

func TestServicesWithDifferentNamespacesDoNotInterfere(t *testing.T) {
	history := []llm.Message{
		llm.NewUserMessage("real request"),
		llm.NewAssistantMessage("working", nil),
		{Role: llm.RoleUser, Name: "beta_internal_note", Content: llm.TextContent("beta host note")},
	}
	acme := NewService(&Config{Enabled: true, HostNamespace: "acme"})
	beta := NewService(&Config{Enabled: true, HostNamespace: "beta"})
	latest := map[string]string{}
	var mu sync.Mutex
	var wg sync.WaitGroup
	for name, svc := range map[string]*Service{"acme": acme, "beta": beta} {
		wg.Add(1)
		go func(name string, svc *Service) {
			defer wg.Done()
			for i := 0; i < 50; i++ {
				envelope, _, err := decodeCompactionMaterialEnvelope(svc.Config.HostNamespace, svc.buildCompactionInput(history, nil, 1, ""))
				if err != nil || envelope.LatestRealUserRequest == nil || envelope.Schema != name+".compaction.material.v1" {
					t.Errorf("%s: envelope=%#v err=%v", name, envelope, err)
					return
				}
				mu.Lock()
				latest[name] = *envelope.LatestRealUserRequest
				mu.Unlock()
			}
		}(name, svc)
	}
	wg.Wait()
	// acme does not know beta's names, so beta's note is plain user content
	// there; beta classifies its own name as internal.
	if latest["acme"] != "beta host note" || latest["beta"] != "real request" {
		t.Fatalf("latest real user anchors = %#v", latest)
	}
}

func TestInvalidNamespaceResolvesLikeDefaultInExportedHelpers(t *testing.T) {
	source := []llm.Message{llm.NewUserMessage("first request")}
	meta := nextLedgerSummary(nil, source, "summary", "")
	summary := llm.Message{Role: llm.RoleUser, Name: CompactionSummaryMessageName, Content: llm.TextContent(withSummaryCheckpoint("", "summary", meta))}
	agentNote := llm.Message{Role: llm.RoleUser, Name: "agent_internal_note", Content: llm.TextContent("host note")}
	// The agent-internal note may precede the summary only if it is internal.
	history := []llm.Message{agentNote, summary}
	for _, ns := range []string{"Bad", "sdk", "x -->", "a_b", strings.Repeat("a", 33)} {
		if got := messageorigin.ResolveHostNamespace(ns); got != messageorigin.DefaultHostNamespace {
			t.Fatalf("ResolveHostNamespace(%q) = %q", ns, got)
		}
		if !messageorigin.IsInternalMessageIn(ns, agentNote) || messageorigin.IsRealUserMessageIn(ns, agentNote) {
			t.Fatalf("ns=%q does not classify agent_internal_* like the default namespace", ns)
		}
		if err := ValidateLedgerSummaryBindingIn(ns, history, meta); err != nil {
			t.Fatalf("ns=%q binding differs from default: %v", ns, err)
		}
		if got := compactionMaterialSchema(ns); got != "agent.compaction.material.v1" {
			t.Fatalf("ns=%q schema = %q", ns, got)
		}
	}
	if err := ValidateLedgerSummaryBindingIn("", history, meta); err != nil {
		t.Fatalf("default binding: %v", err)
	}
}
