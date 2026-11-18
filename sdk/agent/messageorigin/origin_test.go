package messageorigin

import (
	"strings"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

func TestNewInternalUserMessageCarriesStableName(t *testing.T) {
	message := NewInternalUserMessage(KindRequireDone, "reminder")
	if message.Role != llm.RoleUser || message.Name != "sdk_internal_require_done" || message.Content.PlainText() != "reminder" {
		t.Fatalf("internal message = %#v", message)
	}
	if !IsInternalMessage(message) || IsRealUserMessage(message) {
		t.Fatalf("internal message classification = internal:%t real:%t", IsInternalMessage(message), IsRealUserMessage(message))
	}
}

func TestIsRealUserMessagePreservesUnknownNamedUsers(t *testing.T) {
	message := llm.Message{Role: llm.RoleUser, Name: "customer_alias", Content: llm.TextContent(RequireDoneReminderText + " quoted")}
	if !IsRealUserMessage(message) {
		t.Fatalf("unknown named user was classified as internal: %#v", message)
	}
}

func TestIsRealUserMessageRejectsDestroyedUser(t *testing.T) {
	message := llm.NewUserMessage("released")
	message.Destroyed = true
	if IsRealUserMessage(message) {
		t.Fatalf("destroyed user was classified as real: %#v", message)
	}
}

func TestValidateHostNamespace(t *testing.T) {
	for _, ns := range []string{"agent", "goode", "acme2", strings.Repeat("a", 32)} {
		if err := ValidateHostNamespace(ns); err != nil {
			t.Fatalf("ValidateHostNamespace(%q) = %v", ns, err)
		}
	}
	for _, ns := range []string{"", "sdk", "Acme", "a-b", "a_b", "a.b", "1abc", "a b", "x -->", "a\n", "é", strings.Repeat("a", 33)} {
		if err := ValidateHostNamespace(ns); err == nil {
			t.Fatalf("ValidateHostNamespace(%q) accepted", ns)
		}
	}
}

func TestHostNamespaceClassification(t *testing.T) {
	named := func(name string) llm.Message {
		return llm.Message{Role: llm.RoleUser, Name: name, Content: llm.TextContent("host text")}
	}
	cases := []struct {
		name     string
		internal map[string]bool // namespace -> want internal
	}{
		{"goode_internal_collab_history_repair", map[string]bool{"": true, "goode": true, "acme": true, "a_b": true}},
		{"goode_memory", map[string]bool{"": true, "goode": true, "acme": true, "a_b": true}},
		{"sdk_internal_require_done", map[string]bool{"": true, "goode": true, "acme": true}},
		{"agent_internal_x", map[string]bool{"": true, "agent": true, "goode": false, "acme": false}},
		{"agent_memory", map[string]bool{"": true, "goode": false, "acme": false}},
		{"acme_internal_x", map[string]bool{"": false, "goode": false, "acme": true, "acme2": false}},
		{"acme_memory", map[string]bool{"": false, "acme": true}},
		// An invalid or reserved namespace resolves to the default namespace.
		{"a_b_internal_x", map[string]bool{"a_b": false}},
		{"agent_internal_x", map[string]bool{"a_b": true, "sdk": true, "Bad": true}},
		{"agent_memory", map[string]bool{"a_b": true, "sdk": true}},
		{"sdk_memory", map[string]bool{"sdk": false, "": false}},
		{"_internal_x", map[string]bool{"a_b": false, "": false}},
		{"acme_internalx", map[string]bool{"acme": false}},
	}
	for _, tc := range cases {
		for ns, want := range tc.internal {
			msg := named(tc.name)
			if got := IsInternalMessageIn(ns, msg); got != want {
				t.Errorf("IsInternalMessageIn(%q, %q) = %t, want %t", ns, tc.name, got, want)
			}
			if got := IsRealUserMessageIn(ns, msg); got == want {
				t.Errorf("IsRealUserMessageIn(%q, %q) = %t, want %t", ns, tc.name, got, !want)
			}
		}
	}
	// Namespace forms are name-only: identical text without a name stays a real user request.
	if !IsRealUserMessageIn("acme", llm.NewUserMessage("acme_internal_x")) {
		t.Fatal("unnamed text matching a namespace form was classified internal")
	}
	if IsInternalMessage(named("acme_internal_x")) != IsInternalMessageIn("", named("acme_internal_x")) {
		t.Fatal("IsInternalMessage differs from the default namespace")
	}
}

func TestLegacyHostConstantsAreByteIdentical(t *testing.T) {
	if LegacyHostInternalNamePrefix != "goode_internal_" || LegacyHostMemoryName != "goode_memory" ||
		HostInternalNamePrefix != LegacyHostInternalNamePrefix || GoodeMemoryName != LegacyHostMemoryName {
		t.Fatal("legacy host identifiers changed")
	}
}
