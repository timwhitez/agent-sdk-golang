package execenv

import (
	"context"
	"strings"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

func TestBuild_DoesNotIncludeNonAllowlistedSensitiveEnv(t *testing.T) {
	secretName := "NEW006_SENSITIVE_TOKEN"
	secretValue := "super-secret"
	t.Setenv(secretName, secretValue)

	got := envMap(Build(nil))
	if _, ok := got[secretName]; ok {
		t.Fatalf("expected %s to be excluded from default allowlist", secretName)
	}
}

func TestBuild_IncludesExplicitAllowlistedEnv(t *testing.T) {
	name := "NEW006_VISIBLE_ENV"
	value := "visible"
	t.Setenv(name, value)

	got := envMap(Build([]string{" " + name + " ", name, "IGNORED=VALUE"}))
	if got[name] != value {
		t.Fatalf("expected %s=%q, got %q", name, value, got[name])
	}
	if _, ok := got["IGNORED"]; ok {
		t.Fatalf("expected malformed allowlist entry to be ignored")
	}
}

func TestExtraAllowlistFromDeps_NormalizesEntries(t *testing.T) {
	deps := tools.NewContainer()
	tools.Provide(deps, ExtraAllowlistKey, func(context.Context) ([]string, error) {
		return []string{" FOO ", "FOO", "", "BAR=1", "BAR"}, nil
	})

	got := ExtraAllowlistFromDeps(context.Background(), deps)
	if len(got) != 2 || got[0] != "FOO" || got[1] != "BAR" {
		t.Fatalf("unexpected allowlist: %v", got)
	}
}

func envMap(env []string) map[string]string {
	out := make(map[string]string, len(env))
	for _, item := range env {
		parts := strings.SplitN(item, "=", 2)
		if len(parts) != 2 {
			continue
		}
		out[parts[0]] = parts[1]
	}
	return out
}
