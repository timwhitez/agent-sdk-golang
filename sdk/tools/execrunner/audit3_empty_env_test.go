package execrunner

import (
	"os"
	"os/exec"
	"runtime"
	"testing"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/tools/execenv"
)

func TestAuditExplicitEmptyEnvironmentDoesNotInheritParent(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("POSIX shell fixture")
	}
	t.Setenv("AGENTSDK_AUDIT_ENV_CANARY", "must-not-inherit")
	for _, tc := range []struct {
		name string
		env  []string
		want string
	}{
		{"nil_inherits", nil, "must-not-inherit"},
		{"explicit_empty", []string{}, ""},
		{"explicit_value", []string{"AGENTSDK_AUDIT_ENV_CANARY=provided"}, "provided"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			result, err := Run(t.Context(), Options{Program: "/bin/sh", Args: []string{"-c", `printf '%s' "$AGENTSDK_AUDIT_ENV_CANARY"`}, Env: tc.env, Timeout: 2 * time.Second})
			if err != nil || result.Output != tc.want {
				t.Fatalf("output=%q err=%v want=%q", result.Output, err, tc.want)
			}
		})
	}
}

func TestRunEmptyAllowlistDoesNotInheritParent(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("POSIX shell fixture")
	}
	const helper = "AGENTSDK_AUDIT_EMPTY_ALLOWLIST"
	if os.Getenv(helper) != "1" {
		// Isolate the fixture in a process with no default allowlisted variables;
		// changing the environment of the main test process is unnecessary.
		cmd := exec.CommandContext(t.Context(), os.Args[0], "-test.run=^TestRunEmptyAllowlistDoesNotInheritParent$")
		cmd.Env = []string{helper + "=1", "AGENTSDK_AUDIT_ENV_CANARY=must-not-inherit"}
		if output, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("empty allowlist fixture: %v\n%s", err, output)
		}
		return
	}
	env := execenv.Build(nil)
	if env == nil || len(env) != 0 {
		t.Fatalf("Build without any allowlisted variables returned nil=%v, len=%d", env == nil, len(env))
	}
	result, err := Run(t.Context(), Options{
		Program: "/bin/sh",
		Args:    []string{"-c", `printf '%s' "$AGENTSDK_AUDIT_ENV_CANARY"`},
		Env:     env,
		Timeout: 2 * time.Second,
	})
	if err != nil || result.Output != "" {
		t.Fatalf("allowlisted command inherited canary: output=%q err=%v", result.Output, err)
	}
}
