package execenv

import (
	"context"
	"os"
	"runtime"
	"strings"

	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

// ExtraAllowlistKey allows trusted callers to expose additional env vars to command tools.
// Values are copied from the parent process only by variable name (never inline values).
var ExtraAllowlistKey = tools.Dep[[]string]("exec_env_allowlist")

var unixDefaultAllowlist = []string{
	"PATH",
	"HOME",
	"USER",
	"LOGNAME",
	"SHELL",
	"TMPDIR",
	"LANG",
	"LC_ALL",
	"LC_CTYPE",
	"TERM",
	"TZ",
}

var windowsDefaultAllowlist = []string{
	"COMSPEC",
	"PATH",
	"PATHEXT",
	"SYSTEMROOT",
	"SYSTEMDRIVE",
	"WINDIR",
	"TEMP",
	"TMP",
	"USERPROFILE",
	"HOMEDRIVE",
	"HOMEPATH",
}

// EnvFromDeps builds the subprocess environment using the default allowlist
// plus optional trusted extras provided via ExtraAllowlistKey.
func EnvFromDeps(ctx context.Context, deps *tools.Container) []string {
	return Build(ExtraAllowlistFromDeps(ctx, deps))
}

// ExtraAllowlistFromDeps returns the normalized extra allowlist from deps.
func ExtraAllowlistFromDeps(ctx context.Context, deps *tools.Container) []string {
	extra, err := tools.Get(deps, ctx, ExtraAllowlistKey)
	if err != nil {
		return nil
	}
	return NormalizeAllowlist(extra)
}

// NormalizeAllowlist canonicalizes and deduplicates env names.
func NormalizeAllowlist(names []string) []string {
	if len(names) == 0 {
		return nil
	}
	seen := map[string]struct{}{}
	out := make([]string, 0, len(names))
	for _, raw := range names {
		name := normalizeEnvName(raw)
		if name == "" {
			continue
		}
		key := dedupeKey(name)
		if _, ok := seen[key]; ok {
			continue
		}
		seen[key] = struct{}{}
		out = append(out, name)
	}
	return out
}

// Build constructs cmd.Env entries from default allowlist + normalized extras.
func Build(extraAllowlist []string) []string {
	defaults := defaultAllowlist()
	names := make([]string, 0, len(defaults)+len(extraAllowlist))
	names = append(names, defaults...)
	names = append(names, NormalizeAllowlist(extraAllowlist)...)

	seen := map[string]struct{}{}
	out := make([]string, 0, len(names))
	for _, name := range names {
		key := dedupeKey(name)
		if _, ok := seen[key]; ok {
			continue
		}
		seen[key] = struct{}{}
		if value, ok := os.LookupEnv(name); ok {
			out = append(out, name+"="+value)
		}
	}
	return out
}

func defaultAllowlist() []string {
	if runtime.GOOS == "windows" {
		return append([]string(nil), windowsDefaultAllowlist...)
	}
	return append([]string(nil), unixDefaultAllowlist...)
}

func normalizeEnvName(raw string) string {
	name := strings.TrimSpace(raw)
	if name == "" {
		return ""
	}
	if strings.ContainsRune(name, '=') || strings.ContainsRune(name, 0) {
		return ""
	}
	return name
}

func dedupeKey(name string) string {
	if runtime.GOOS == "windows" {
		return strings.ToUpper(name)
	}
	return name
}
