package execenv

import (
	"context"
	"os"
	"runtime"
	"sort"
	"strings"

	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

// ExtraAllowlistKey allows trusted callers to expose additional env vars to command tools.
// Values are copied from the parent process only by variable name (never inline values).
var ExtraAllowlistKey = tools.Dep[[]string]("exec_env_allowlist")

// AllowlistHint tells the model how to recover when a build fails because a
// required env var was pruned by the allowlist.
const AllowlistHint = "Command environment is allowlisted, not inherited; request extra variables via the exec_env_allowlist dependency."

// maxReportedPrunedNames bounds the pruned-name list attached to tool metadata.
const maxReportedPrunedNames = 24

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

// toolchainEnvPrefixes lists env name prefixes that commonly break builds when
// withheld (toolchain caches, registries, proxies). Matched case-insensitively
// against upper-cased names, so lowercase conventions such as npm_config_* and
// http_proxy are covered too.
var toolchainEnvPrefixes = []string{
	"GO",
	"NPM_",
	"NODE_",
	"PNPM_",
	"YARN_",
	"PYTHON",
	"PIP_",
	"POETRY_",
	"UV_",
	"CONDA_",
	"JAVA_",
	"MAVEN_",
	"GRADLE_",
	"CARGO_",
	"RUST",
	"PKG_CONFIG_",
	"DOCKER_",
}

// toolchainEnvNames lists individual env names with the same effect.
var toolchainEnvNames = []string{
	"CC",
	"CXX",
	"CFLAGS",
	"CXXFLAGS",
	"LDFLAGS",
	"MAKEFLAGS",
	"VIRTUAL_ENV",
	"KUBECONFIG",
	"NO_PROXY",
}

// EnvFromDeps builds the subprocess environment using the default allowlist
// plus optional trusted extras provided via ExtraAllowlistKey. The resulting
// allowlist decision is reported through tool result metadata so a build that
// fails because of a pruned variable does not fail silently.
func EnvFromDeps(ctx context.Context, deps *tools.Container) []string {
	extra := ExtraAllowlistFromDeps(ctx, deps)
	env, exported := build(extra)
	tools.UpsertToolResultMetadata(ctx, Diagnostics(extra, exported))
	return env
}

// Diagnostics describes how the subprocess environment was derived: which
// variables were exported, which build-relevant ones were withheld, and how to
// request more. exported is the list of exported names (see Build); pass nil to
// have it recomputed.
func Diagnostics(extraAllowlist []string, exported []string) map[string]any {
	if exported == nil {
		_, exported = build(extraAllowlist)
	}
	meta := map[string]any{"env_allowlisted": exported}
	if pruned := PrunedToolchainEnv(extraAllowlist); len(pruned) > 0 {
		meta["env_pruned"] = pruned
		meta["env_allowlist_hint"] = AllowlistHint
	}
	return meta
}

// PrunedToolchainEnv returns the names (never values) of toolchain, registry or
// proxy env vars that exist in the parent process but are withheld from the
// subprocess by the allowlist. The result is sorted and bounded.
func PrunedToolchainEnv(extraAllowlist []string) []string {
	allowed := map[string]struct{}{}
	for _, name := range effectiveAllowlist(extraAllowlist) {
		allowed[dedupeKey(name)] = struct{}{}
	}
	var out []string
	for _, entry := range os.Environ() {
		name := entry
		if idx := strings.IndexByte(entry, '='); idx >= 0 {
			name = entry[:idx]
		}
		if name == "" {
			continue
		}
		if _, ok := allowed[dedupeKey(name)]; ok {
			continue
		}
		if !isToolchainEnvName(name) {
			continue
		}
		out = append(out, name)
	}
	sort.Strings(out)
	if len(out) > maxReportedPrunedNames {
		out = out[:maxReportedPrunedNames]
	}
	return out
}

// secretEnvNameMarkers keeps credential-shaped names out of the reported
// pruned list: names are diagnostics, and even a name should not hint at which
// secrets the parent process holds.
var secretEnvNameMarkers = []string{
	"KEY",
	"TOKEN",
	"SECRET",
	"PASSWORD",
	"PASSWD",
	"CREDENTIAL",
	"AUTH",
	"COOKIE",
	"SESSION",
	"PRIVATE",
	"SIGNATURE",
}

func isToolchainEnvName(name string) bool {
	upper := strings.ToUpper(name)
	for _, marker := range secretEnvNameMarkers {
		if strings.Contains(upper, marker) {
			return false
		}
	}
	if strings.HasSuffix(upper, "_PROXY") {
		return true
	}
	for _, exact := range toolchainEnvNames {
		if upper == exact {
			return true
		}
	}
	for _, prefix := range toolchainEnvPrefixes {
		if strings.HasPrefix(upper, prefix) {
			return true
		}
	}
	return false
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
	env, _ := build(extraAllowlist)
	return env
}

// build returns cmd.Env entries plus the names actually exported (a variable is
// only exported when it is present in the parent process).
func build(extraAllowlist []string) (env []string, exported []string) {
	names := effectiveAllowlist(extraAllowlist)
	env = make([]string, 0, len(names))
	exported = make([]string, 0, len(names))
	for _, name := range names {
		if value, ok := os.LookupEnv(name); ok {
			env = append(env, name+"="+value)
			exported = append(exported, name)
		}
	}
	return env, exported
}

// effectiveAllowlist returns the deduplicated default allowlist plus normalized
// extras, in precedence order.
func effectiveAllowlist(extraAllowlist []string) []string {
	defaults := defaultAllowlist()
	extras := NormalizeAllowlist(extraAllowlist)
	names := make([]string, 0, len(defaults)+len(extras))
	names = append(names, defaults...)
	names = append(names, extras...)

	seen := map[string]struct{}{}
	out := make([]string, 0, len(names))
	for _, name := range names {
		key := dedupeKey(name)
		if _, ok := seen[key]; ok {
			continue
		}
		seen[key] = struct{}{}
		out = append(out, name)
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
