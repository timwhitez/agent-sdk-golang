package agent

import (
	"sort"
	"strings"

	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

var toolAliasCandidates = map[string][]string{
	"ls":          {"list"},
	"list":        {"ls"},
	"list_dir":    {"list", "ls"},
	"read":        {"read_file"},
	"read_file":   {"read"},
	"grep":        {"grep_files"},
	"grep_files":  {"grep"},
	"apply_patch": {"patch"},
	"patch":       {"apply_patch"},
	"webfetch":    {"web_request"},
	"web_request": {"webfetch"},
	"skill":       {"skill_read"},
	"skill_read":  {"skill"},
	"todoread":    {"todo_read"},
	"todo_read":   {"todoread"},
	"todowrite":   {"todo_write"},
	"todo_write":  {"todowrite"},
}

func buildNormalizedToolMap(toolMap map[string]tools.Tool, ordered []tools.Tool) map[string]tools.Tool {
	out := map[string]tools.Tool{}
	if len(toolMap) == 0 {
		return out
	}
	addTool := func(name string, tool tools.Tool) {
		norm := tools.NormalizeToolName(name)
		if norm == "" {
			return
		}
		if _, ok := out[norm]; !ok {
			out[norm] = tool
		}
	}
	seen := map[string]struct{}{}
	if len(ordered) > 0 {
		for _, tool := range ordered {
			name := strings.TrimSpace(tool.Name)
			if name == "" {
				continue
			}
			if mapped, ok := toolMap[name]; ok {
				addTool(name, mapped)
			} else {
				addTool(name, tool)
			}
			seen[name] = struct{}{}
		}
	}
	if len(ordered) == 0 || len(seen) < len(toolMap) {
		remaining := make([]string, 0, len(toolMap))
		for name := range toolMap {
			if _, ok := seen[name]; ok {
				continue
			}
			remaining = append(remaining, name)
		}
		sort.Strings(remaining)
		for _, name := range remaining {
			addTool(name, toolMap[name])
		}
	}
	for alias, candidates := range toolAliasCandidates {
		aliasNorm := tools.NormalizeToolName(alias)
		if aliasNorm == "" {
			continue
		}
		if _, ok := out[aliasNorm]; ok {
			continue
		}
		if tool, ok := pickToolCandidate(candidates, toolMap, out); ok {
			out[aliasNorm] = tool
		}
	}
	return out
}

func pickToolCandidate(candidates []string, toolMap map[string]tools.Tool, normMap map[string]tools.Tool) (tools.Tool, bool) {
	for _, cand := range candidates {
		cand = strings.TrimSpace(cand)
		if cand == "" {
			continue
		}
		if tool, ok := toolMap[cand]; ok {
			return tool, true
		}
		norm := tools.NormalizeToolName(cand)
		if norm == "" {
			continue
		}
		if tool, ok := normMap[norm]; ok {
			return tool, true
		}
	}
	return tools.Tool{}, false
}

func (a *Agent) resolveToolByName(name string) (tools.Tool, string, bool, bool) {
	raw := strings.TrimSpace(name)
	if raw == "" {
		return tools.Tool{}, "", false, false
	}
	if tool, ok := a.toolMap[raw]; ok {
		return tool, raw, true, false
	}
	norm := tools.NormalizeToolName(raw)
	if norm == "" {
		return tools.Tool{}, "", false, false
	}
	if tool, ok := a.toolMapNormalized[norm]; ok {
		return tool, strings.TrimSpace(tool.Name), true, true
	}
	return tools.Tool{}, "", false, false
}

func (a *Agent) hasToolNamed(name string) bool {
	if a == nil {
		return false
	}
	raw := strings.TrimSpace(name)
	if raw == "" {
		return false
	}
	if _, ok := a.toolMap[raw]; ok {
		return true
	}
	norm := tools.NormalizeToolName(raw)
	if norm == "" {
		return false
	}
	_, ok := a.toolMapNormalized[norm]
	return ok
}
