package agent

import (
	"container/list"
	"context"
	"crypto/sha256"
	"encoding/json"
	"sync"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

// Match only the existing provider printf contract and the actual request.
// Unknown diagnostics and host-generated summaries always pass through.
const schemaCompatibilityWarningFormat = "OpenAI tool %q uses non-strict parameters to preserve its schema: %s"
const schemaWarningCapacity = 1024

type schemaWarningFingerprint struct{ tool, cause, schema [32]byte }
type schemaWarningDeclaration struct {
	cause string
	key   schemaWarningFingerprint
}

// Display history belongs to an Agent, not its shared provider client. Only
// bounded hashes survive an invocation; eviction can repeat an old diagnostic.
type schemaWarningGate struct {
	mu     sync.Mutex
	seen   map[schemaWarningFingerprint]*list.Element
	recent list.List
}

func (g *schemaWarningGate) suppress(key schemaWarningFingerprint) bool {
	g.mu.Lock()
	defer g.mu.Unlock()
	if entry, ok := g.seen[key]; ok {
		g.recent.MoveToFront(entry)
		return true
	}
	if g.seen == nil {
		g.seen = make(map[schemaWarningFingerprint]*list.Element)
	}
	if len(g.seen) >= schemaWarningCapacity {
		entry := g.recent.Back()
		delete(g.seen, entry.Value.(schemaWarningFingerprint))
		entry.Value = key
		g.recent.MoveToFront(entry)
		g.seen[key] = entry
	} else {
		g.seen[key] = g.recent.PushFront(key)
	}
	return false
}

func (g *schemaWarningGate) bind(ctx context.Context, req llm.InvokeRequest, sink func(string, ...any)) context.Context {
	counts := make(map[string]int)
	for _, d := range req.Tools {
		counts[d.Name]++
	}
	keys := make(map[string]schemaWarningDeclaration)
	for _, d := range req.Tools {
		if d.Strict || d.StrictWarning == "" || d.Name == "" || counts[d.Name] != 1 {
			continue
		}
		// Never invoke user marshalers for display classification. If the
		// declaration cannot be represented safely, leave its warning visible.
		owned, err := llm.CloneStaticJSONMap(d.Parameters)
		if err != nil {
			continue
		}
		encoded, err := json.Marshal(owned)
		if err != nil {
			continue
		}
		keys[d.Name] = schemaWarningDeclaration{d.StrictWarning, schemaWarningFingerprint{sha256.Sum256([]byte(d.Name)), sha256.Sum256([]byte(d.StrictWarning)), sha256.Sum256(encoded)}}
	}
	return llm.WithWarningSink(ctx, func(format string, args ...any) {
		if format == schemaCompatibilityWarningFormat && len(args) == 2 {
			tool, toolOK := args[0].(string)
			cause, causeOK := args[1].(string)
			if d, ok := keys[tool]; ok && toolOK && causeOK && d.cause == cause && g.suppress(d.key) {
				return
			}
		}
		sink(format, args...)
	})
}
