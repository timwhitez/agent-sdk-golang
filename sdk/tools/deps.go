package tools

import (
	"context"
	"fmt"
	"strings"
	"sync"
)

// DepKey identifies a dependency.
// Use a stable string name (e.g. "sandbox") to match override semantics.
type DepKey[T any] struct{ Name string }

func Dep[T any](name string) DepKey[T] { return DepKey[T]{Name: name} }

type Provider[T any] func(ctx context.Context) (T, error)

// ---- tool call context ----

type ctxKey string

const (
	toolCallIDKey ctxKey = "tools.tool_call_id"
	toolResultMetaKey ctxKey = "tools.tool_result_meta"
)

// WithToolCallID attaches a tool_call_id to the context for tool handlers.
// Interactive clients can use it to correlate confirmations/previews/results.
func WithToolCallID(ctx context.Context, id string) context.Context {
	id = strings.TrimSpace(id)
	if ctx == nil || id == "" {
		return ctx
	}
	return context.WithValue(ctx, toolCallIDKey, id)
}

func ToolCallID(ctx context.Context) string {
	if ctx == nil {
		return ""
	}
	v, _ := ctx.Value(toolCallIDKey).(string)
	return strings.TrimSpace(v)
}

// ---- tool result metadata ----

// ToolResultMetadata is an ephemeral metadata bag tools can attach to their result.
// Interactive clients may use it to enhance UI without polluting the LLM-visible content.
//
// NOTE: This is intentionally best-effort and not persisted in llm.Message.
type ToolResultMetadata struct {
	mu sync.Mutex
	m  map[string]any
}

// WithToolResultMetadata attaches a mutable metadata store to the context.
// Tools can call Set/Upsert helpers to record metadata for the current tool call.
func WithToolResultMetadata(ctx context.Context) context.Context {
	if ctx == nil {
		return ctx
	}
	if _, ok := ctx.Value(toolResultMetaKey).(*ToolResultMetadata); ok {
		return ctx
	}
	return context.WithValue(ctx, toolResultMetaKey, &ToolResultMetadata{m: map[string]any{}})
}

func toolResultMeta(ctx context.Context) *ToolResultMetadata {
	if ctx == nil {
		return nil
	}
	v, _ := ctx.Value(toolResultMetaKey).(*ToolResultMetadata)
	return v
}

// SetToolResultMetadata replaces metadata for the current tool call.
func SetToolResultMetadata(ctx context.Context, meta map[string]any) {
	s := toolResultMeta(ctx)
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if meta == nil {
		s.m = map[string]any{}
		return
	}
	out := make(map[string]any, len(meta))
	for k, v := range meta {
		if strings.TrimSpace(k) == "" {
			continue
		}
		out[k] = v
	}
	s.m = out
}

// UpsertToolResultMetadata merges metadata for the current tool call.
func UpsertToolResultMetadata(ctx context.Context, meta map[string]any) {
	s := toolResultMeta(ctx)
	if s == nil || meta == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.m == nil {
		s.m = map[string]any{}
	}
	for k, v := range meta {
		kk := strings.TrimSpace(k)
		if kk == "" {
			continue
		}
		s.m[kk] = v
	}
}

// ToolResultMetadataSnapshot returns a defensive copy of current tool metadata.
func ToolResultMetadataSnapshot(ctx context.Context) map[string]any {
	s := toolResultMeta(ctx)
	if s == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.m) == 0 {
		return nil
	}
	out := make(map[string]any, len(s.m))
	for k, v := range s.m {
		out[k] = v
	}
	return out
}

// TakeToolResultMetadataSnapshot returns metadata and clears it.
func TakeToolResultMetadataSnapshot(ctx context.Context) map[string]any {
	s := toolResultMeta(ctx)
	if s == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.m) == 0 {
		return nil
	}
	out := make(map[string]any, len(s.m))
	for k, v := range s.m {
		out[k] = v
	}
	// clear
	s.m = map[string]any{}
	return out
}

// Container resolves dependencies using registered providers with optional overrides.
// Resolved values are memoized per Container instance.
type Container struct {
	mu        sync.Mutex
	providers map[string]any
	overrides map[string]any
	cache     map[string]any
}

func NewContainer() *Container {
	return &Container{providers: map[string]any{}, overrides: map[string]any{}, cache: map[string]any{}}
}

func (c *Container) ProvideAny(name string, provider any) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.providers[name] = provider
}

func (c *Container) OverrideAny(name string, provider any) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.overrides[name] = provider
	delete(c.cache, name)
}

func Provide[T any](c *Container, key DepKey[T], p Provider[T]) { c.ProvideAny(key.Name, p) }

func Override[T any](c *Container, key DepKey[T], p Provider[T]) { c.OverrideAny(key.Name, p) }

func Get[T any](c *Container, ctx context.Context, key DepKey[T]) (T, error) {
	var zero T
	c.mu.Lock()
	if v, ok := c.cache[key.Name]; ok {
		c.mu.Unlock()
		vv, ok := v.(T)
		if !ok {
			return zero, fmt.Errorf("dependency %q has unexpected type", key.Name)
		}
		return vv, nil
	}
	provAny, ok := c.overrides[key.Name]
	if !ok {
		provAny, ok = c.providers[key.Name]
	}
	c.mu.Unlock()
	if !ok {
		return zero, fmt.Errorf("missing dependency provider: %q", key.Name)
	}
	prov, ok := provAny.(Provider[T])
	if !ok {
		return zero, fmt.Errorf("dependency %q provider has incompatible type", key.Name)
	}
	v, err := prov(ctx)
	if err != nil {
		return zero, err
	}
	c.mu.Lock()
	c.cache[key.Name] = v
	c.mu.Unlock()
	return v, nil
}
