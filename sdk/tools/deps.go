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
