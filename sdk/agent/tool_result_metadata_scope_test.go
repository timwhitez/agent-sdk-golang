package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"sync"
	"testing"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

func metaCall(id, name string, args map[string]any) llm.ToolCall {
	raw, _ := json.Marshal(args)
	return llm.ToolCall{ID: id, Type: "function", Function: llm.FunctionCall{Name: name, Arguments: string(raw)}}
}

// An outer tool call runs two nested agents in parallel inside its handler
// (as a delegation tool does). Each nested tool call owns its metadata
// store: the siblings' metadata does not mix, the nested calls neither write
// into nor drain the outer call's store, and the outer call's own metadata
// reaches its result intact.
func TestNestedParallelToolCallsDoNotShareMetadata(t *testing.T) {
	// Both nested marks are inside their handlers before either returns, so
	// a shared store would hold both keys when each call settles.
	arrived := make(chan struct{}, 2)
	bothStarted := make(chan struct{})
	go func() {
		<-arrived
		<-arrived
		close(bothStarted)
	}()
	mark := tools.Func[struct {
		Child string `json:"child"`
	}]("mark", "record child metadata", func(ctx context.Context, a struct {
		Child string `json:"child"`
	}, _ *tools.Container) (any, error) {
		tools.UpsertToolResultMetadata(ctx, map[string]any{"child_" + a.Child: true})
		arrived <- struct{}{}
		select {
		case <-bothStarted:
		case <-time.After(5 * time.Second):
			return nil, fmt.Errorf("nested calls did not overlap")
		}
		return "marked " + a.Child, nil
	})

	var mu sync.Mutex
	childMeta := map[string]map[string]any{}
	var outerAfterChildren map[string]any
	outer := tools.Tool{Name: "outer", Handler: func(ctx context.Context, _ json.RawMessage, _ *tools.Container) (llm.Content, error) {
		tools.UpsertToolResultMetadata(ctx, map[string]any{"outer": "kept"})
		var wg sync.WaitGroup
		errs := make(chan error, 2)
		for _, name := range []string{"a", "b"} {
			wg.Add(1)
			go func(name string) {
				defer wg.Done()
				child, err := New(Config{
					LLM:   &turnModel{turns: []*llm.Completion{{StopReason: "tool_calls", ToolCalls: []llm.ToolCall{metaCall("child-"+name, "mark", map[string]any{"child": name})}}}},
					Tools: []tools.Tool{mark},
				})
				if err != nil {
					errs <- err
					return
				}
				for event := range child.QueryStream(ctx, llm.TextContent("go")) {
					if result, ok := event.(ToolResultEvent); ok {
						mu.Lock()
						childMeta[name] = result.Metadata
						mu.Unlock()
					}
				}
			}(name)
		}
		wg.Wait()
		close(errs)
		for err := range errs {
			return llm.Content{}, err
		}
		outerAfterChildren = tools.ToolResultMetadataSnapshot(ctx)
		return llm.TextContent("outer done"), nil
	}}

	parent, err := New(Config{
		LLM:   &turnModel{turns: []*llm.Completion{{StopReason: "tool_calls", ToolCalls: []llm.ToolCall{metaCall("outer-1", "outer", map[string]any{})}}}},
		Tools: []tools.Tool{outer},
	})
	if err != nil {
		t.Fatal(err)
	}
	var outerResult *ToolResultEvent
	for event := range parent.QueryStream(context.Background(), llm.TextContent("go")) {
		if result, ok := event.(ToolResultEvent); ok && result.ToolCallID == "outer-1" {
			result := result
			outerResult = &result
		}
	}
	if outerResult == nil || outerResult.IsError {
		t.Fatalf("outer result = %+v", outerResult)
	}
	for _, name := range []string{"a", "b"} {
		want := map[string]any{"child_" + name: true}
		if got := childMeta[name]; !reflect.DeepEqual(got, want) {
			t.Errorf("nested call %s metadata = %v, want %v", name, got, want)
		}
	}
	want := map[string]any{"outer": "kept"}
	if !reflect.DeepEqual(outerAfterChildren, want) {
		t.Errorf("outer store after nested calls = %v, want %v", outerAfterChildren, want)
	}
	if got := outerResult.Metadata["outer"]; got != "kept" {
		t.Errorf("outer result metadata = %v", outerResult.Metadata)
	}
	for key := range outerResult.Metadata {
		if key == "child_a" || key == "child_b" {
			t.Errorf("nested metadata %q leaked into the outer result: %v", key, outerResult.Metadata)
		}
	}
}

// A new scope shadows an enclosing store; WithToolResultMetadata keeps
// sharing it.
func TestToolResultMetadataScopeShadowsEnclosingStore(t *testing.T) {
	outer := tools.WithToolResultMetadata(context.Background())
	tools.UpsertToolResultMetadata(outer, map[string]any{"outer": 1})

	shared := tools.WithToolResultMetadata(outer)
	tools.UpsertToolResultMetadata(shared, map[string]any{"shared": 2})
	if got := tools.ToolResultMetadataSnapshot(outer); !reflect.DeepEqual(got, map[string]any{"outer": 1, "shared": 2}) {
		t.Fatalf("WithToolResultMetadata no longer shares an existing store: %v", got)
	}

	scoped := tools.WithToolResultMetadataScope(outer)
	tools.UpsertToolResultMetadata(scoped, map[string]any{"inner": 3})
	if got := tools.TakeToolResultMetadataSnapshot(scoped); !reflect.DeepEqual(got, map[string]any{"inner": 3}) {
		t.Fatalf("scoped take = %v", got)
	}
	if got := tools.ToolResultMetadataSnapshot(outer); !reflect.DeepEqual(got, map[string]any{"outer": 1, "shared": 2}) {
		t.Fatalf("scope wrote into or drained the enclosing store: %v", fmt.Sprint(got))
	}
}
