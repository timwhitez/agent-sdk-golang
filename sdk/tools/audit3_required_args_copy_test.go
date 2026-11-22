package tools

import (
	"context"
	"strings"
	"sync"
	"testing"
)

func TestAuditRequiredPreparedCopiesOwnTypedArguments(t *testing.T) {
	type args struct {
		Options map[string]string `json:"options"`
	}
	var seen []string
	tool := Func[args]("record", "record", func(_ context.Context, input args, _ *Container) (any, error) {
		seen = append(seen, input.Options["mode"])
		input.Options["mode"] = "mutated by earlier invocation"
		return "ok", nil
	})
	prepared, _ := tool.PrepareCall(`{"options":{"mode":"planned"}}`)
	first, second := prepared.RequireFinalArgs(), prepared.RequireFinalArgs()
	if _, err := first.Execute(t.Context(), nil); err != nil {
		t.Fatal(err)
	}
	view, ok := second.FinalArgs()
	if !ok || !strings.Contains(string(view), `"planned"`) {
		t.Fatalf("second plan view=%s available=%v", view, ok)
	}
	if _, err := second.Execute(t.Context(), nil); err != nil {
		t.Fatal(err)
	}
	if _, err := prepared.Execute(t.Context(), nil); err != nil {
		t.Fatal(err)
	}
	if _, err := prepared.RequireFinalArgs().Execute(t.Context(), nil); err != nil {
		t.Fatal(err)
	}
	if len(seen) != 4 || seen[0] != "planned" || seen[1] != "planned" || seen[2] != "planned" || seen[3] != "planned" {
		t.Fatalf("executed modes=%q despite each prepared view declaring planned", seen)
	}
}

func TestRequiredPreparedCopiesDoNotShareNestedReferences(t *testing.T) {
	type nested struct {
		Value string `json:"value"`
	}
	type args struct {
		Options map[string]string `json:"options,omitempty"`
		Items   []string          `json:"items"`
		Pointer *nested           `json:"pointer"`
	}
	seen := make(chan string, 4)
	tool := Func[args]("record", "record", func(_ context.Context, input args, _ *Container) (any, error) {
		// Preserve a non-nil empty map even though it is omitted in FinalArgs.
		input.Options["changed"] = "yes"
		seen <- input.Items[0] + "/" + input.Pointer.Value
		input.Items[0] = "mutated"
		input.Pointer.Value = "mutated"
		return "ok", nil
	})
	p, _ := tool.PrepareCall(`{"options":{},"items":["planned"],"pointer":{"value":"planned"}}`)
	var copies []PreparedCall
	for range 4 {
		copies = append(copies, p.RequireFinalArgs())
	}
	start := make(chan struct{})
	var wg sync.WaitGroup
	for _, copy := range copies {
		wg.Add(1)
		go func(copy PreparedCall) {
			defer wg.Done()
			<-start
			if _, err := copy.Execute(t.Context(), nil); err != nil {
				t.Errorf("execute: %v", err)
			}
		}(copy)
	}
	close(start)
	wg.Wait()
	close(seen)
	for got := range seen {
		if got != "planned/planned" {
			t.Fatalf("copy received %q", got)
		}
	}
	if p.FinalArgsOutcome() != FinalArgsUnused {
		t.Fatalf("original outcome changed: %s", p.FinalArgsOutcome())
	}
}
