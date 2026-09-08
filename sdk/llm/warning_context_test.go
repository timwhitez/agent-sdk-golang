package llm_test

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

func TestWarningContextScopeAndCancellation(t *testing.T) {
	var fallback, parent, child int
	f := func(string, ...any) { fallback++ }
	root, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	bound := llm.WithWarningSink(root, func(string, ...any) { parent++ })
	nested := llm.WithWarningSink(bound, func(string, ...any) { child++ })
	llm.WarningSink(context.Background(), f)("fixture")
	llm.WarningSink(bound, f)("fixture")
	llm.WarningSink(nested, f)("fixture")
	if fallback != 1 || parent != 1 || child != 1 {
		t.Fatal("warning scope leaked")
	}
	llm.WarningSink(llm.WithWarningSink(bound, nil), f)("fixture")
	if fallback != 2 || parent != 1 {
		t.Fatal("explicit nil did not restore fallback")
	}
	deadline, _ := root.Deadline()
	got, _ := bound.Deadline()
	if !got.Equal(deadline) || bound.Done() != root.Done() {
		t.Fatal("binding changed deadline/cancellation")
	}
	cancel()
	if !errors.Is(nested.Err(), context.Canceled) {
		t.Fatal("nested cancellation lost")
	}
}

func TestProviderWarningContextsDoNotMutateSharedClient(t *testing.T) {
	for _, provider := range []string{"anthropic", "chat", "responses"} {
		for _, stream := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/%v", provider, stream), func(t *testing.T) {
				var fallback atomic.Int32
				model := admissionModel(provider, func(r *http.Request) (*http.Response, error) {
					return &http.Response{StatusCode: 401, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(`{"error":{"message":"fixture"}}`)), Request: r}, nil
				}, func(string, ...any) { fallback.Add(1) })
				request := admissionRequest(t, llm.CacheBestEffort)
				var counts [8]atomic.Int32
				var workers sync.WaitGroup
				for i := range counts {
					workers.Add(1)
					go func(index int) {
						defer workers.Done()
						ctx := llm.WithWarningSink(context.Background(), func(string, ...any) { counts[index].Add(1) })
						_, _ = callAdmissionModel(ctx, model, request, stream)
					}(i)
				}
				workers.Wait()
				for i := range counts {
					if counts[i].Load() != 1 {
						t.Fatalf("sink %d received %d warnings", i, counts[i].Load())
					}
				}
				if fallback.Load() != 0 {
					t.Fatal("invocation used configured global sink")
				}
				_, _ = callAdmissionModel(context.Background(), model, request, stream)
				if fallback.Load() != 1 {
					t.Fatal("invocation mutated direct-client fallback")
				}
			})
		}
	}
}
