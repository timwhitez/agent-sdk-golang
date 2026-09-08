package agent

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

type invocationWarningModel struct {
	entered atomic.Int32
	setters atomic.Int32
	missing atomic.Int32
	ready   chan struct{}
}

func (*invocationWarningModel) Provider() string                   { return "fixture" }
func (*invocationWarningModel) Model() string                      { return "fixture" }
func (m *invocationWarningModel) SetWarningf(func(string, ...any)) { m.setters.Add(1) }
func (m *invocationWarningModel) warn(ctx context.Context, request llm.InvokeRequest) error {
	if m.entered.Add(1) == 8 {
		close(m.ready)
	}
	select {
	case <-m.ready:
	case <-ctx.Done():
		return ctx.Err()
	}
	sink := llm.WarningSink(ctx, nil)
	if sink == nil {
		m.missing.Add(1)
	} else {
		sink("%s", request.Messages[len(request.Messages)-1].Content.PlainText())
	}
	return nil
}
func (m *invocationWarningModel) Invoke(ctx context.Context, request llm.InvokeRequest) (*llm.Completion, error) {
	if err := m.warn(ctx, request); err != nil {
		return nil, err
	}
	return &llm.Completion{Content: llm.TextContent("ok"), StopReason: "stop"}, nil
}

type invocationWarningStreamModel struct{ *invocationWarningModel }

func (m *invocationWarningStreamModel) InvokeStream(ctx context.Context, request llm.InvokeRequest) (<-chan llm.StreamEvent, error) {
	out := make(chan llm.StreamEvent, 2)
	go func() {
		defer close(out)
		if err := m.warn(ctx, request); err != nil {
			out <- llm.StreamErrorEvent{Err: err}
			return
		}
		out <- llm.StreamTextDeltaEvent{Delta: "ok"}
		out <- llm.StreamDoneEvent{StopReason: "stop"}
	}()
	return out, nil
}

func TestSharedModelWarningsBelongToInvokingAgent(t *testing.T) {
	temporary := t.TempDir()
	t.Setenv("TMPDIR", temporary)
	t.Setenv("TMP", temporary)
	t.Setenv("TEMP", temporary)
	for _, stream := range []bool{false, true} {
		t.Run(fmt.Sprint(stream), func(t *testing.T) {
			base := &invocationWarningModel{ready: make(chan struct{})}
			var model llm.ChatModel = base
			if stream {
				model = &invocationWarningStreamModel{base}
			}
			var counts [8]atomic.Int32
			var workers sync.WaitGroup
			ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
			defer cancel()
			for i := range counts {
				workers.Add(1)
				go func(index int) {
					defer workers.Done()
					label := fmt.Sprintf("agent-%d", index)
					a, err := New(Config{LLM: model, MaxIterations: 1, Warningf: func(format string, args ...any) {
						if got := fmt.Sprintf(format, args...); got != label {
							t.Errorf("sink %s received %s", label, got)
						}
						counts[index].Add(1)
					}})
					if err != nil {
						t.Error(err)
						return
					}
					if _, err := a.Query(ctx, label); err != nil {
						t.Error(err)
					}
				}(i)
			}
			workers.Wait()
			if base.setters.Load() != 0 || base.missing.Load() != 0 {
				t.Fatalf("shared setters=%d missing binding=%d", base.setters.Load(), base.missing.Load())
			}
			for i := range counts {
				if counts[i].Load() != 1 {
					t.Fatalf("agent%d warnings=%d", i, counts[i].Load())
				}
			}
		})
	}
}
