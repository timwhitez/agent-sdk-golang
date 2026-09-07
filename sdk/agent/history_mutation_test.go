package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/agent/compaction"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

func TestContinuationSystemPrefixShiftKeepsCurrentAssistant(t *testing.T) {
	for _, mode := range []string{"insert", "remove"} {
		t.Run(mode, func(t *testing.T) {
			var ag *Agent
			providers, handled, repairs := 0, 0, 0
			shifted := false
			model := &frameScriptModel{invoke: func(llm.InvokeRequest) (*llm.Completion, error) {
				providers++
				if providers == 1 {
					return &llm.Completion{Content: llm.TextContent("partial"), StopReason: "max_tokens", ToolCalls: []llm.ToolCall{{ID: "a", Function: llm.FunctionCall{Name: "work", Arguments: `{"value":`}}}}, nil
				}
				if providers == 2 {
					return &llm.Completion{Content: llm.TextContent("final fragment"), ToolCalls: []llm.ToolCall{{ID: "a", Function: llm.FunctionCall{Name: "work", Arguments: `"ok"}`}}}}, nil
				}
				return &llm.Completion{Content: llm.TextContent("done")}, nil
			}}
			var err error
			ag, err = New(Config{LLM: model, SystemPrompt: "base", Tools: []tools.Tool{{Name: "work", Handler: func(_ context.Context, args json.RawMessage, _ *tools.Container) (llm.Content, error) {
				handled++
				if !strings.Contains(string(args), "ok") {
					t.Error("lost merged args")
				}
				return llm.TextContent("result"), nil
			}}}, Compaction: &compaction.Config{Enabled: true, ContextWindow: 100000, TokenEstimator: func(string) int {
				if ag != nil && providers == 2 && !shifted {
					messages := ag.Messages()
					if len(messages) > 0 && messages[len(messages)-1].Content.PlainText() == "final fragment" {
						shifted = true
						beforeLength := len(messages)
						if mode == "insert" {
							messages = append([]llm.Message{llm.NewSystemMessage("added")}, messages...)
						} else {
							messages = messages[1:]
						}
						if updateErr := ag.ReplaceHistoryChecked(messages); updateErr != nil {
							t.Error(updateErr)
						}
						actual := ag.Messages()
						wantLength := beforeLength + 1
						if mode == "remove" {
							wantLength = beforeLength - 1
						}
						if len(actual) != wantLength || (mode == "insert" && actual[0].Content.PlainText() != "added") || (mode == "remove" && actual[0].Role == llm.RoleSystem) {
							t.Error("system prefix shift was not applied")
						}
					}
				}
				return 1
			}}, Warningf: func(string, ...any) {}})
			if err != nil {
				t.Fatal(err)
			}
			for e := range ag.QueryStream(context.Background(), llm.TextContent("run")) {
				if w, ok := e.(WarnEvent); ok && w.Kind == "tool_pairing_repaired" {
					repairs++
				}
			}
			if !shifted || providers != 3 || handled != 1 {
				t.Fatalf("shift/provider/handler=%v/%d/%d", shifted, providers, handled)
			}
			if _, changed, _ := repairToolCallPairsDetailed(ag.Messages()); changed || repairs != 0 {
				t.Fatal("system prefix shift corrupted continuation topology")
			}
			calls := 0
			for _, m := range ag.Messages() {
				if len(m.ToolCalls) > 0 {
					calls++
					if m.Content.PlainText() != "final fragment" || m.ToolCalls[0].Function.Arguments != `{"value":"ok"}` {
						t.Fatal("wrong assistant anchor updated")
					}
				}
			}
			if calls != 1 {
				t.Fatalf("assistant blocks=%d", calls)
			}
		})
	}
}

func TestCheckedActiveHistoryMutationContract(t *testing.T) {
	for _, stage := range []string{"provider_pending", "handler_active"} {
		for _, mutation := range []string{"clear_checked", "replace_checked", "clear_legacy", "replace_legacy", "system_prefix", "system_tail", "prior_tool_gap"} {
			t.Run(stage+"/"+mutation, func(t *testing.T) {
				var ag *Agent
				var warnings atomic.Int32
				var applyMutation func()
				entered, release := make(chan struct{}), make(chan struct{})
				providers, handled := 0, 0
				model := &frameScriptModel{invoke: func(req llm.InvokeRequest) (*llm.Completion, error) {
					providers++
					if providers == 1 {
						if stage == "provider_pending" {
							close(entered)
							<-release
						}
						if req.Messages[0].Content.PlainText() != "base" {
							t.Error("in-flight request was mutated")
						}
						return &llm.Completion{ToolCalls: []llm.ToolCall{{ID: "reused", Function: llm.FunctionCall{Name: "work", Arguments: "{}"}}}}, nil
					}
					if mutation == "system_prefix" || (mutation == "system_tail" && stage == "provider_pending") {
						updated := false
						for _, message := range req.Messages {
							updated = updated || (message.Role == llm.RoleSystem && message.Content.PlainText() == "updated")
						}
						if !updated {
							t.Error("next logical request did not receive successful system update")
						}
					}
					return &llm.Completion{Content: llm.TextContent("done")}, nil
				}}
				var err error
				ag, err = New(Config{LLM: model, Tools: []tools.Tool{{Name: "work", Handler: func(context.Context, json.RawMessage, *tools.Container) (llm.Content, error) {
					handled++
					if stage == "handler_active" {
						applyMutation()
					}
					return llm.TextContent("observed result"), nil
				}}}, Warningf: func(format string, args ...any) {
					warnings.Add(1)
					if strings.Contains(format, "private replacement") {
						t.Error("mutation contents leaked")
					}
					if ag != nil {
						_ = ag.Messages()
					} // reject warning must be outside a.mu
				}})
				if err != nil {
					t.Fatal(err)
				}
				seed := []llm.Message{llm.NewSystemMessage("base"), llm.NewUserMessage("prior"), llm.NewAssistantMessage("prior", []llm.ToolCall{{ID: "reused", Function: llm.FunctionCall{Name: "work", Arguments: "{}"}}}), llm.NewToolMessage("reused", "work", llm.TextContent("prior result"), false)}
				ag.ReplaceHistory(seed)
				wantRejected := mutation != "system_prefix" && !(mutation == "system_tail" && stage == "provider_pending")
				applyMutation = func() {
					before := ag.Messages()
					var mutationErr error
					switch mutation {
					case "clear_checked":
						mutationErr = ag.ClearHistoryChecked()
					case "replace_checked":
						mutationErr = ag.ReplaceHistoryChecked([]llm.Message{llm.NewUserMessage("private replacement")})
					case "clear_legacy":
						ag.ClearHistory()
					case "replace_legacy":
						ag.ReplaceHistory([]llm.Message{llm.NewUserMessage("private replacement")})
					case "system_prefix":
						mutationErr = ag.ReplaceHistoryChecked(append([]llm.Message{llm.NewSystemMessage("updated")}, before...))
					case "system_tail":
						mutationErr = ag.ReplaceHistoryChecked(append(before, llm.NewSystemMessage("updated")))
					case "prior_tool_gap":
						candidate := append([]llm.Message(nil), before[:3]...)
						candidate = append(candidate, llm.NewSystemMessage("updated"))
						candidate = append(candidate, before[3:]...)
						mutationErr = ag.ReplaceHistoryChecked(candidate)
					}
					legacy := strings.HasSuffix(mutation, "legacy")
					if !legacy && errors.Is(mutationErr, ErrActiveHistoryMutation) != wantRejected {
						t.Errorf("rejected=%v want %v", mutationErr, wantRejected)
					}
					if wantRejected && !messageJSONEqual(before, ag.Messages()) {
						t.Error("rejected mutation changed history")
					}
					if !wantRejected && mutationErr != nil {
						t.Error(mutationErr)
					}
					if !wantRejected && messageJSONEqual(before, ag.Messages()) {
						t.Error("successful update silently ignored")
					}
				}
				stream := ag.QueryStream(context.Background(), llm.TextContent("run"))
				if stage == "provider_pending" {
					select {
					case <-entered:
						applyMutation()
						close(release)
					case <-time.After(5 * time.Second):
						close(release)
						t.Fatal("provider did not enter")
					}
				}
				repairs, failures, results := 0, 0, 0
				for event := range stream {
					switch e := event.(type) {
					case WarnEvent:
						if e.Kind == "tool_pairing_repaired" {
							repairs++
						}
					case ErrorEvent:
						failures++
					case ToolResultEvent:
						results++
					}
				}
				if providers != 2 || handled != 1 || repairs != 0 || failures != 0 || results != 1 {
					t.Fatalf("provider/handler/repair/errors/results=%d/%d/%d/%d/%d", providers, handled, repairs, failures, results)
				}
				wantWarnings := int32(0)
				if strings.HasSuffix(mutation, "legacy") {
					wantWarnings = 1
				}
				if warnings.Load() != wantWarnings {
					t.Fatalf("warnings=%d want %d", warnings.Load(), wantWarnings)
				}
				if _, changed, _ := repairToolCallPairsDetailed(ag.Messages()); changed {
					t.Fatal("stored history needs repair")
				}
				priorResults := 0
				for _, m := range ag.Messages() {
					if m.Role == llm.RoleTool && m.Content.PlainText() == "prior result" {
						priorResults++
					}
				}
				if priorResults != 1 {
					t.Fatal("prior completed block with reused ID changed")
				}
				if err := ag.ClearHistoryChecked(); err != nil || len(ag.Messages()) != 0 {
					t.Fatal("idle clearing changed")
				}
			})
		}
	}
}

func TestActiveHistoryMutationUsesFullMessageJSON(t *testing.T) {
	content, err := llm.WithProviderState(llm.TextContent("visible"), []llm.ProviderState{{Provider: "fixture", Kind: "opaque", Data: json.RawMessage(`{"private":1}`)}})
	if err != nil {
		t.Fatal(err)
	}
	current := []llm.Message{llm.NewSystemMessage("base"), {Role: llm.RoleUser, Content: content, Cache: true}}
	candidate := llm.CloneMessages(current)
	candidate[1].ToolCalls = []llm.ToolCall{} // omitempty-equivalent
	if !activeHistoryReplacementSafe(current, candidate) {
		t.Fatal("nil/empty JSON equivalence rejected")
	}
	for _, mutate := range []func(*llm.Message){func(m *llm.Message) { m.Cache = false }, func(m *llm.Message) { m.Content = llm.TextContent("visible") }, func(m *llm.Message) { m.IsError = true }, func(m *llm.Message) { m.Name = "changed" }} {
		candidate = llm.CloneMessages(current)
		mutate(&candidate[1])
		if activeHistoryReplacementSafe(current, candidate) {
			t.Fatal("non-visible message identity drift accepted")
		}
	}
	for _, mutate := range []func(*llm.Message){func(m *llm.Message) { m.ToolCallID = "fake" }, func(m *llm.Message) { m.ToolCalls = []llm.ToolCall{{ID: "fake"}} }, func(m *llm.Message) { m.Ephemeral = true }, func(m *llm.Message) { m.Destroyed = true }} {
		candidate = llm.CloneMessages(current)
		mutate(&candidate[0])
		if activeHistoryReplacementSafe(current, candidate) {
			t.Fatal("system role smuggled tool-only control fields")
		}
	}
}

func TestRejectedHistoryMutationKeepsTrackingAndDump(t *testing.T) {
	ag, err := New(Config{LLM: historyCloneModel{}})
	if err != nil {
		t.Fatal(err)
	}
	ag.ReplaceHistory([]llm.Message{llm.NewUserMessage("active")})
	path := filepath.Join(t.TempDir(), "owned-result.txt")
	if err := os.WriteFile(path, []byte("fixture"), 0600); err != nil {
		t.Fatal(err)
	}
	ag.toolResultDumps[path] = toolResultDumpLifecycleEntry{ExpiresAt: time.Now().Add(time.Hour)}
	ag.ephemeralScanFrom = 7
	ag.ephemeralByKey = map[string][]int{"key": {1}}
	ag.ephemeralSigByCall = map[string]string{"call": "signature"}
	ag.turnActive.Store(true)
	defer ag.turnActive.Store(false)
	if !errors.Is(ag.ClearHistoryChecked(), ErrActiveHistoryMutation) {
		t.Fatal("clear did not reject")
	}
	if !errors.Is(ag.ReplaceHistoryChecked([]llm.Message{llm.NewUserMessage("replacement")}), ErrActiveHistoryMutation) {
		t.Fatal("replace did not reject")
	}
	if ag.ephemeralScanFrom != 7 || len(ag.ephemeralByKey["key"]) != 1 || ag.ephemeralSigByCall["call"] != "signature" || len(ag.toolResultDumps) != 1 {
		t.Fatal("rejection reset owned tracking")
	}
	if data, err := os.ReadFile(path); err != nil || string(data) != "fixture" {
		t.Fatal("rejection cleaned owned dump")
	}
}

func BenchmarkCheckedSystemHistoryUpdate(b *testing.B) {
	for _, count := range []int{1, 32, 256} {
		b.Run(fmt.Sprintf("messages%d", count), func(b *testing.B) {
			messages := []llm.Message{llm.NewSystemMessage("base")}
			for i := 0; i < count; i++ {
				messages = append(messages, llm.NewUserMessage(strings.Repeat("fixture ", 128)))
			}
			ag, err := New(Config{LLM: historyCloneModel{}, InitialMessages: messages})
			if err != nil {
				b.Fatal(err)
			}
			ag.turnActive.Store(true)
			defer ag.turnActive.Store(false)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if err := ag.ReplaceHistoryChecked(messages); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
