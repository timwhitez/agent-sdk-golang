package agent

import (
	"fmt"
	"strings"
	"testing"
)

func TestShadowAutomaticCompactionDecision(t *testing.T) {
	for _, test := range []struct {
		name        string
		observation automaticCompactionObservation
		want        compactionDecision
	}{
		{
			name:        "below watermark",
			observation: automaticCompactionObservation{trigger: "usage"},
			want:        compactionDecision{trigger: "usage"},
		},
		{
			name:        "ordinary admission",
			observation: automaticCompactionObservation{ordinaryAdmission: true, trigger: "usage", targetWatermark: "snip"},
			want:        compactionDecision{run: true, trigger: "usage", targetWatermark: "snip"},
		},
		{
			name:        "hard overflow bypass",
			observation: automaticCompactionObservation{overflow: true, trigger: "overflow", targetWatermark: "overflow"},
			want:        compactionDecision{run: true, trigger: "overflow", targetWatermark: "overflow"},
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			if got := shadowAutomaticCompactionDecision(test.observation); got != test.want {
				t.Fatalf("decision=%#v want %#v", got, test.want)
			}
		})
	}
}

func TestObserveAutomaticCompactionDecisionKeepsLegacyAuthorityAndSafeWarning(t *testing.T) {
	legacy := compactionDecision{run: true, trigger: "usage", targetWatermark: "snip"}
	for _, test := range []struct {
		name         string
		shadow       compactionDecision
		wantWarnings int
	}{
		{name: "equal", shadow: legacy},
		{name: "run", shadow: compactionDecision{trigger: "usage", targetWatermark: "snip"}, wantWarnings: 1},
		{name: "trigger", shadow: compactionDecision{run: true, trigger: "todo_checkpoint", targetWatermark: "snip"}, wantWarnings: 1},
		{name: "watermark", shadow: compactionDecision{run: true, trigger: "usage", targetWatermark: "prune"}, wantWarnings: 1},
	} {
		t.Run(test.name, func(t *testing.T) {
			warnings := 0
			agent := &Agent{warningf: func(string, ...any) { warnings++ }}
			agent.observeAutomaticCompactionDecision(legacy, test.shadow)
			if got := warnings; got != test.wantWarnings {
				t.Fatalf("warnings=%d want %d", got, test.wantWarnings)
			}
		})
	}

	var warning string
	agent := &Agent{warningf: func(format string, args ...any) { warning = fmt.Sprintf(format, args...) }}
	agent.observeAutomaticCompactionDecision(
		compactionDecision{run: true, trigger: "secret-trigger", targetWatermark: "secret-watermark"},
		compactionDecision{trigger: "usage", targetWatermark: "snip"},
	)
	if !strings.Contains(warning, "legacy_run=true shadow_run=false legacy_trigger=unknown shadow_trigger=usage legacy_watermark=unknown shadow_watermark=snip") || strings.Contains(warning, "secret") {
		t.Fatalf("unsafe mismatch warning: %q", warning)
	}
}

var benchmarkCompactionDecision compactionDecision

func BenchmarkShadowAutomaticCompactionDecision(b *testing.B) {
	observation := automaticCompactionObservation{
		ordinaryAdmission: true,
		trigger:           "todo_checkpoint",
		targetWatermark:   "prune",
	}
	legacy := compactionDecision{run: true, trigger: "todo_checkpoint", targetWatermark: "prune"}
	agent := &Agent{warningf: func(string, ...any) {}}
	b.ReportAllocs()
	var decision compactionDecision
	for i := 0; i < b.N; i++ {
		decision = shadowAutomaticCompactionDecision(observation)
		agent.observeAutomaticCompactionDecision(legacy, decision)
	}
	benchmarkCompactionDecision = decision
}
