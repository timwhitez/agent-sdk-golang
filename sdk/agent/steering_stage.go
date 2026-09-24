package agent

import (
	"context"
	"sync"
)

// activeStage is one interruptible provider or tool stage.
type activeStage struct {
	cancel   context.CancelFunc
	steering bool // interrupted for steering
}

// beginSteeringInterruptibleStage creates a child context for one provider or
// tool stage. Canceling this child lets a host stop only the active stages so
// a queued steering message can be applied without canceling the whole
// query. Stages that run concurrently (tool calls of one bounded wave) are
// all interruptible; the returned finish reports whether this stage was
// interrupted for steering and removes it.
func (a *Agent) beginSteeringInterruptibleStage(parent context.Context) (context.Context, func() bool) {
	if parent == nil {
		parent = context.Background()
	}
	ctx, cancel := context.WithCancel(parent)
	if a == nil {
		return ctx, func() bool {
			cancel()
			return false
		}
	}

	stage := &activeStage{cancel: cancel}
	a.activeStageMu.Lock()
	a.activeStageGeneration++
	generation := a.activeStageGeneration
	if a.activeStages == nil {
		a.activeStages = map[uint64]*activeStage{}
	}
	a.activeStages[generation] = stage
	a.activeStageMu.Unlock()

	var once sync.Once
	interruptedForSteering := false
	return ctx, func() bool {
		once.Do(func() {
			a.activeStageMu.Lock()
			interruptedForSteering = stage.steering
			delete(a.activeStages, generation)
			a.activeStageMu.Unlock()
			cancel()
		})
		return interruptedForSteering
	}
}

// InterruptActiveStageForSteering stops every active provider or tool stage
// while leaving the root query context alive. Callers should first enqueue a
// non-empty SteeringMsg, or retain host-side knowledge that a recently queued
// message was applied but its acknowledgement event is still pending.
func (a *Agent) InterruptActiveStageForSteering() bool {
	if a == nil {
		return false
	}
	a.activeStageMu.Lock()
	defer a.activeStageMu.Unlock()
	for _, stage := range a.activeStages {
		stage.steering = true
		stage.cancel()
	}
	return len(a.activeStages) > 0
}
