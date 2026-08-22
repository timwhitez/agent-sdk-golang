package agent

import (
	"context"
	"sync"
)

// beginSteeringInterruptibleStage creates a child context for one provider or
// tool stage. Canceling this child lets a host stop only the active stage so a
// queued steering message can be applied without canceling the whole query.
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

	a.activeStageMu.Lock()
	a.activeStageGeneration++
	if a.activeStageGeneration == 0 {
		a.activeStageGeneration++
	}
	generation := a.activeStageGeneration
	a.activeStageCancel = cancel
	a.activeStageSteering = false
	a.activeStageMu.Unlock()

	var once sync.Once
	interruptedForSteering := false
	return ctx, func() bool {
		once.Do(func() {
			a.activeStageMu.Lock()
			if a.activeStageGeneration == generation {
				interruptedForSteering = a.activeStageSteering
				a.activeStageCancel = nil
				a.activeStageSteering = false
			}
			a.activeStageMu.Unlock()
			cancel()
		})
		return interruptedForSteering
	}
}

// InterruptActiveStageForSteering stops the current provider or tool stage
// while leaving the root query context alive. Callers should first enqueue a
// non-empty SteeringMsg, or retain host-side knowledge that a recently queued
// message was applied but its acknowledgement event is still pending.
func (a *Agent) InterruptActiveStageForSteering() bool {
	if a == nil {
		return false
	}
	a.activeStageMu.Lock()
	cancel := a.activeStageCancel
	if cancel == nil {
		a.activeStageMu.Unlock()
		return false
	}
	a.activeStageSteering = true
	cancel()
	a.activeStageMu.Unlock()
	return true
}
