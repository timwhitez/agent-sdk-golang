package agent

import "errors"

// ErrAgentBusy reports that a second turn was submitted while this Agent still
// owns an active turn. Agent history, compaction, steering, and cancellation
// state form one turn-scoped state machine and are not shared concurrently.
var ErrAgentBusy = errors.New("agent: another query is already in progress")
