package agent

import "errors"

// ErrAgentBusy reports a conflicting query or public manual compaction.
// Query-owned automatic compaction stays within its existing private lifecycle.
var ErrAgentBusy = errors.New("agent: another query or manual compaction is already in progress")
