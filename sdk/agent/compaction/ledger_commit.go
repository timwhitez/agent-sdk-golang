package compaction

import (
	"context"
	"fmt"
	"strings"
)

type deferredLedgerStore struct {
	ledger *Ledger
	saved  bool
}

func (s *deferredLedgerStore) Load(_ context.Context, sessionID string) (*Ledger, error) {
	if s == nil {
		return nil, fmt.Errorf("compaction: deferred ledger store is nil")
	}
	if s.ledger == nil {
		return NewLedger(sessionID), nil
	}
	return s.ledger.Clone(), nil
}

func (s *deferredLedgerStore) Save(_ context.Context, _ string, ledger *Ledger) error {
	if s == nil {
		return fmt.Errorf("compaction: deferred ledger store is nil")
	}
	if ledger == nil {
		return fmt.Errorf("compaction: deferred ledger candidate is nil")
	}
	s.saved = true
	s.ledger = ledger.Clone()
	return nil
}

// CommitPendingLedger persists a summary-ledger update that was deferred until
// the Agent is ready to write the matching replayable runtime checkpoint.
func (s *Service) CommitPendingLedger(ctx context.Context, res *Result) error {
	if res == nil || res.pendingLedger == nil {
		return nil
	}
	if s == nil || s.Config.LedgerStore == nil {
		return fmt.Errorf("compaction: ledger store unavailable for pending checkpoint transaction")
	}
	return s.saveLedger(ctx, strings.TrimSpace(s.Config.SessionID), res.pendingLedger)
}

// RollbackPendingLedger restores the ledger version that preceded a deferred
// summary update. The pending transaction remains attached to the result so a
// retry can commit it again before retrying checkpoint persistence.
func (s *Service) RollbackPendingLedger(ctx context.Context, res *Result) error {
	if res == nil || res.pendingLedger == nil || res.previousLedger == nil {
		return nil
	}
	if s == nil || s.Config.LedgerStore == nil {
		return fmt.Errorf("compaction: ledger store unavailable for checkpoint rollback")
	}
	return s.saveLedger(ctx, strings.TrimSpace(s.Config.SessionID), res.previousLedger)
}

// FinalizePendingLedger drops transient transaction state after both ledger and
// runtime checkpoint persistence have succeeded.
func (s *Service) FinalizePendingLedger(res *Result) {
	if res == nil {
		return
	}
	res.pendingLedger = nil
	res.previousLedger = nil
}
