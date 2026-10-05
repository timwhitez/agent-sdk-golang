package agent

import (
	"context"
	"errors"

	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

// Inspect leaves, not errors.Is/As on a joined tree: a cancellation or done
// branch must not conceal a different branch's independent failure. The budget
// treats malformed/cyclic unwrap trees conservatively as failure evidence.
func independentHandlerError(err error) bool {
	budget := 256
	var visit func(error) bool
	visit = func(err error) bool {
		if err == nil {
			return false
		}
		budget--
		if budget < 0 {
			return true
		}
		if joined, ok := err.(interface{ Unwrap() []error }); ok {
			hasChild := false
			for _, child := range joined.Unwrap() {
				hasChild = hasChild || child != nil
				if visit(child) {
					return true
				}
			}
			if hasChild {
				return false
			}
		}
		if wrapped, ok := err.(interface{ Unwrap() error }); ok {
			if child := wrapped.Unwrap(); child != nil {
				return visit(child)
			}
		}
		var done *tools.TaskCompleteError
		return !errors.Is(err, context.Canceled) && !errors.Is(err, context.DeadlineExceeded) && !errors.As(err, &done)
	}
	return visit(err)
}
