//go:build !unix

package agent

import (
	"fmt"
	"os"
)

func validateToolResultDumpOwner(_ os.FileInfo) error {
	return fmt.Errorf("tool result dump ownership verification is unavailable on this platform")
}
