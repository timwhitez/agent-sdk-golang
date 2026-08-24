package sandbox

import (
	"bytes"
	"encoding/json"
	"fmt"
)

// UnmarshalJSON preserves strict argument decoding while rejecting model-
// controlled timeout values before confirmation, DNS resolution, or dialing.
// The handler still resolves omitted and non-positive values to its existing
// 30-second default; this method only establishes the safe numeric boundary.
func (a *webfetchArgs) UnmarshalJSON(data []byte) error {
	type plainWebfetchArgs webfetchArgs
	var decoded plainWebfetchArgs
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&decoded); err != nil {
		return err
	}
	if _, _, err := checkedSandboxTimeout(decoded.Timeout, 30); err != nil {
		return fmt.Errorf("invalid webfetch timeout: %w; use a timeout from 1 to %d seconds", err, maxSandboxTimeoutSeconds)
	}
	*a = webfetchArgs(decoded)
	return nil
}
