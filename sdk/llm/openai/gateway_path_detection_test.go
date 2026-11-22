package openai

import "testing"

func TestResponsesForceStringInputVersionDetection(t *testing.T) {
	for _, tc := range []struct {
		base  string
		force bool
		want  bool
	}{
		{"https://host/api/vision", false, false},
		{"https://host/api/v2beta", false, false},
		{"https://host/api/v", false, false},
		{"https://host/api/v3", false, true},
		{"https://host/proxy/api/v12/tenant", false, true},
		{"https://host/api/photos", true, true},
	} {
		t.Run(tc.base, func(t *testing.T) {
			client := &ResponsesClient{BaseURL: tc.base, ForceStringInput: tc.force}
			if got := client.forceStringInput(); got != tc.want {
				t.Fatalf("string input=%v want %v", got, tc.want)
			}
		})
	}
}
