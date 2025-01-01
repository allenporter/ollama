package server

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/ollama/ollama/api"
)

func unmarshalFile(t *testing.T, base, name string, v any) {
	t.Helper()

	bts, err := os.ReadFile(filepath.Join(base, name))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(bts, v); err != nil {
		t.Fatal(err)
	}
}

// normalizeMap modifies values of type []interface{} containing strings to
// []string to workaround json.Unmarshal conversion, which hurts comparisons.
func normalizeMap(m *map[string]interface{}) {
	for k, v := range *m {
		switch val := v.(type) {
		case []interface{}:
			// Convert []interface{} containing strings to []string
			if len(val) > 0 {
				if _, ok := val[0].(string); ok {
					strSlice := make([]string, len(val))
					for i, item := range val {
						strSlice[i] = item.(string)
					}
					(*m)[k] = strSlice
				} else {
					for i := range val {
						switch subval := val[i].(type) {
						case map[string]interface{}:
							normalizeMap(&subval)
						}
					}
				}
			}
		case map[string]interface{}:
			normalizeMap(&val)
		}
	}
}
func TestToolsFormat(t *testing.T) {
	p := filepath.Join("testdata", "format")

	tests := []struct {
		inputFile string
		wantFile  string
	}{
		{
			inputFile: "get-current-weather.json",
			wantFile:  "get-current-weather.schema.json",
		},
		{
			inputFile: "git.json",
			wantFile:  "git.schema.json",
		},
	}

	for _, tc := range tests {
		t.Run(tc.inputFile, func(t *testing.T) {
			var tools []api.Tool
			unmarshalFile(t, p, tc.inputFile, &tools)

			gotMap := toolsFormat(tools)

			var wantMap map[string]interface{}
			unmarshalFile(t, p, tc.wantFile, &wantMap)
			normalizeMap(&wantMap)
			if diff := cmp.Diff(gotMap, wantMap); diff != "" {
				t.Errorf("mismatch (-got +want):\n%s", diff)
			}
		})
	}
}
