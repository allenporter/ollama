package server

import (
	"encoding/json"

	"github.com/ollama/ollama/api"
)

type Property struct {
	Type string   `json:"type"`
	Enum []string `json:"enum,omitempty"`
}

// parametersFormat accepts tool function paraemeters as input and returns a
// JSON schema for the parameters that can be used for a grammar for a
// chat completion request.
func parametersFormat(parameters api.ToolParameters) map[string]interface{} {
	properties := make(map[string]interface{})
	for name, param := range parameters.Properties {
		field := map[string]interface{}{
			"type": param.Type,
		}
		if len(param.Enum) > 0 {
			field["enum"] = param.Enum
		}
		properties[name] = field
	}
	result := map[string]interface{}{
		"type":       "object",
		"properties": properties,
	}
	if len(parameters.Required) > 0 {
		result["required"] = parameters.Required
	}
	return result
}

// toolFormat determines the tool completion format based on the tool provided,
// converting a tool definintion into a JSON schema.
func toolFormat(tool api.Tool) map[string]interface{} {
	return map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"name": map[string]interface{}{
				"type": "string",
				"enum": []string{tool.Function.Name},
			},
			"parameters": parametersFormat(tool.Function.Parameters),
		},
		"required": []string{"name", "parameters"},
	}
}

// toolsFormat determines the tools completion format based on the tools provided,
// converting a list of tool definitions into a JSON schema.
func toolsFormat(tools []api.Tool) map[string]interface{} {
	if len(tools) == 1 {
		return toolFormat(tools[0])
	}
	toolFormats := make([]interface{}, 0, len(tools))
	for _, tool := range tools {
		toolFormats = append(toolFormats, toolFormat(tool))
	}
	return map[string]interface{}{
		"anyOf": toolFormats,
	}
}

// chatFormat determins the chat completion format based on the incoming request
// format and tools rpovided. Tools are provided in an OpenAPI schema, which
// can be used as the models output format.
func chatFormat(reqFormat json.RawMessage, tools []api.Tool) (format json.RawMessage) {
	// Use the existing request format if provided
	if len(reqFormat) > 0 || len(tools) == 0 {
		return reqFormat
	}
	format, _ = json.Marshal(toolsFormat(tools))
	return format
}
