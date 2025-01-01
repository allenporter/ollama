package server

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/model/mllama"
	"github.com/ollama/ollama/template"
)

type tokenizeFunc func(context.Context, string) ([]int, error)

var errTooManyImages = errors.New("vision model only supports a single image per message")

// chatPrompt accepts a list of messages and returns the prompt and images that should be used for the next chat turn.
// chatPrompt truncates any messages that exceed the context window of the model, making sure to always include 1) the
// latest message and 2) system messages
func chatPrompt(ctx context.Context, m *Model, tokenize tokenizeFunc, opts *api.Options, msgs []api.Message, tools []api.Tool) (prompt string, images []llm.ImageData, _ error) {
	var system []api.Message

	isMllama := checkMllamaModelFamily(m)

	var imageNumTokens int
	// TODO: Ideally we would compute this from the projector metadata but some pieces are implementation dependent
	if isMllama {
		// Our mllama implementation packs all of the embeddings into a single token
		imageNumTokens = 1
	} else {
		// Clip images are represented as 768 tokens, each an embedding
		imageNumTokens = 768
	}

	n := len(msgs) - 1
	// in reverse, find all messages that fit into context window
	for i := n; i >= 0; i-- {
		if isMllama && len(msgs[i].Images) > 1 {
			return "", nil, errTooManyImages
		}

		// always include the last message
		if i == n {
			continue
		}

		system = make([]api.Message, 0)
		for j := range i {
			if msgs[j].Role == "system" {
				system = append(system, msgs[j])
			}
		}

		var b bytes.Buffer
		if err := m.Template.Execute(&b, template.Values{Messages: append(system, msgs[i:]...), Tools: tools}); err != nil {
			return "", nil, err
		}

		s, err := tokenize(ctx, b.String())
		if err != nil {
			return "", nil, err
		}

		ctxLen := len(s)
		if m.ProjectorPaths != nil {
			for _, m := range msgs[i:] {
				ctxLen += imageNumTokens * len(m.Images)
			}
		}

		if ctxLen > opts.NumCtx {
			slog.Debug("truncating input messages which exceed context length", "truncated", len(msgs[i:]))
			break
		} else {
			n = i
		}
	}

	currMsgIdx := n

	for cnt, msg := range msgs[currMsgIdx:] {
		prefix := ""
		imgPrompt := ""
		prompt := msg.Content

		for _, i := range msg.Images {
			var imgData llm.ImageData

			if isMllama {
				data, opts, err := mllama.Preprocess(bytes.NewReader(i))
				if err != nil {
					return "", nil, err
				}

				buf := new(bytes.Buffer)
				err = binary.Write(buf, binary.LittleEndian, data)
				if err != nil {
					return "", nil, err
				}

				ar, ok := opts["aspectRatioIndex"].(int)
				if !ok {
					return "", nil, fmt.Errorf("missing aspect ratio for image")
				}

				imgData = llm.ImageData{
					ID:            len(images),
					Data:          buf.Bytes(),
					AspectRatioID: ar,
				}
				imgPrompt = "<|image|>"
			} else {
				imgData = llm.ImageData{
					ID:   len(images),
					Data: i,
				}
			}

			imgTag := fmt.Sprintf("[img-%d]", imgData.ID)
			if !strings.Contains(prompt, "[img]") {
				prefix += imgTag
			} else {
				prompt = strings.Replace(prompt, "[img]", imgTag, 1)
			}

			images = append(images, imgData)
		}
		msgs[currMsgIdx+cnt].Content = prefix + imgPrompt + prompt
	}

	// truncate any messages that do not fit into the context window
	var b bytes.Buffer
	if err := m.Template.Execute(&b, template.Values{Messages: append(system, msgs[currMsgIdx:]...), Tools: tools}); err != nil {
		return "", nil, err
	}

	return b.String(), images, nil
}

func checkMllamaModelFamily(m *Model) bool {
	for _, arch := range m.Config.ModelFamilies {
		if arch == "mllama" {
			return true
		}
	}
	return false
}

// chatFormat looks at the incoming request format and set of tools and determines the
// format of the chat response. Tools are provided in an OpenAPI schema which can
// be used to determine the format of the response to avoid the model producing
// tool calls that are not the right format.
func chatFormat(reqFormat json.RawMessage, tools []api.Tool) (format json.RawMessage) {
	// Use the existing request format if provided
	if len(reqFormat) > 0 {
		return reqFormat
	}
	toolFormats := make([]interface{}, 0, len(tools))
	for _, tool := range tools {
		// Create an object type with a properties ield that has a name
		// with a string and the enum is the name of the tool. The parameters
		// are the parameters of the tool.
		toolFormat := map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"name": map[string]interface{}{
					"type": "string",
					"enum": []string{tool.Function.Name},
				},
				"parameters": tool.Function.Parameters,
			},
			"required": []string{"name", "parameters"},
		}
		toolFormats = append(toolFormats, toolFormat)
	}
	// Return the tool formats as an array of objects
	resultFormat := map[string]interface{}{
		"anyOf": toolFormats,
	}
	format, _ = json.Marshal(resultFormat)
	return format
}
