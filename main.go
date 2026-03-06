package main

import (
	"bytes"
	"crypto/rand"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

type ExtRequest struct {
	Action string   `json:"action"`
	Input  string   `json:"input"`
	ChatID string   `json:"chat_id"`
	UserID string   `json:"user_id"`
	Images []string `json:"images,omitempty"`
}

type ExtResponse struct {
	Output string         `json:"output"`
	Error  string         `json:"error"`
	Data   map[string]any `json:"data"`
}

type InputParams struct {
	Prompt string `json:"prompt"`
	Model  string `json:"model"`
}

type OpenRouterRequest struct {
	Model      string            `json:"model"`
	Messages   []OpenRouterMsg   `json:"messages"`
	Modalities []string          `json:"modalities"`
}

type OpenRouterMsg struct {
	Role    string `json:"role"`
	Content any    `json:"content"`
}

type ContentPart struct {
	Type     string    `json:"type"`
	Text     string    `json:"text,omitempty"`
	ImageURL *ImageURL `json:"image_url,omitempty"`
}

type ImageURL struct {
	URL string `json:"url"`
}

type OpenRouterResponse struct {
	Choices []struct {
		Message struct {
			Content string            `json:"content"`
			Images  []OpenRouterImage `json:"images"`
		} `json:"message"`
	} `json:"choices"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error"`
}

type OpenRouterImage struct {
	Type     string `json:"type"`
	ImageURL struct {
		URL string `json:"url"`
	} `json:"image_url"`
}

const defaultModel = "google/gemini-3.1-flash-image-preview"

func main() {
	data, err := io.ReadAll(os.Stdin)
	if err != nil {
		writeError("failed to read stdin: " + err.Error())
		return
	}

	var req ExtRequest
	if err := json.Unmarshal(data, &req); err != nil {
		writeError("invalid request: " + err.Error())
		return
	}

	var params InputParams
	if err := json.Unmarshal([]byte(req.Input), &params); err != nil {
		writeError("invalid input params: " + err.Error())
		return
	}

	if params.Prompt == "" {
		writeError("prompt is required")
		return
	}

	apiKey := os.Getenv("TORII_OPENROUTER_API_KEY")
	if apiKey == "" {
		writeError("TORII_OPENROUTER_API_KEY not set")
		return
	}

	model := params.Model
	if model == "" {
		model = os.Getenv("TORII_IMAGE_MODEL")
	}
	if model == "" {
		model = defaultModel
	}

	// Clean up old images before generating new one
	outputDir := getOutputDir()
	cleanupOldImages(outputDir)

	// Call OpenRouter
	imageData, err := generateImage(apiKey, model, params.Prompt, req.Images)
	if err != nil {
		writeError("image generation failed: " + err.Error())
		return
	}

	// Save image
	imagePath, err := saveImage(outputDir, imageData)
	if err != nil {
		writeError("failed to save image: " + err.Error())
		return
	}

	writeResponse(ExtResponse{
		Output: "Image generated",
		Data: map[string]any{
			"image_path": imagePath,
		},
	})
}

func generateImage(apiKey, model, prompt string, images []string) ([]byte, error) {
	var userContent any
	if len(images) > 0 {
		parts := []ContentPart{
			{Type: "text", Text: prompt},
		}
		for _, b64 := range images {
			parts = append(parts, ContentPart{
				Type:     "image_url",
				ImageURL: &ImageURL{URL: "data:image/jpeg;base64," + b64},
			})
		}
		userContent = parts
	} else {
		userContent = prompt
	}

	reqBody := OpenRouterRequest{
		Model: model,
		Messages: []OpenRouterMsg{
			{Role: "user", Content: userContent},
		},
		Modalities: []string{"text", "image"},
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	httpReq, err := http.NewRequest("POST", "https://openrouter.ai/api/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+apiKey)

	client := &http.Client{Timeout: 120 * time.Second}
	resp, err := client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("http request: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("API returned %d: %s", resp.StatusCode, string(respBody))
	}

	var orResp OpenRouterResponse
	if err := json.Unmarshal(respBody, &orResp); err != nil {
		return nil, fmt.Errorf("parse response: %w", err)
	}

	if orResp.Error != nil {
		return nil, fmt.Errorf("API error: %s", orResp.Error.Message)
	}

	if len(orResp.Choices) == 0 {
		return nil, fmt.Errorf("no choices in response")
	}

	msg := orResp.Choices[0].Message

	// Try images array first (newer models like Gemini 3.1)
	if len(msg.Images) > 0 {
		imageData, err := extractImageData(msg.Images[0].ImageURL.URL)
		if err == nil {
			return imageData, nil
		}
	}

	// Fallback: extract from content string (older models)
	if msg.Content != "" {
		imageData, err := extractImageData(msg.Content)
		if err != nil {
			return nil, fmt.Errorf("extract image: %w", err)
		}
		return imageData, nil
	}

	return nil, fmt.Errorf("no image data in response (content empty, images array length: %d)", len(msg.Images))
}

func extractImageData(content string) ([]byte, error) {
	// Look for data URI pattern: data:image/...;base64,...
	prefix := "data:image/"
	idx := strings.Index(content, prefix)
	if idx == -1 {
		// Try to find raw base64 in multipart content
		// Some models return the image as a content part with inline_data
		return nil, fmt.Errorf("no image data found in response (content length: %d)", len(content))
	}

	dataURI := content[idx:]
	// Find the end of the data URI (it might be followed by ) or " or whitespace)
	endChars := []string{")", "\"", "'", "\n", " ", "]"}
	endIdx := len(dataURI)
	for _, ch := range endChars {
		if i := strings.Index(dataURI, ch); i != -1 && i < endIdx {
			endIdx = i
		}
	}
	dataURI = dataURI[:endIdx]

	// Extract base64 part after ";base64,"
	b64Idx := strings.Index(dataURI, ";base64,")
	if b64Idx == -1 {
		return nil, fmt.Errorf("no base64 encoding found in data URI")
	}
	b64Data := dataURI[b64Idx+len(";base64,"):]

	decoded, err := base64.StdEncoding.DecodeString(b64Data)
	if err != nil {
		// Try with padding fixes
		decoded, err = base64.RawStdEncoding.DecodeString(strings.TrimRight(b64Data, "="))
		if err != nil {
			return nil, fmt.Errorf("base64 decode: %w", err)
		}
	}

	return decoded, nil
}

func getOutputDir() string {
	// Use the directory where the executable is located
	execPath, err := os.Executable()
	if err != nil {
		return "./output"
	}
	return filepath.Join(filepath.Dir(execPath), "output")
}

func saveImage(outputDir string, data []byte) (string, error) {
	if err := os.MkdirAll(outputDir, 0o755); err != nil {
		return "", fmt.Errorf("create output dir: %w", err)
	}

	// Generate safe filename: timestamp + random suffix
	randBytes := make([]byte, 4)
	rand.Read(randBytes)
	filename := fmt.Sprintf("img-%d-%x.png", time.Now().UnixMilli(), randBytes)
	path := filepath.Join(outputDir, filename)

	if err := os.WriteFile(path, data, 0o644); err != nil {
		return "", fmt.Errorf("write file: %w", err)
	}

	absPath, err := filepath.Abs(path)
	if err != nil {
		return path, nil
	}
	return absPath, nil
}

func cleanupOldImages(outputDir string) {
	entries, err := os.ReadDir(outputDir)
	if err != nil {
		return
	}

	cutoff := time.Now().Add(-1 * time.Hour)
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		info, err := entry.Info()
		if err != nil {
			continue
		}
		if info.ModTime().Before(cutoff) {
			os.Remove(filepath.Join(outputDir, entry.Name()))
		}
	}
}

func writeResponse(resp ExtResponse) {
	json.NewEncoder(os.Stdout).Encode(resp)
}

func writeError(msg string) {
	writeResponse(ExtResponse{Error: msg})
}
