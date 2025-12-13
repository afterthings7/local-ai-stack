# Local AI Stack

A unified local AI infrastructure for an M2 MacBook Air (24GB RAM), optimized for Apple Silicon with MPS acceleration.

## Components

### ComfyUI
Node-based UI for Stable Diffusion image generation.

- **Location**: `./ComfyUI/`
- **Launch**: `./ComfyUI/launch.sh`
- **Web UI**: http://127.0.0.1:8188
- **PyTorch**: 2.9.1 with MPS (Metal Performance Shaders) support

### Ollama
Local LLM inference server.

- **Version**: 0.13.3
- **Service**: `brew services start/stop ollama`
- **API**: http://127.0.0.1:11434
- **Models**: Stored in `~/.ollama/models/`
- **Installed**: llama3.2:1b (test model)

## Directory Structure

```
local-ai-stack/
├── ComfyUI/           # Stable Diffusion UI
├── docs/              # Documentation
├── models/            # Symlinks to model locations
├── scripts/           # Utility scripts
└── workflows/         # ComfyUI workflow templates
```

## Quick Start

### ComfyUI
```bash
cd ~/local-ai-stack/ComfyUI
./launch.sh
# Open http://127.0.0.1:8188
```

### Ollama
```bash
# Start service (auto-starts on login)
brew services start ollama

# Run a model
ollama run llama3.2:1b "Your prompt here"

# Pull more models
ollama pull llama3.2      # 3B model
ollama pull codellama     # Code-focused
ollama pull mistral       # Good all-rounder
```

## System Requirements

- macOS with Apple Silicon (M2)
- 24GB unified memory
- Python 3.11+

## Notes

- Model files (.safetensors, .ckpt, .pth) are excluded from git
- ComfyUI uses MPS for GPU acceleration on Apple Silicon
- Output directories are excluded from version control
