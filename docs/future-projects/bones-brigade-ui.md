# Bones Brigade Control Deck - Phase 2 UI Project

## Overview
Custom web UI integrating Ollama and ComfyUI with retro skateboard aesthetic (Powell Peralta / Bones Brigade theme).

## Status
📋 **Planned** - Spec saved for future development

## Why Separate Project
- `local-ai-stack` = backend infrastructure (this repo)
- `bones-brigade-control-deck` = frontend UI (future repo)
- Clean separation of concerns
- Two portfolio pieces instead of one
- Backend must be stable before building custom UI

## Architecture Preview
```
┌─────────────────────────────────────────────────────────────┐
│              Bones Brigade Control Deck                     │
│              (Next.js + TypeScript + Tailwind)              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌────────────────────┐    ┌──────────────────────┐       │
│  │   Chat Interface   │    │  Image Compositor    │       │
│  │   (Ollama)         │    │  (ComfyUI)           │       │
│  │                    │    │                      │       │
│  │ - Streaming tokens │    │ - Workflow selector  │       │
│  │ - System prompts   │    │ - Live progress      │       │
│  │ - Markdown render  │    │ - Output gallery     │       │
│  └────────────────────┘    └──────────────────────┘       │
│           │                          │                     │
│           ├──────────────────────────┤                     │
│           │                          │                     │
│  ┌────────▼──────────────────────────▼────────┐           │
│  │     Unified API Layer (Next.js)            │           │
│  │  - WebSocket manager                       │           │
│  │  - State persistence                       │           │
│  └────────┬──────────────────────────┬────────┘           │
│           │                          │                     │
└───────────┼──────────────────────────┼─────────────────────┘
            │                          │
    ┌───────▼─────────┐        ┌──────▼──────────┐
    │  Ollama API     │        │  ComfyUI API    │
    │  :11434         │        │  :8188          │
    └─────────────────┘        └─────────────────┘
```

## Full Project Specification

---

# Bones Brigade Control Deck - Complete Specification

Build a Mac-friendly local "Bones Brigade Control Deck" UI that integrates Ollama (local LLM) and ComfyUI (Stable Diffusion). Target: Mac Silicon. ComfyUI is installed and running; Ollama will be installed and running locally.

## Goals

Create a custom Control Deck app (browser app on localhost) that:

- Chats with Ollama with streaming tokens
- Submits image workflows to ComfyUI
- Shows live progress for ComfyUI jobs
- Provides an image gallery of outputs
- Uses a cohesive Powell Peralta / Bones Brigade visual theme (neon, skull/"ripper" vibe, airbrush gradients, subtle VHS scanlines, bold typography). Keep it tasteful and readable.

Create a ComfyUI in-app extension (Comfy web client JS extension via custom node) that:

- Injects the same Bones Brigade theme CSS into ComfyUI's UI
- Adds a small "Bones Panel" with a few convenience actions:
  - Load a favorite workflow JSON
  - Apply preset values (sampler/steps/CFG/size)
  - Quick "Send to Control Deck gallery" (at minimum: copy last output filename(s) or provide a link panel)
- (Optional but useful) Provide a themed Open WebUI custom.css file that matches the same aesthetic, as a backup chat UI.

## Non-goals

- No cloud services. Everything local.
- Don't overcomplicate auth. Assume localhost and trusted environment.

## Architecture constraints

- Control Deck: Next.js (App Router) + TypeScript + Tailwind preferred.
- Keep code clean, modular, and runnable with pnpm dev (or npm run dev if you must).
- Use minimal dependencies. If you add one, justify it in comments.
- Provide exact instructions for running on macOS.

## Local endpoints (assume defaults, but make configurable via env)

**Ollama base:** `http://localhost:11434`
- Use streaming endpoint: POST `/api/generate` (or `/api/chat` if you choose), handle newline-delimited JSON streaming.

**ComfyUI base:** `http://localhost:8188`
- Submit: POST `/prompt` with workflow JSON (and optionally client_id)
- Progress: WebSocket `ws://localhost:8188/ws`
- Outputs: fetch images by filename/subfolder/type (implement a pragmatic approach; if you need to query history first, do so).

All base URLs must be configurable via `.env.local`:
- `OLLAMA_BASE_URL`
- `COMFY_BASE_URL`
- `COMFY_WS_URL`

## Control Deck: UI requirements

### Create pages/routes:

**`/` Dashboard** with two tabs or split view:
- Left: Chat (Ollama)
- Right: Image job composer + progress + gallery (ComfyUI)

**A top bar with:**
- Model selector (Ollama) populated by calling Ollama "list models" endpoint if available; if not, accept manual entry.
- Theme toggle (Dark default; keep Bones Brigade consistent)
- Connection status indicators for Ollama and ComfyUI

### Chat (Ollama):

- Conversation list (localStorage persistence is fine)
- System prompt editor (collapsed drawer)
- Streaming output with a "typing" cursor effect
- Markdown rendering is optional; if you add it, keep it minimal and safe.

### ComfyUI (image generation):

- Allow selecting a saved workflow JSON (store in `data/workflows/` and also support user import)
- Provide a small parameter form to override common fields (prompt, negative prompt, seed, steps, CFG, width/height)
- Submit job to ComfyUI
- Show queue status + progress with updates from WebSocket
- After completion, pull output images and display in gallery grid with:
  - click to view modal
  - copy prompt/workflow id
  - download link

### Data/persistence:

- Use localStorage for chat sessions and saved workflows metadata
- No database

## ComfyUI extension requirements

Implement as a ComfyUI custom node with a `js/` web extension.

Create a folder under `ComfyUI/custom_nodes/bones_brigade/`

Include `__init__.py` exporting `WEB_DIRECTORY = "./js"` so Comfy loads the frontend assets.

### In `js/bones.js`:

- Inject CSS (either inline or load `bones.css`)
- Add a small floating panel "Bones Panel" with:
  - Button: "Load Favorite Workflow" (loads a local workflow file or injects a workflow JSON into the UI if possible; otherwise just provides a one-click link/instructions)
  - Button group: apply presets (steps/CFG/sampler)
  - Display last output filenames if accessible
- Keep it robust: if Comfy UI internals change, fail gracefully without breaking Comfy.

### Include install instructions:

- Where to copy the custom_nodes folder
- How to restart ComfyUI
- How to verify the extension loaded

## Bones Brigade theme spec

Create a shared theme file(s) used by:
- Control Deck (Tailwind + CSS variables)
- Comfy extension CSS
- Open WebUI custom.css

### Theme characteristics:

- **Base:** near-black background
- **Accents:** neon green, magenta, electric cyan
- **Subtle scanlines overlay** (low opacity)
- **Grain/noise overlay** (low opacity)
- **Headings:** bold condensed / display font look
- **Body:** clean sans or terminal mono
- **Cards:** soft glow edges, but do NOT reduce legibility
- **Optional watermark:** faint skull silhouette behind main panels (very subtle)

Provide the actual CSS (no placeholders). Use system fonts unless you include font files; if you reference Google Fonts, also provide a "no-internet" fallback path.

## Deliverables

A complete repo scaffold for the Control Deck app with:
- README.md with step-by-step setup/run instructions
- `.env.example`
- `package.json` scripts
- Working Ollama streaming chat
- Working ComfyUI submit + WebSocket progress
- Gallery renders at least the returned images from completed prompts

The ComfyUI custom node + JS extension folder ready to drop into `custom_nodes`

`openwebui-custom.css` (optional) matching theme

## Implementation details you must follow

- Use TypeScript.
- Handle streaming safely (no UI freezing).
- Don't hardcode ports; use env with sensible defaults.
- Provide a "connection test" button for both Ollama and Comfy.
- Add clear error UI states when services aren't running.
- Start by outlining the file tree, then generate the code files in full (or in logical chunks). Include exact curl commands I can run to verify endpoints locally before launching the UI.

---

## Implementation Notes

When building this project:

1. **Prerequisites from local-ai-stack:**
   - Ollama running on `:11434`
   - ComfyUI running on `:8188`
   - Models already downloaded and tested

2. **Development approach:**
   - Create as separate repo: `bones-brigade-control-deck`
   - Reference this repo as backend infrastructure
   - Build incrementally and test each component

3. **Testing strategy:**
   - Test Ollama streaming first (simplest)
   - Add ComfyUI submit + progress
   - Build gallery last
   - Theme application throughout

4. **Portfolio considerations:**
   - Document design decisions
   - Include screenshots/demos
   - Explain technical choices
   - Show before/after with theme

## Success Criteria

- [ ] Single localhost page loads successfully
- [ ] Can chat with Ollama models with streaming
- [ ] Can submit ComfyUI job and see progress
- [ ] Gallery displays completed images
- [ ] Theme applied consistently across all components
- [ ] ComfyUI extension loads without breaking ComfyUI
- [ ] All endpoints configurable via environment
- [ ] Clear error states when services unavailable
- [ ] Professional documentation for setup

## Timeline Estimate

- **Week 1:** Next.js scaffold + Ollama integration
- **Week 2:** ComfyUI integration + WebSocket progress
- **Week 3:** Theme implementation + polish
- **Week 4:** ComfyUI extension + documentation

Total: ~4 weeks part-time development

## Related Resources

- [Ollama API Documentation](https://github.com/ollama/ollama/blob/main/docs/api.md)
- [ComfyUI API Examples](https://github.com/comfyanonymous/ComfyUI/tree/master/script_examples)
- [Next.js App Router](https://nextjs.org/docs/app)
- Powell Peralta/Bones Brigade aesthetic references (search for vintage skateboard graphics)

---

*This specification will be used to create a separate repository and portfolio project once the backend infrastructure (local-ai-stack) is stable and tested.*
