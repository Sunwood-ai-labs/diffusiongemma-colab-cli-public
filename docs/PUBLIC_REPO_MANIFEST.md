# Public Repository Manifest

This repository is organized so the Colab experiment, the raw evidence, and the
demo video can be published without private credentials or transient runtime
files.

## Working Folder

Repository root:

```text
diffusiongemma-colab-cli-public/
```

This folder is already a Git repository.

## Public Scripts

- `scripts/run_diffusiongemma_probe.py`
  - Full Transformers probe for `google/diffusiongemma-26B-A4B-it`.
- `scripts/run_diffusiongemma_gguf_probe.py`
  - L4/T4-oriented GGUF smoke test using Unsloth's `Q4_K_M` GGUF route.
- `scripts/run_diffusiongemma_gguf_visual_capture.py`
  - Colab-side PTY capture of the real `llama-diffusion-cli`
    `--diffusion-visual` terminal surface.
- `scripts/render_ansi_capture_video.py`
  - Local renderer that replays the captured ANSI terminal stream into an MP4.
  - Current style uses the Sunwood AI Labs color palette.

## Article Drafts

- `articles/diffusiongemma-l4-speed-check.md`
  - Publication-ready Japanese technical article covering the Colab L4
    DiffusionGemma speed check, the Gemma4 baseline correction, and the limits
    of the current comparison.
- `articles/assets/thumbnails/diffusiongemma-l4-speed-check-thumbnail-1200x630.png`
  - Final 1200x630 article thumbnail generated from the Maki technical article
    thumbnail scaffold.
- `articles/assets/thumbnails/diffusiongemma-l4-speed-check-thumbnail-prompt.txt`
  - Exact Image Gen prompt used for the thumbnail.

## Final Demo Artifacts

- Final MP4:
  `videos/l4-quant-generation-flow/final/diffusiongemma-l4-q4-real-terminal-visual-capture-sunwood.mp4`
- iOS-compatible MP4:
  `videos/l4-quant-generation-flow/final/diffusiongemma-sunwood-demo-ios-compatible.mp4`
- Mobile preview page:
  `videos/l4-quant-generation-flow/final/preview.html`
- Mobile-width screenshot:
  `videos/l4-quant-generation-flow/final/mobile-preview-screenshot.png`
- QA frame at 9s:
  `videos/l4-quant-generation-flow/final/qa-009s.png`
- QA frame at 19s:
  `videos/l4-quant-generation-flow/final/qa-019s.png`

The same final MP4 is also kept at the renderer's canonical output path:

```text
videos/l4-quant-generation-flow/renders/diffusiongemma-l4-q4-real-terminal-visual-capture.mp4
```

## Background Asset

- Public repo copy:
  `assets/backgrounds/sunwood-ai-labs-felt-room-background-maki-banner.png`

The renderer blends this image lightly into the deep green terminal background
so the felt-room scene is visible without reducing text readability.

## Reproduction Inputs

- Raw ANSI terminal capture:
  `results/20260611T080801Z_visual_capture/visual_terminal_capture.ansi`
- Colab local transcript:
  `results/visual_capture_local/colab_exec_visual_retry2_20260611T080748Z.typescript`
- Visual run log:
  `results/20260611T080801Z_visual_capture/llama_diffusion_visual.log`

## Re-render Command

```sh
python3 scripts/render_ansi_capture_video.py \
  results/20260611T080801Z_visual_capture/visual_terminal_capture.ansi \
  videos/l4-quant-generation-flow/renders/diffusiongemma-l4-q4-real-terminal-visual-capture.mp4 \
  --frames-dir videos/l4-quant-generation-flow/qa/render_frames_tmp \
  --background-image assets/backgrounds/sunwood-ai-labs-felt-room-background-maki-banner.png
```

After rendering, extract representative QA frames with ffmpeg if needed.
Generated frame directories are intentionally ignored by Git.

## Public Hygiene

- `node_modules/` is ignored; use `npm install` inside
  `videos/l4-quant-generation-flow/` if video helper dependencies are needed.
- Generated frame directories under `videos/**/qa/*frames*/` are ignored.
- Earlier draft videos and draft QA frames are under
  `videos/l4-quant-generation-flow/archive/` and ignored by Git.
- Do not commit OAuth tokens, Hugging Face tokens, Colab runtime proxy URLs, or
  Google account routing notes.
