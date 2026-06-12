# DiffusionGemma L4 Visual Capture Report

## Summary

- Session: `diffusiongemma-l4-gguf-visual`
- GPU: `NVIDIA L4`, 22.034 GiB VRAM
- Model: `unsloth/diffusiongemma-26B-A4B-it-GGUF`
- Quant: `Q4_K_M`
- Runner: `llama-diffusion-cli`
- Capture mode: raw PTY ANSI terminal output with `--diffusion-visual`

## Evidence

- Raw Colab terminal capture: `visual_terminal_capture.ansi`
- Local `colab exec` terminal transcript: `../visual_capture_local/colab_exec_visual_retry2_20260611T080748Z.typescript`
- llama.cpp visual log: `llama_diffusion_visual.log`
- Rendered replay MP4: `../../videos/l4-quant-generation-flow/renders/diffusiongemma-l4-q4-real-terminal-visual-capture.mp4`
- Sunwood AI Labs color replay MP4: `../../videos/l4-quant-generation-flow/renders/diffusiongemma-l4-q4-real-terminal-visual-capture-sunwood.mp4`
- Public final replay MP4: `../../videos/l4-quant-generation-flow/final/diffusiongemma-l4-q4-real-terminal-visual-capture-sunwood.mp4`
- Readability-fixed replay MP4 draft: `../../videos/l4-quant-generation-flow/archive/renders/diffusiongemma-l4-q4-real-terminal-visual-capture-readable.mp4`
- QA frames:
  - `../../videos/l4-quant-generation-flow/final/qa-009s.png`
  - `../../videos/l4-quant-generation-flow/final/qa-019s.png`
  - `../../videos/l4-quant-generation-flow/qa/real-terminal-sunwood-009s.png`
  - `../../videos/l4-quant-generation-flow/qa/real-terminal-sunwood-019s.png`

## Observed Result

The captured terminal surface showed visual diffusion updates from
`diffusion step: 0/48` through `diffusion step: 20/48`, followed by final output:

```text
total time: 9289.70ms, time per step: 442.37ms (21 steps over 1 blocks, entropy-bound)
throughput: 27.6 tok/s (256 tok in 9289.70ms), in-step parallel 579 tok/s (256-tok canvas x 21.0 steps/block)
```

The generated answer still conflates DiffusionGemma with image generation, so
this is runtime/process evidence, not an answer-quality pass.

## Readability Fix

The first rendered replay used a Latin terminal font path and showed CJK text as
missing-glyph boxes. The renderer now uses a CJK-capable font and line-level
text drawing; user-facing MP4 paths were regenerated and checked against
representative frames with Japanese text visible.

## Sunwood AI Labs Color Rerender

The terminal replay renderer was restyled with the local Sunwood AI Labs color
set: deep green felt background, cream text, gold diffusion-step emphasis, red
and gold header rules, and cyan timing metrics. The background image was copied
into the public repo as
`../../assets/backgrounds/sunwood-ai-labs-felt-room-background-maki-banner.png`
and blended faintly behind the terminal surface. The
canonical replay MP4 path was overwritten with this Sunwood version, and the
separate `*-sunwood.mp4` copy was kept for traceability.

QA check: extracted frames at 9s and 19s from the final MP4 were visually
inspected. Both frames retained readable Japanese text without missing-glyph
boxes and showed the faint background image without obscuring the terminal text.

## Caveat

`result_summary.md` reports `inference_failed` because the capture script's PTY
process-wait path returned `-9` after the final terminal output had already been
captured. The script has been patched to wait for process exit before classifying
the return code in future runs.
