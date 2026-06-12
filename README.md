# DiffusionGemma Colab CLI Experiment

Public, reproducible experiment scaffold for trying Google's DiffusionGemma from Google Colab CLI.

## Repository layout

This folder is prepared as a public repository:

```text
diffusiongemma-colab-cli-public/
```

Start with the public manifest for exact artifact locations:

- `docs/PUBLIC_REPO_MANIFEST.md`

Main folders:

- `scripts/`: Colab experiment scripts and the ANSI-to-MP4 renderer.
- `results/`: recovered Colab evidence, metadata, logs, and raw terminal capture.
- `docs/`: experiment notes and publication manifest.
- `assets/backgrounds/`: public background asset used in the final video.
- `videos/l4-quant-generation-flow/final/`: final Sunwood AI Labs color demo video and QA frames.
- `videos/l4-quant-generation-flow/renders/`: canonical renderer output path.
- `videos/l4-quant-generation-flow/archive/`: older draft renders and QA frames, ignored by Git.

## What this tests

- Colab CLI session creation and GPU assignment.
- Google Drive mounted inside Colab for the full Transformers probe, and local `/content` result recovery for the GGUF L4/T4 route.
- Hugging Face model reachability for `google/diffusiongemma-26B-A4B-it`.
- Whether the assigned GPU has enough memory to run a tiny DiffusionGemma generation.

The official Hugging Face Transformers notebook says the model requires a GPU with more than 60 GB of memory, such as NVIDIA G4 or H100. This repo therefore records insufficient-VRAM outcomes as a valid experiment result instead of pretending a T4 smoke test proves model execution.

## Official references

- Google AI docs: https://ai.google.dev/gemma/docs/diffusiongemma/inference-diffusiongemma-with-hf
- Google Developers Blog: https://developers.googleblog.com/diffusiongemma-the-developer-guide/
- Hugging Face model: https://huggingface.co/google/diffusiongemma-26B-A4B-it

## Colab CLI run

```sh
SESSION=diffusiongemma-h100-smoke
colab new -s "$SESSION" --gpu H100
colab drivemount -s "$SESSION"
colab upload -s "$SESSION" scripts/run_diffusiongemma_probe.py /content/run_diffusiongemma_probe.py
colab exec -s "$SESSION" --timeout 1800 -f /content/run_diffusiongemma_probe.py
colab stop -s "$SESSION"
```

If H100 is unavailable, use `--gpu G4` or `--gpu A100` only as a separate named run. For L4/T4, use the GGUF quantized route below instead of the full Transformers route.

## L4/T4 quantized route

The smaller route uses Unsloth's GGUF quantization and the DiffusionGemma-specific `llama-diffusion-cli` runner from the llama.cpp PR branch. This is the intended path for L4/T4-class experiments.

```sh
SESSION=diffusiongemma-l4-quant-smoke
colab new -s "$SESSION" --gpu L4
colab exec -s "$SESSION" --timeout 7200 -f scripts/run_diffusiongemma_gguf_probe.py
colab stop -s "$SESSION"
```

Visual terminal capture route:

```sh
SESSION=diffusiongemma-l4-gguf-visual
colab new -s "$SESSION" --gpu L4
colab exec -s "$SESSION" --timeout 7200 -f scripts/run_diffusiongemma_gguf_visual_capture.py
colab stop -s "$SESSION"
```

Default quantized model:

```text
unsloth/diffusiongemma-26B-A4B-it-GGUF
include: *Q4_K_M*
```

Notes:

- Q4_K_M is a smaller GGUF intended to fit a single 24 GB GPU.
- The standard `llama-cli` / `llama-server` path is not sufficient for DiffusionGemma generation yet; use `llama-diffusion-cli`.
- NVFP4 checkpoints exist, but the published NVIDIA card targets Blackwell/Hopper/vLLM rather than T4/L4 as the first route.

## Output

The script saves:

- `experiment_metadata.json`
- `progress.jsonl`
- `result_summary.md`
- `generation_stdout.txt` and `generation_stderr.txt` for the GGUF smoke test
- `visual_terminal_capture.ansi`, `generation_terminal_tail.txt`, and `llama_diffusion_visual.log` for the visual capture test

Default full Transformers Drive output:

```text
/content/drive/MyDrive/diffusiongemma-colab-cli-public/<timestamp>/
```

Default GGUF L4/T4 output inside the Colab runtime:

```text
/content/diffusiongemma-colab-cli-public/<timestamp>/
```

## Final demo video

The current public demo is the real terminal replay captured from
`llama-diffusion-cli --diffusion-visual`, rendered with the Sunwood AI Labs color
palette and a faint felt-room background from
`assets/backgrounds/sunwood-ai-labs-felt-room-background-maki-banner.png`.

Original explainer MP4:

```text
videos/l4-quant-generation-flow/final/diffusiongemma-l4-q4-real-terminal-visual-capture-sunwood.mp4
```

Current alternate example MP4:

```text
videos/l4-quant-generation-flow/final/diffusiongemma-l4-q4-robot-barista-llama-diffusion-cli.mp4
```

Current alternate example iOS-compatible MP4:

```text
videos/l4-quant-generation-flow/final/diffusiongemma-robot-barista-ios-compatible.mp4
```

Original explainer iOS-compatible MP4:

```text
videos/l4-quant-generation-flow/final/diffusiongemma-sunwood-demo-ios-compatible.mp4
```

Current alternate example mobile preview screenshot:

```text
videos/l4-quant-generation-flow/final/robot-barista-mobile-preview-screenshot.png
```

Current alternate example QA frames:

```text
videos/l4-quant-generation-flow/final/robot-barista-qa-009s.png
videos/l4-quant-generation-flow/final/robot-barista-qa-019s.png
```

Re-render from the raw terminal capture:

```sh
python3 scripts/render_ansi_capture_video.py \
  results/20260611T080801Z_visual_capture/visual_terminal_capture.ansi \
  videos/l4-quant-generation-flow/renders/diffusiongemma-l4-q4-real-terminal-visual-capture.mp4 \
  --frames-dir videos/l4-quant-generation-flow/qa/render_frames_tmp \
  --background-image assets/backgrounds/sunwood-ai-labs-felt-room-background-maki-banner.png
```

Re-render the robot-barista example:

```sh
python3 scripts/render_ansi_capture_video.py \
  results/20260611T113411Z_robot_barista_visual_capture/visual_terminal_capture.ansi \
  videos/l4-quant-generation-flow/renders/diffusiongemma-l4-q4-robot-barista-llama-diffusion-cli.mp4 \
  --frames-dir videos/l4-quant-generation-flow/qa/robot_barista_frames \
  --background-image assets/backgrounds/sunwood-ai-labs-felt-room-background-maki-banner.png
```

## Public repo hygiene

Do not commit OAuth tokens, Hugging Face tokens, Colab runtime proxy URLs, or Google account routing notes. Keep private run notes in ignored `*.local.*` files.
