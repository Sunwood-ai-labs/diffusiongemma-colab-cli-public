# Run Report: 20260611T044221Z

Status: `generated`

Route:

- Accelerator: L4
- Model repo: `unsloth/diffusiongemma-26B-A4B-it-GGUF`
- Quant include: `*Q4_K_M*`
- Runner: llama.cpp PR `24423`, `llama-diffusion-cli`

Observed:

- GPU: `NVIDIA L4`
- VRAM: `22.034 GiB`
- Build elapsed: `575.346 sec`
- Download elapsed: `65.315 sec`
- Inference elapsed: `29.424 sec`
- Inference return code: `0`

Generated-output quality note:

The runtime generated text, but the answer quality is not accepted as a good DiffusionGemma explanation. It conflated text diffusion with image diffusion. This run proves the L4 + Q4_K_M + `llama-diffusion-cli` path can execute, not that the prompt/output settings are production-ready.

Artifact note:

Drive was not mounted. This was a short smoke run, and artifacts were recovered from `/content` before stopping the session.
