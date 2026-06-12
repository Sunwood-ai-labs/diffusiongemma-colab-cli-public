# DiffusionGemma fast-feel L4 experiment

Date: 2026-06-12 JST

## Purpose

This experiment checks whether DiffusionGemma can be made to feel faster than
regular Gemma 4 on Colab L4 by using a setup that matches its core advantage:
one 256-token canvas and fewer denoising steps.

This is not a quality benchmark. It is a latency/throughput demonstration.

## Route and caveat

- Route: CLI-created Colab L4 diagnostic run
- GPU: NVIDIA L4, 23034 MiB
- Upstream PR checkout: `10a2613aa0b2686f7d0608520c4f0ea05219df03`
- llama.cpp build: `b9600-10a2613aa`
- Requested generation length: `-n 256`
- DiffusionGemma model: `diffusiongemma-26B-A4B-it-Q4_K_M.gguf`
- Gemma 4 model: `gemma-4-26B-A4B-it-Q4_K_M.gguf`
- Colab CLI keep-alive still fails with `403 USER_PROJECT_DENIED`; treat this
  as a completed short diagnostic run, not a long-run-safe Colab route.

## Prompt

```text
In one compact but vivid technical field note, describe a Mars cafe robot barista calibrating espresso pressure, grind size, and crema quality under 0.38g gravity. Keep it under 190 words and make it feel immediately useful.
```

## Commands

DiffusionGemma fast-feel shape:

```text
/content/llama.cpp/build/bin/llama-diffusion-cli \
  -m /content/models/unsloth__diffusiongemma-26B-A4B-it-GGUF/diffusiongemma-26B-A4B-it-Q4_K_M.gguf \
  -p "$PROMPT" \
  -n 256 --temp 0.2 -ngl 99 -cnv \
  --diffusion-eb-max-steps <8|12|16|24|48>
```

Gemma 4 baseline:

```text
/content/llama.cpp/build/bin/llama-cli \
  -m /content/models/ggml-org__gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-Q4_K_M.gguf \
  -p "$PROMPT" \
  -n 256 --temp 0.2 -ngl 99 -cnv -st
```

## Results

| Run | Generation time | Reported throughput | Steps | Output words | Readability note |
| --- | ---: | ---: | ---: | ---: | --- |
| DiffusionGemma EB 8 | 2131.66 ms | 120.1 tok/s | 8 | 122 | Fastest, visibly degraded |
| DiffusionGemma EB 12 | 2776.02 ms | 92.2 tok/s | 12 | 150 | Fast, still degraded |
| DiffusionGemma EB 16 | 3575.78 ms | 71.6 tok/s | 16 | 155 | Slightly faster than Gemma 4, degraded |
| DiffusionGemma EB 24 | 4006.95 ms | 63.9 tok/s | 18 | 165 | Slower than Gemma 4 |
| DiffusionGemma EB 48 | 4213.27 ms | 60.8 tok/s | 19 | 160 | Slower than Gemma 4 |
| Gemma 4 baseline | n/a | 69.1 generation tok/s | n/a | 219 | Baseline |

## Interpretation

Yes, DiffusionGemma can be made to feel faster on L4 when the experiment is
shaped around its 256-token canvas and the denoising step budget is reduced.

Best speed datapoint:

- DiffusionGemma EB 8: 120.1 tok/s
- Gemma 4 baseline: 69.1 tok/s
- Speedup: 1.74x by reported generation throughput

More conservative datapoint:

- DiffusionGemma EB 16: 71.6 tok/s
- Gemma 4 baseline: 69.1 tok/s
- Speedup: 1.04x by reported generation throughput

The tradeoff is visible quality. EB 8 and EB 12 are fast but rough. EB 16 is the
first setting that still beats Gemma 4 in this run, but the output is not clean
enough to call it a quality win.

## Practical takeaway

For a speed demo, use a 256-token target and a low EB step budget:

```text
-n 256 --diffusion-eb-max-steps 8
```

For a less aggressive demo:

```text
-n 256 --diffusion-eb-max-steps 16
```

For a balanced article/demo, report both:

- "fast-preview mode": 8 steps, 120.1 tok/s, visibly rough
- "less aggressive mode": 16 steps, 71.6 tok/s, just above Gemma 4 on this L4 run

## Evidence

- `results/20260612T035040Z_l4_fast_feel_diffusiongemma/metadata.json`
- `results/20260612T035040Z_l4_fast_feel_diffusiongemma/run_report.md`
- `results/20260612T035040Z_l4_fast_feel_diffusiongemma/progress.jsonl`
- `results/20260612T035040Z_l4_fast_feel_diffusiongemma/diffusiongemma_eb8.stdout.txt`
- `results/20260612T035040Z_l4_fast_feel_diffusiongemma/diffusiongemma_eb12.stdout.txt`
- `results/20260612T035040Z_l4_fast_feel_diffusiongemma/diffusiongemma_eb16.stdout.txt`
- `results/20260612T035040Z_l4_fast_feel_diffusiongemma/diffusiongemma_eb24.stdout.txt`
- `results/20260612T035040Z_l4_fast_feel_diffusiongemma/diffusiongemma_eb48.stdout.txt`
- `results/20260612T035040Z_l4_fast_feel_diffusiongemma/gemma4.stdout.txt`
