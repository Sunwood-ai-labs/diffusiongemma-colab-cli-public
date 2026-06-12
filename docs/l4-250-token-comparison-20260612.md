# DiffusionGemma vs Gemma 4 L4 250+ token comparison

Date: 2026-06-12 JST

## Purpose

This run follows up on upstream feedback that an 8-token regular Gemma 4
baseline is too short to compare against DiffusionGemma's 256-token canvas.

## Route and caveat

- Route: CLI-created Colab L4 diagnostic run
- GPU: NVIDIA L4, 23034 MiB
- Upstream PR checkout: `10a2613aa0b2686f7d0608520c4f0ea05219df03`
- Requested generation length: `-n 320` for both commands
- Caveat: Colab CLI keep-alive still failed with `403 USER_PROJECT_DENIED`, but the run completed before the session was pruned. Treat this as a completed diagnostic result, not proof that the CLI-created runtime is safe for long jobs.
- Caveat: Gemma 4 produced 319 decode tokens, but the visible output was mostly repeated bullet/star tokens. The timing is usable as a speed datapoint; the run should not be used as a quality comparison.

## Prompt

```text
Write a detailed technical lab note of at least 280 English words about a robot barista calibrating espresso shots in a tiny Mars cafe. Include sections for environment, calibration procedure, observations, failure modes, and conclusion. Keep the answer continuous and do not stop early.
```

## Commands

DiffusionGemma:

```text
/content/llama.cpp/build/bin/llama-diffusion-cli \
  -m /content/models/unsloth__diffusiongemma-26B-A4B-it-GGUF/diffusiongemma-26B-A4B-it-Q4_K_M.gguf \
  -p "$PROMPT" \
  -n 320 --temp 0.2
```

Gemma 4:

```text
/content/llama.cpp/build/bin/llama-completion \
  -m /content/models/ggml-org__gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-Q4_K_M.gguf \
  -p "$PROMPT" \
  -n 320 --temp 0.2 -ngl 99 --no-warmup --jinja -no-cnv
```

## Results

| Model | Internal timing | Reported throughput | Token count |
| --- | ---: | ---: | ---: |
| DiffusionGemma Q4_K_M | 9222.39 ms | 55.5 canvas tok/s | 512 canvas tok |
| Gemma 4 Q4_K_M | 4937.75 ms | 72.18 decode tok/s | 319 decode tok |

Additional timing:

- DiffusionGemma command elapsed: 75.534 s
- Gemma 4 command elapsed: 67.144 s
- DiffusionGemma: 41 diffusion steps over 2 blocks, 224.94 ms/step
- Gemma 4 prompt eval: 56 tokens at 206.53 tok/s

## Interpretation

On this L4 diagnostic run, the earlier "DiffusionGemma is not slower" conclusion
does not hold under a 250+ token comparison. For this prompt and PR head,
regular Gemma 4 reports faster autoregressive decode throughput than
DiffusionGemma reports canvas throughput: 72.18 tok/s vs 55.5 tok/s.

The metrics are still not perfectly equivalent because DiffusionGemma's canvas
throughput and Gemma 4's autoregressive decode throughput measure different
generation mechanisms. The safest upstream phrasing is: on Colab L4 Q4_K_M,
with both commands requesting 320 tokens, DiffusionGemma reported 55.5 canvas
tok/s while regular Gemma 4 reported 72.18 decode tok/s.

## Evidence

- `results/20260612T024606Z_l4_250_token_comparison/metadata.json`
- `results/20260612T024606Z_l4_250_token_comparison/run_report.md`
- `results/20260612T024606Z_l4_250_token_comparison/progress.jsonl`
- `results/20260612T024606Z_l4_250_token_comparison/diffusiongemma.stdout.txt`
- `results/20260612T024606Z_l4_250_token_comparison/diffusiongemma.stderr.txt`
- `results/20260612T024606Z_l4_250_token_comparison/gemma4.stdout.txt`
- `results/20260612T024606Z_l4_250_token_comparison/gemma4.stderr.txt`
