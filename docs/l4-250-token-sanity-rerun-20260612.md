# L4 250+ token sanity rerun with explicit GPU layers

Date: 2026-06-12 JST

## Purpose

This rerun checks whether the previous L4 250+ token comparison was skewed by
questionable settings.

Changes from the first 250+ token comparison:

- DiffusionGemma explicitly used `-ngl 99` instead of relying on the default.
- Regular Gemma 4 used `llama-cli -cnv -st` instead of
  `llama-completion --jinja -no-cnv`, because the first baseline produced
  mostly repeated bullet/star tokens.

## Route and caveat

- Route: CLI-created Colab L4 diagnostic run
- GPU: NVIDIA L4, 23034 MiB
- Upstream PR checkout: `10a2613aa0b2686f7d0608520c4f0ea05219df03`
- llama.cpp build: `b9600-10a2613aa`
- Requested generation length: `-n 320` for both commands
- Colab CLI keep-alive failed with `403 USER_PROJECT_DENIED` and then stopped
  after two consecutive 4xx errors. The run completed before session pruning,
  so this is a completed short diagnostic result, not proof that this
  CLI-created runtime is safe for long jobs.

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
  -n 320 --temp 0.2 -ngl 99
```

Gemma 4:

```text
/content/llama.cpp/build/bin/llama-cli \
  -m /content/models/ggml-org__gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-Q4_K_M.gguf \
  -p "$PROMPT" \
  -n 320 --temp 0.2 -ngl 99 -cnv -st
```

## Results

| Model | Command elapsed | Internal/report timing | Reported throughput | Output words |
| --- | ---: | ---: | ---: | ---: |
| DiffusionGemma Q4_K_M | 74.778 s | 9191.46 ms, 41 steps over 2 blocks | 55.7 canvas tok/s | 349 |
| Gemma 4 Q4_K_M | 67.549 s | llama-cli summary | 67.7 generation tok/s | 252 |

Additional DiffusionGemma details:

- Time per diffusion step: 224.18 ms
- Canvas tokens: 512
- In-step parallel throughput: 1142.0 tok/s
- Runtime log: `gpu_sampling=on sample_reduce=on`

## Interpretation

With the cleaner settings, the speed result still points the same way:
DiffusionGemma was slower than regular Gemma 4 on this Colab L4 diagnostic run.

The narrow comparison is:

- DiffusionGemma: 55.7 canvas tok/s
- Gemma 4: 67.7 generation tok/s

That makes DiffusionGemma about 17.7% lower by reported throughput under this
specific setup. By whole-command elapsed time, DiffusionGemma was about 10.7%
slower.

This should not be generalized as "DiffusionGemma is always slower". It is a
single L4 Q4_K_M diagnostic against PR `10a2613aa`, with different generation
mechanics and a CLI-created Colab runtime whose keep-alive path is still broken.

## Evidence

- `results/20260612T031109Z_l4_250_token_sanity/metadata.json`
- `results/20260612T031109Z_l4_250_token_sanity/run_report.md`
- `results/20260612T031109Z_l4_250_token_sanity/progress.jsonl`
- `results/20260612T031109Z_l4_250_token_sanity/diffusiongemma_explicit_ngl.stdout.txt`
- `results/20260612T031109Z_l4_250_token_sanity/diffusiongemma_explicit_ngl.stderr.txt`
- `results/20260612T031109Z_l4_250_token_sanity/gemma4_cli_chat.stdout.txt`
- `results/20260612T031109Z_l4_250_token_sanity/gemma4_cli_chat.stderr.txt`
