# Draft comment for ggml-org/llama.cpp PR #24423

Posted:

- https://github.com/ggml-org/llama.cpp/pull/24423#issuecomment-4686635225

I tested the current DiffusionGemma path on a Google Colab L4 with Q4_K_M and
wanted to add one L4 data point, especially because several comments compare
DiffusionGemma throughput with regular Gemma 4 decode speed.

Environment:

- Hardware: Google Colab L4, 22.0 GiB VRAM
- Diffusion model: `unsloth/diffusiongemma-26B-A4B-it-GGUF`, Q4_K_M
- Runner: `llama-diffusion-cli`
- Baseline model: `ggml-org/gemma-4-26B-A4B-it-GGUF`, Q4_K_M
- Baseline runner: `llama-completion`
- Prompt used for both runs: `Write a concise 3-bullet lab note about a robot barista calibrating espresso shots on a tiny Mars cafe.`

DiffusionGemma command:

```text
/content/llama.cpp/build/bin/llama-diffusion-cli \
  -m /content/models/unsloth__diffusiongemma-26B-A4B-it-GGUF/diffusiongemma-26B-A4B-it-Q4_K_M.gguf \
  -p "Write a concise 3-bullet lab note about a robot barista calibrating espresso shots on a tiny Mars cafe." \
  -n 96 --temp 0.2 \
  --diffusion-visual --diffusion-visual-progress --diffusion-visual-interval 1
```

DiffusionGemma result:

```text
total time: 4551.07ms, time per step: 216.72ms (21 steps over 1 blocks, entropy-bound)
throughput: 56.3 tok/s (256 tok in 4551.07ms), in-step parallel 1181 tok/s (256-tok canvas x 21.0 steps/block)
```

Regular Gemma 4 baseline command:

```text
/content/llama.cpp/build/bin/llama-completion \
  -m /content/models/ggml-org__gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-Q4_K_M.gguf \
  -p "Write a concise 3-bullet lab note about a robot barista calibrating espresso shots on a tiny Mars cafe." \
  -n 8 --temp 0.2 -ngl 99 --no-warmup --jinja -no-cnv
```

Regular Gemma 4 result:

```text
prompt eval time = 320.41 ms / 24 tokens (13.35 ms per token, 74.91 tokens per second)
eval time = 132.05 ms / 7 runs (18.86 ms per token, 53.01 tokens per second)
total time = 462.40 ms / 31 tokens
```

My takeaway from this L4 run:

- On this specific prompt and Q4_K_M setup, I do not see DiffusionGemma being obviously slower than the regular Gemma 4 short decode baseline if I compare the reported `tok/s` numbers directly: 56.3 tok/s vs 53.01 tok/s.
- However, I do not think these two numbers are strictly equivalent. DiffusionGemma is reporting diffusion/canvas throughput over a 256-token canvas and entropy-bound steps, while `llama-completion` reports sequential decode timing for a short completion.
- It may be useful for the PR to document a recommended apples-to-apples benchmarking recipe for DiffusionGemma versus regular Gemma 4, including whether to compare end-to-end wall time, reported throughput, or some normalized effective-token metric.

Related note: for regular Gemma 4 I had to use `llama-completion --jinja -no-cnv`.
Using `llama-cli -no-cnv` entered interactive/chat behavior or warned that
`--no-conversation` is not supported by `llama-cli`.
