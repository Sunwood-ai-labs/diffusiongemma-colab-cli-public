# Browser-Owned Colab L4 Gemma4 Completion Speed Check

Date: 2026-06-12 JST

## Route

- Route label: browser-owned-adopted
- Browser surface: Colab UI opened in Chrome CDP profile and connected to L4
- CLI surface: browser-owned `[?]` assignment adopted with a temporary `--config`
- Endpoint: `gpu-l4-s-kkb-ass1a2-2ue03uoev4dhq`
- GPU proof: `NVIDIA L4`, 22.034 GiB from inside Colab

## Correct command

Regular Gemma4 non-conversation benchmarking must use `llama-completion`, not `llama-cli`.

Observed during verification:

- `llama-cli -no-cnv` entered chat/interactive prompt mode and timed out.
- `llama-cli -no-cnv -st` exited, but printed `--no-conversation is not supported by llama-cli; please use llama-completion instead`.
- `llama-completion` required `--jinja` for this Gemma4 custom chat template.

Final command:

```text
/content/llama.cpp/build/bin/llama-completion \
  -m /content/models/ggml-org__gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-Q4_K_M.gguf \
  -p "Write a concise 3-bullet lab note about a robot barista calibrating espresso shots on a tiny Mars cafe." \
  -n 8 --temp 0.2 -ngl 99 --no-warmup --jinja -no-cnv
```

## Result

- Prompt eval: 24 tokens, 13.35 ms/token, 74.91 tok/s
- Decode eval: 7 runs, 18.86 ms/token, 53.01 tok/s
- Total timing: 462.40 ms / 31 tokens
- Command elapsed wall time: 12.192 s including model load/process overhead
- Return code: 0

## Comparison notes

Existing DiffusionGemma L4 visual-capture runs in this repo reported:

- Robot-barista prompt: 4551.07 ms total, 216.72 ms/step, 56.3 tok/s over a 256-token canvas
- Sunwood note prompt: 4754.52 ms total, 216.11 ms/step, 53.8 tok/s over a 256-token canvas
- Earlier Japanese explainer prompt: 9289.70 ms total, 442.37 ms/step, 27.6 tok/s over a 256-token canvas

Conclusion: the claim "DiffusionGemma is slower than regular Gemma4" is not supported by the tok/s numbers from this local check. On the closest robot-barista prompt, DiffusionGemma reported 56.3 tok/s, while regular Gemma4 completion decode reported 53.01 tok/s. However, these are not perfectly equivalent measurements: DiffusionGemma reports diffusion/canvas throughput over 256 tokens and steps, while regular Gemma4 reports sequential decode timing for a short 8-token completion.

## Local Evidence

- `metadata.json`
- `stdout.txt`
- `stderr.txt`
