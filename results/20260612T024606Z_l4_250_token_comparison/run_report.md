# L4 250+ Token Comparison

- Status: completed
- N predict: 320
- Prompt: Write a detailed technical lab note of at least 280 English words about a robot barista calibrating espresso shots in a tiny Mars cafe. Include sections for environment, calibration procedure, observations, failure modes, and conclusion. Keep the answer continuous and do not stop early.
- Git head: 10a2613aa0b2686f7d0608520c4f0ea05219df03
10a2613aa diffusion: stop per-step device-sample fallback spam

## Metrics

```json
{
  "diffusiongemma": {
    "diffusion_total_ms": 9222.39,
    "diffusion_step_ms": 224.94,
    "diffusion_steps": 41,
    "diffusion_blocks": 2,
    "diffusion_canvas_tokens_per_second": 55.5,
    "diffusion_canvas_tokens": 512,
    "diffusion_canvas_total_ms": 9222.39,
    "diffusion_in_step_parallel_tokens_per_second": 1138.0
  },
  "gemma4": {
    "prompt_tokens": 56,
    "prompt_ms_per_token": 4.84,
    "prompt_tokens_per_second": 206.53,
    "decode_tokens": 319,
    "decode_ms_per_token": 13.85,
    "decode_tokens_per_second": 72.18,
    "ar_total_ms": 4937.75,
    "ar_total_tokens": 375
  },
  "output_words": {
    "diffusiongemma": 349,
    "gemma4": 53
  },
  "elapsed_sec": {
    "diffusiongemma": 75.534,
    "gemma4": 67.144
  }
}
```
