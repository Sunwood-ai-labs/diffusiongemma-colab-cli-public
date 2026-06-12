# L4 250+ Token Sanity Rerun

```json
{
  "status": "completed",
  "n_predict": 320,
  "git_head": "10a2613aa0b2686f7d0608520c4f0ea05219df03\n10a2613aa diffusion: stop per-step device-sample fallback spam",
  "metrics": {
    "diffusiongemma_explicit_ngl": {
      "diffusion_total_ms": 9191.46,
      "diffusion_step_ms": 224.18,
      "diffusion_steps": 41,
      "diffusion_blocks": 2,
      "diffusion_canvas_tokens_per_second": 55.7,
      "diffusion_canvas_tokens": 512,
      "diffusion_canvas_total_ms": 9191.46,
      "diffusion_in_step_parallel_tokens_per_second": 1142.0
    },
    "gemma4_cli_chat": {
      "prompt_tokens_per_second": 249.6,
      "generation_tokens_per_second": 67.7
    }
  },
  "output_words": {
    "diffusiongemma_explicit_ngl": 349,
    "gemma4_cli_chat": 252
  },
  "elapsed_sec": {
    "diffusiongemma_explicit_ngl": 74.778,
    "gemma4_cli_chat": 67.549
  }
}
```

Gemma 4 metrics were parsed post-hoc from the llama-cli stdout summary:
`[ Prompt: 249.6 t/s | Generation: 67.7 t/s ]`.
