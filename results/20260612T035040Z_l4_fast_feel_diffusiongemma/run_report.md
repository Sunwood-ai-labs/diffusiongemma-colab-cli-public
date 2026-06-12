# DiffusionGemma Fast-Feel L4 Experiment

```json
{
  "status": "completed",
  "n_predict": 256,
  "prompt": "In one compact but vivid technical field note, describe a Mars cafe robot barista calibrating espresso pressure, grind size, and crema quality under 0.38g gravity. Keep it under 190 words and make it feel immediately useful.",
  "git_head": "10a2613aa0b2686f7d0608520c4f0ea05219df03\n10a2613aa diffusion: stop per-step device-sample fallback spam",
  "summary": {
    "best_diffusion_label": "diffusiongemma_eb8",
    "best_diffusion_max_steps": 8,
    "best_diffusion_tokens_per_second": 120.1,
    "best_diffusion_generation_ms": 2131.66,
    "gemma4_generation_tokens_per_second": 69.1,
    "speedup_vs_gemma4_generation_tps": 1.7380607814761215
  },
  "diffusion": [
    {
      "label": "diffusiongemma_eb8",
      "max_steps": 8,
      "returncode": 0,
      "elapsed_sec": 61.07,
      "metrics": {
        "total_ms": 2131.66,
        "step_ms": 266.46,
        "steps": 8,
        "blocks": 1,
        "tokens_per_second": 120.1,
        "tokens": 256,
        "token_ms": 2131.66,
        "in_step_parallel_tokens_per_second": 961.0
      },
      "output_words": 122,
      "cmd": [
        "/content/llama.cpp/build/bin/llama-diffusion-cli",
        "-m",
        "/content/models/unsloth__diffusiongemma-26B-A4B-it-GGUF/diffusiongemma-26B-A4B-it-Q4_K_M.gguf",
        "-p",
        "In one compact but vivid technical field note, describe a Mars cafe robot barista calibrating espresso pressure, grind size, and crema quality under 0.38g gravity. Keep it under 190 words and make it feel immediately useful.",
        "-n",
        "256",
        "--temp",
        "0.2",
        "-ngl",
        "99",
        "-cnv",
        "--diffusion-eb-max-steps",
        "8"
      ]
    },
    {
      "label": "diffusiongemma_eb12",
      "max_steps": 12,
      "returncode": 0,
      "elapsed_sec": 15.201,
      "metrics": {
        "total_ms": 2776.02,
        "step_ms": 231.34,
        "steps": 12,
        "blocks": 1,
        "tokens_per_second": 92.2,
        "tokens": 256,
        "token_ms": 2776.02,
        "in_step_parallel_tokens_per_second": 1107.0
      },
      "output_words": 150,
      "cmd": [
        "/content/llama.cpp/build/bin/llama-diffusion-cli",
        "-m",
        "/content/models/unsloth__diffusiongemma-26B-A4B-it-GGUF/diffusiongemma-26B-A4B-it-Q4_K_M.gguf",
        "-p",
        "In one compact but vivid technical field note, describe a Mars cafe robot barista calibrating espresso pressure, grind size, and crema quality under 0.38g gravity. Keep it under 190 words and make it feel immediately useful.",
        "-n",
        "256",
        "--temp",
        "0.2",
        "-ngl",
        "99",
        "-cnv",
        "--diffusion-eb-max-steps",
        "12"
      ]
    },
    {
      "label": "diffusiongemma_eb16",
      "max_steps": 16,
      "returncode": 0,
      "elapsed_sec": 17.116,
      "metrics": {
        "total_ms": 3575.78,
        "step_ms": 223.49,
        "steps": 16,
        "blocks": 1,
        "tokens_per_second": 71.6,
        "tokens": 256,
        "token_ms": 3575.78,
        "in_step_parallel_tokens_per_second": 1145.0
      },
      "output_words": 155,
      "cmd": [
        "/content/llama.cpp/build/bin/llama-diffusion-cli",
        "-m",
        "/content/models/unsloth__diffusiongemma-26B-A4B-it-GGUF/diffusiongemma-26B-A4B-it-Q4_K_M.gguf",
        "-p",
        "In one compact but vivid technical field note, describe a Mars cafe robot barista calibrating espresso pressure, grind size, and crema quality under 0.38g gravity. Keep it under 190 words and make it feel immediately useful.",
        "-n",
        "256",
        "--temp",
        "0.2",
        "-ngl",
        "99",
        "-cnv",
        "--diffusion-eb-max-steps",
        "16"
      ]
    },
    {
      "label": "diffusiongemma_eb24",
      "max_steps": 24,
      "returncode": 0,
      "elapsed_sec": 17.36,
      "metrics": {
        "total_ms": 4006.95,
        "step_ms": 222.61,
        "steps": 18,
        "blocks": 1,
        "tokens_per_second": 63.9,
        "tokens": 256,
        "token_ms": 4006.95,
        "in_step_parallel_tokens_per_second": 1150.0
      },
      "output_words": 165,
      "cmd": [
        "/content/llama.cpp/build/bin/llama-diffusion-cli",
        "-m",
        "/content/models/unsloth__diffusiongemma-26B-A4B-it-GGUF/diffusiongemma-26B-A4B-it-Q4_K_M.gguf",
        "-p",
        "In one compact but vivid technical field note, describe a Mars cafe robot barista calibrating espresso pressure, grind size, and crema quality under 0.38g gravity. Keep it under 190 words and make it feel immediately useful.",
        "-n",
        "256",
        "--temp",
        "0.2",
        "-ngl",
        "99",
        "-cnv",
        "--diffusion-eb-max-steps",
        "24"
      ]
    },
    {
      "label": "diffusiongemma_eb48",
      "max_steps": 48,
      "returncode": 0,
      "elapsed_sec": 17.102,
      "metrics": {
        "total_ms": 4213.27,
        "step_ms": 221.75,
        "steps": 19,
        "blocks": 1,
        "tokens_per_second": 60.8,
        "tokens": 256,
        "token_ms": 4213.27,
        "in_step_parallel_tokens_per_second": 1154.0
      },
      "output_words": 160,
      "cmd": [
        "/content/llama.cpp/build/bin/llama-diffusion-cli",
        "-m",
        "/content/models/unsloth__diffusiongemma-26B-A4B-it-GGUF/diffusiongemma-26B-A4B-it-Q4_K_M.gguf",
        "-p",
        "In one compact but vivid technical field note, describe a Mars cafe robot barista calibrating espresso pressure, grind size, and crema quality under 0.38g gravity. Keep it under 190 words and make it feel immediately useful.",
        "-n",
        "256",
        "--temp",
        "0.2",
        "-ngl",
        "99",
        "-cnv",
        "--diffusion-eb-max-steps",
        "48"
      ]
    }
  ],
  "gemma4": {
    "returncode": 0,
    "elapsed_sec": 62.29,
    "metrics": {
      "prompt_tokens_per_second": 267.9,
      "generation_tokens_per_second": 69.1
    },
    "output_words": 219,
    "cmd": [
      "/content/llama.cpp/build/bin/llama-cli",
      "-m",
      "/content/models/ggml-org__gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-Q4_K_M.gguf",
      "-p",
      "In one compact but vivid technical field note, describe a Mars cafe robot barista calibrating espresso pressure, grind size, and crema quality under 0.38g gravity. Keep it under 190 words and make it feel immediately useful.",
      "-n",
      "256",
      "--temp",
      "0.2",
      "-ngl",
      "99",
      "-cnv",
      "-st"
    ]
  }
}
```
