#!/usr/bin/env python3
"""Visual capture example with a Sunwood AI Labs experiment-note prompt."""

from __future__ import annotations

import os
import runpy


os.environ["SESSION_LABEL"] = "diffusiongemma-l4-llama-diffusion-sunwood-note"
os.environ["PROMPT"] = "日本語で、Sunwood AI LabsがL4上で量子化LLMを検証した実験メモを3点だけ簡潔に書いてください。"

runpy.run_path("scripts/run_diffusiongemma_gguf_visual_capture.py", run_name="__main__")
