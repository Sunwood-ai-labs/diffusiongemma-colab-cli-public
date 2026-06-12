#!/usr/bin/env python3
"""Run the corrected Gemma 4 L4 benchmark with 8 generated tokens."""

from __future__ import annotations

import os
import runpy


os.environ["GEMMA4_N_PREDICT"] = "8"
runpy.run_path("/content/scripts/benchmark_gemma4_nocnv_l4.py", run_name="__main__")
