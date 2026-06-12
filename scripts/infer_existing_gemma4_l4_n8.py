#!/usr/bin/env python3
"""Run a short Gemma 4 GGUF inference using existing Colab build artifacts."""

from __future__ import annotations

import json
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


OUT_ROOT = Path("/content/diffusiongemma-colab-cli-public")
LLAMA_CLI = Path("/content/llama.cpp/build/bin/llama-cli")
MODEL_DIR = Path("/content/models/ggml-org__gemma-4-26B-A4B-it-GGUF")
PROMPT = "Write a concise 3-bullet lab note about a robot barista calibrating espresso shots on a tiny Mars cafe."
N_PREDICT = "8"


def parse_metrics(text: str) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {}
    match = re.search(
        r"prompt eval time\s*=\s*[\d.]+ ms\s*/\s*(\d+) tokens\s*\(\s*([\d.]+) ms per token,\s*([\d.]+) tokens per second\)",
        text,
    )
    if match:
        metrics["prompt_tokens"] = int(match.group(1))
        metrics["prompt_ms_per_token"] = float(match.group(2))
        metrics["prompt_tokens_per_second"] = float(match.group(3))
    match = re.search(
        r"eval time\s*=\s*[\d.]+ ms\s*/\s*(\d+) runs\s*\(\s*([\d.]+) ms per token,\s*([\d.]+) tokens per second\)",
        text,
    )
    if match:
        metrics["decode_tokens"] = int(match.group(1))
        metrics["decode_ms_per_token"] = float(match.group(2))
        metrics["decode_tokens_per_second"] = float(match.group(3))
    match = re.search(r"total time\s*=\s*([\d.]+) ms\s*/\s*(\d+) tokens", text)
    if match:
        metrics["total_ms"] = float(match.group(1))
        metrics["total_tokens"] = int(match.group(2))
    return metrics


def main() -> int:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = OUT_ROOT / f"{stamp}_gemma4_existing_n8_speed"
    out_dir.mkdir(parents=True, exist_ok=True)
    model = sorted(MODEL_DIR.rglob("*.gguf"))[0]
    cmd = [
        str(LLAMA_CLI),
        "-m",
        str(model),
        "-p",
        PROMPT,
        "-n",
        N_PREDICT,
        "--temp",
        "0.2",
        "-ngl",
        "99",
        "--no-warmup",
        "-no-cnv",
    ]
    started = time.time()
    proc = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=900,
        check=False,
    )
    elapsed = round(time.time() - started, 3)
    combined = proc.stdout + "\n" + proc.stderr
    meta = {
        "status": "completed" if proc.returncode == 0 else "failed",
        "returncode": proc.returncode,
        "elapsed_sec": elapsed,
        "cmd": cmd,
        "model": str(model),
        "metrics": parse_metrics(combined),
    }
    (out_dir / "stdout.txt").write_text(proc.stdout, encoding="utf-8")
    (out_dir / "stderr.txt").write_text(proc.stderr, encoding="utf-8")
    (out_dir / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)
    print(f"RESULT_DIR={out_dir}", flush=True)
    return 0 if proc.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
