#!/usr/bin/env python3
"""Colab-side short Gemma 4 GGUF speed probe on L4."""

from __future__ import annotations

import json
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


OUT_ROOT = Path("/content/diffusiongemma-colab-cli-public")
LLAMA_DIR = Path("/content/llama.cpp")
MODEL_ROOT = Path("/content/models")
PROMPT = "Write a concise 3-bullet lab note about a robot barista calibrating espresso shots on a tiny Mars cafe."


def run(cmd: str | list[str], timeout: int = 120, cwd: Path | None = None, tail: int = 60000) -> dict[str, Any]:
    started = time.time()
    try:
        proc = subprocess.run(
            cmd,
            shell=isinstance(cmd, str),
            cwd=str(cwd) if cwd else None,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        return {
            "cmd": cmd,
            "returncode": proc.returncode,
            "stdout": proc.stdout[-tail:],
            "stderr": proc.stderr[-tail:],
            "elapsed_sec": round(time.time() - started, 3),
        }
    except Exception as exc:  # noqa: BLE001
        return {"cmd": cmd, "error": repr(exc), "elapsed_sec": round(time.time() - started, 3)}


def append(path: Path, event: str, **fields: Any) -> None:
    row = {"ts": datetime.now(timezone.utc).isoformat(), "event": event, **fields}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(row, ensure_ascii=False), flush=True)


def parse_llama_metrics(text: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    match = re.search(r"eval time\s*=\s*[\d.]+ ms\s*/\s*(\d+) runs\s*\(\s*([\d.]+) ms per token,\s*([\d.]+) tokens per second\)", text)
    if match:
        metrics["decode_tokens"] = int(match.group(1))
        metrics["decode_ms_per_token"] = float(match.group(2))
        metrics["decode_tokens_per_second"] = float(match.group(3))
    match = re.search(r"prompt eval time\s*=\s*[\d.]+ ms\s*/\s*(\d+) tokens\s*\(\s*([\d.]+) ms per token,\s*([\d.]+) tokens per second\)", text)
    if match:
        metrics["prompt_tokens"] = int(match.group(1))
        metrics["prompt_ms_per_token"] = float(match.group(2))
        metrics["prompt_tokens_per_second"] = float(match.group(3))
    return metrics


def main() -> int:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = OUT_ROOT / f"{stamp}_gemma4_short_speed"
    out_dir.mkdir(parents=True, exist_ok=True)
    progress = out_dir / "progress.jsonl"
    meta: dict[str, Any] = {"prompt": PROMPT, "output_dir": str(out_dir)}
    append(progress, "start", **meta)

    try:
        try:
            import torch

            meta["gpu"] = {
                "name": torch.cuda.get_device_name(0),
                "total_memory_gib": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 3),
            }
        except Exception as exc:  # noqa: BLE001
            meta["gpu_error"] = repr(exc)
        append(progress, "gpu", **meta.get("gpu", {}))

        for name, cmd, timeout in [
            ("apt_update", "apt-get update -y", 240),
            ("apt_install", "apt-get install -y cmake ninja-build git git-lfs build-essential", 360),
            ("pip_install", f"{shlex.quote(sys.executable)} -m pip install -U 'huggingface_hub[cli]'", 300),
        ]:
            append(progress, f"{name}_start")
            meta[name] = run(cmd, timeout=timeout)
            append(progress, f"{name}_done", returncode=meta[name].get("returncode"), elapsed_sec=meta[name].get("elapsed_sec"))
            if meta[name].get("returncode") != 0:
                raise RuntimeError(f"{name} failed")

        if not LLAMA_DIR.exists():
            append(progress, "clone_start")
            meta["clone"] = run("git clone --depth 1 https://github.com/ggml-org/llama.cpp /content/llama.cpp", timeout=600)
            append(progress, "clone_done", returncode=meta["clone"].get("returncode"))
            if meta["clone"].get("returncode") != 0:
                raise RuntimeError("clone failed")

        append(progress, "fetch_pr_start")
        meta["fetch_pr"] = run("git fetch origin pull/24423/head", timeout=600, cwd=LLAMA_DIR)
        append(progress, "fetch_pr_done", returncode=meta["fetch_pr"].get("returncode"))
        append(progress, "checkout_start")
        meta["checkout"] = run("git checkout -B diffusiongemma FETCH_HEAD", timeout=120, cwd=LLAMA_DIR)
        append(progress, "checkout_done", returncode=meta["checkout"].get("returncode"))

        append(progress, "cmake_config_start")
        meta["cmake_config"] = run("cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release", timeout=600, cwd=LLAMA_DIR)
        append(progress, "cmake_config_done", returncode=meta["cmake_config"].get("returncode"))
        append(progress, "build_start")
        meta["build"] = run("cmake --build build -j --config Release --target llama-cli", timeout=1800, cwd=LLAMA_DIR)
        append(progress, "build_done", returncode=meta["build"].get("returncode"), elapsed_sec=meta["build"].get("elapsed_sec"))
        if meta["build"].get("returncode") != 0:
            raise RuntimeError("build failed")

        model_dir = MODEL_ROOT / "ggml-org__gemma-4-26B-A4B-it-GGUF"
        model_dir.mkdir(parents=True, exist_ok=True)
        append(progress, "download_start", repo="ggml-org/gemma-4-26B-A4B-it-GGUF", include="*Q4_K_M*")
        meta["download"] = run(["hf", "download", "ggml-org/gemma-4-26B-A4B-it-GGUF", "--local-dir", str(model_dir), "--include", "*Q4_K_M*"], timeout=3600)
        append(progress, "download_done", returncode=meta["download"].get("returncode"), elapsed_sec=meta["download"].get("elapsed_sec"))
        if meta["download"].get("returncode") != 0:
            raise RuntimeError("download failed")

        ggufs = sorted(model_dir.rglob("*.gguf"))
        if not ggufs:
            raise RuntimeError("no GGUF")
        model = ggufs[0]
        meta["model"] = str(model)

        cmd = [
            str(LLAMA_DIR / "build/bin/llama-cli"),
            "-m",
            str(model),
            "-p",
            PROMPT,
            "-n",
            "32",
            "--temp",
            "0.2",
            "-ngl",
            "99",
            "--no-warmup",
        ]
        append(progress, "inference_start", cmd=cmd)
        meta["inference"] = run(cmd, timeout=300)
        append(progress, "inference_done", returncode=meta["inference"].get("returncode"), elapsed_sec=meta["inference"].get("elapsed_sec"))
        (out_dir / "gemma4_short.stdout.txt").write_text(meta["inference"].get("stdout", ""), encoding="utf-8")
        (out_dir / "gemma4_short.stderr.txt").write_text(meta["inference"].get("stderr", ""), encoding="utf-8")
        meta["metrics"] = parse_llama_metrics(meta["inference"].get("stdout", "") + "\n" + meta["inference"].get("stderr", ""))
        meta["status"] = "completed" if meta["inference"].get("returncode") == 0 else "inference_failed"
    except Exception as exc:  # noqa: BLE001
        meta["status"] = "failed"
        meta["error"] = repr(exc)
        append(progress, "failed", error=repr(exc))

    meta["completed_at"] = datetime.now(timezone.utc).isoformat()
    (out_dir / "gemma4_short_benchmark_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "gemma4_short_speed_report.md").write_text(
        "# Gemma 4 Short L4 Speed Probe\n\n"
        + json.dumps({"status": meta["status"], "gpu": meta.get("gpu"), "metrics": meta.get("metrics"), "error": meta.get("error")}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(f"RESULT_DIR={out_dir}", flush=True)
    print(f"RESULT_STATUS={meta['status']}", flush=True)
    return 0 if meta["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
