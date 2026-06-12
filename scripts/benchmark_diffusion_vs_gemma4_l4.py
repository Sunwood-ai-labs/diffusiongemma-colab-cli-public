#!/usr/bin/env python3
"""Colab-side L4 benchmark: DiffusionGemma vs regular Gemma 4.

The goal is not a perfect academic benchmark. It records a same-runtime,
same-llama.cpp-checkout smoke comparison so claims about "slower/faster than
regular Gemma 4" are tied to evidence.
"""

from __future__ import annotations

import json
import os
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


def run(cmd: str | list[str], timeout: int = 120, cwd: Path | None = None, tail: int = 40000) -> dict[str, Any]:
    started = time.time()
    shell = isinstance(cmd, str)
    try:
        proc = subprocess.run(
            cmd,
            shell=shell,
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


def gpu_snapshot() -> dict[str, Any]:
    snap = {"nvidia_smi": run("nvidia-smi || true", timeout=30)}
    try:
        import torch

        snap["torch_version"] = torch.__version__
        snap["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            snap["gpu_name"] = props.name
            snap["gpu_total_memory_gib"] = round(props.total_memory / 1024**3, 3)
    except Exception as exc:  # noqa: BLE001
        snap["torch_error"] = repr(exc)
    return snap


def download_model(repo: str, include: str, out_dir: Path, progress: Path, label: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    append(progress, f"{label}_download_start", repo=repo, include=include)
    res = run(["hf", "download", repo, "--local-dir", str(out_dir), "--include", include], timeout=3600)
    append(progress, f"{label}_download_done", returncode=res.get("returncode"), elapsed_sec=res.get("elapsed_sec"))
    if res.get("returncode") != 0:
        raise RuntimeError(f"{label} download failed: {res.get('stderr') or res.get('stdout')}")
    ggufs = sorted(out_dir.rglob("*.gguf"))
    if not ggufs:
        raise RuntimeError(f"{label} download produced no GGUF under {out_dir}")
    return ggufs[0]


def parse_metrics(text: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for name, pattern in {
        "eval_ms_per_token": r"eval time\s*=\s*[\d.]+ ms\s*/\s*(\d+) runs\s*\(\s*([\d.]+) ms per token,\s*([\d.]+) tokens per second\)",
        "prompt_eval_ms_per_token": r"prompt eval time\s*=\s*[\d.]+ ms\s*/\s*(\d+) tokens\s*\(\s*([\d.]+) ms per token,\s*([\d.]+) tokens per second\)",
        "diffusion_total": r"total time:\s*([\d.]+)ms,\s*time per step:\s*([\d.]+)ms\s*\((\d+) steps over (\d+) blocks",
        "diffusion_throughput": r"throughput:\s*([\d.]+) tok/s\s*\((\d+) tok in ([\d.]+)ms\),\s*in-step parallel\s*([\d.]+) tok/s",
    }.items():
        match = re.search(pattern, text)
        if not match:
            continue
        metrics[name] = match.groups()
    if "eval_ms_per_token" in metrics:
        runs, ms_per_token, tokens_per_second = metrics["eval_ms_per_token"]
        metrics["decode_tokens"] = int(runs)
        metrics["decode_ms_per_token"] = float(ms_per_token)
        metrics["decode_tokens_per_second"] = float(tokens_per_second)
    if "prompt_eval_ms_per_token" in metrics:
        tokens, ms_per_token, tokens_per_second = metrics["prompt_eval_ms_per_token"]
        metrics["prompt_tokens"] = int(tokens)
        metrics["prompt_ms_per_token"] = float(ms_per_token)
        metrics["prompt_tokens_per_second"] = float(tokens_per_second)
    if "diffusion_total" in metrics:
        total_ms, step_ms, steps, blocks = metrics["diffusion_total"]
        metrics["diffusion_total_ms"] = float(total_ms)
        metrics["diffusion_step_ms"] = float(step_ms)
        metrics["diffusion_steps"] = int(steps)
        metrics["diffusion_blocks"] = int(blocks)
    if "diffusion_throughput" in metrics:
        tok_s, toks, total_ms, in_step_tok_s = metrics["diffusion_throughput"]
        metrics["diffusion_canvas_tokens_per_second"] = float(tok_s)
        metrics["diffusion_canvas_tokens"] = int(toks)
        metrics["diffusion_canvas_total_ms"] = float(total_ms)
        metrics["diffusion_in_step_parallel_tokens_per_second"] = float(in_step_tok_s)
    return metrics


def main() -> int:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = OUT_ROOT / f"{stamp}_speed_compare"
    out_dir.mkdir(parents=True, exist_ok=True)
    progress = out_dir / "progress.jsonl"
    meta: dict[str, Any] = {
        "prompt": PROMPT,
        "output_dir": str(out_dir),
        "comparison_note": "DiffusionGemma canvas-throughput is not identical to autoregressive Gemma 4 decode tok/s.",
    }
    append(progress, "start", **meta)

    try:
        meta["gpu"] = gpu_snapshot()
        append(progress, "gpu", gpu=meta["gpu"].get("gpu_name"), vram=meta["gpu"].get("gpu_total_memory_gib"))

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
        if meta["fetch_pr"].get("returncode") != 0:
            raise RuntimeError("fetch PR 24423 failed")

        append(progress, "checkout_start")
        meta["checkout"] = run("git checkout -B diffusiongemma FETCH_HEAD", timeout=120, cwd=LLAMA_DIR)
        append(progress, "checkout_done", returncode=meta["checkout"].get("returncode"))
        if meta["checkout"].get("returncode") != 0:
            raise RuntimeError("checkout failed")

        append(progress, "cmake_config_start")
        meta["cmake_config"] = run("cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release", timeout=600, cwd=LLAMA_DIR)
        append(progress, "cmake_config_done", returncode=meta["cmake_config"].get("returncode"))
        if meta["cmake_config"].get("returncode") != 0:
            raise RuntimeError("cmake config failed")

        append(progress, "build_start")
        meta["build"] = run("cmake --build build -j --config Release --target llama-cli llama-diffusion-cli", timeout=1800, cwd=LLAMA_DIR)
        append(progress, "build_done", returncode=meta["build"].get("returncode"), elapsed_sec=meta["build"].get("elapsed_sec"))
        if meta["build"].get("returncode") != 0:
            raise RuntimeError("build failed")

        diffusion_model = download_model(
            "unsloth/diffusiongemma-26B-A4B-it-GGUF",
            "*Q4_K_M*",
            MODEL_ROOT / "unsloth__diffusiongemma-26B-A4B-it-GGUF",
            progress,
            "diffusiongemma",
        )
        gemma_model = download_model(
            "unsloth/gemma-4-26B-A4B-it-GGUF",
            "*UD-Q4_K_M*",
            MODEL_ROOT / "unsloth__gemma-4-26B-A4B-it-GGUF",
            progress,
            "gemma4",
        )
        meta["models"] = {"diffusiongemma": str(diffusion_model), "gemma4": str(gemma_model)}

        diff_cmd = [
            str(LLAMA_DIR / "build/bin/llama-diffusion-cli"),
            "-m",
            str(diffusion_model),
            "-p",
            PROMPT,
            "-n",
            "96",
            "--temp",
            "0.2",
        ]
        append(progress, "diffusiongemma_inference_start", cmd=diff_cmd)
        meta["diffusiongemma_inference"] = run(diff_cmd, timeout=900, tail=80000)
        append(progress, "diffusiongemma_inference_done", returncode=meta["diffusiongemma_inference"].get("returncode"), elapsed_sec=meta["diffusiongemma_inference"].get("elapsed_sec"))

        gemma_cmd = [
            str(LLAMA_DIR / "build/bin/llama-cli"),
            "-m",
            str(gemma_model),
            "-p",
            PROMPT,
            "-n",
            "256",
            "--temp",
            "0.2",
            "-ngl",
            "99",
            "--no-warmup",
        ]
        append(progress, "gemma4_inference_start", cmd=gemma_cmd)
        meta["gemma4_inference"] = run(gemma_cmd, timeout=900, tail=80000)
        append(progress, "gemma4_inference_done", returncode=meta["gemma4_inference"].get("returncode"), elapsed_sec=meta["gemma4_inference"].get("elapsed_sec"))

        for label in ["diffusiongemma_inference", "gemma4_inference"]:
            res = meta[label]
            (out_dir / f"{label}.stdout.txt").write_text(res.get("stdout", ""), encoding="utf-8")
            (out_dir / f"{label}.stderr.txt").write_text(res.get("stderr", ""), encoding="utf-8")
            meta[f"{label}_metrics"] = parse_metrics((res.get("stdout", "") or "") + "\n" + (res.get("stderr", "") or ""))

        meta["status"] = "completed"
    except Exception as exc:  # noqa: BLE001
        meta["status"] = "failed"
        meta["error"] = repr(exc)
        append(progress, "failed", error=repr(exc))

    meta["completed_at"] = datetime.now(timezone.utc).isoformat()
    (out_dir / "benchmark_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = [
        "# DiffusionGemma vs Gemma 4 L4 Speed Compare",
        "",
        f"- Status: {meta['status']}",
        f"- GPU: {meta.get('gpu', {}).get('gpu_name', 'unknown')}",
        f"- Prompt: {PROMPT}",
        "- Caveat: DiffusionGemma canvas throughput is not the same metric as autoregressive Gemma 4 decode throughput.",
        "",
        "## Parsed Metrics",
        "",
        "```json",
        json.dumps(
            {
                "diffusiongemma": meta.get("diffusiongemma_inference_metrics"),
                "gemma4": meta.get("gemma4_inference_metrics"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        "```",
    ]
    (out_dir / "speed_compare_report.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"RESULT_DIR={out_dir}", flush=True)
    print(f"RESULT_STATUS={meta['status']}", flush=True)
    return 0 if meta["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
