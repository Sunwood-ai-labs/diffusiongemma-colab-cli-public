#!/usr/bin/env python3
"""Colab L4 250+ token comparison for DiffusionGemma vs regular Gemma 4.

This script is meant to answer the upstream review comment that short
generations do not exercise DiffusionGemma's 256-token canvas fairly.
"""

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
N_PREDICT = "320"
PROMPT = (
    "Write a detailed technical lab note of at least 280 English words about a "
    "robot barista calibrating espresso shots in a tiny Mars cafe. Include "
    "sections for environment, calibration procedure, observations, failure "
    "modes, and conclusion. Keep the answer continuous and do not stop early."
)


def run(
    cmd: str | list[str],
    timeout: int = 120,
    cwd: Path | None = None,
    tail: int = 120000,
) -> dict[str, Any]:
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
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", "replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", "replace")
        return {
            "cmd": cmd,
            "returncode": None,
            "error": "TimeoutExpired",
            "stdout": stdout[-tail:],
            "stderr": stderr[-tail:],
            "elapsed_sec": round(time.time() - started, 3),
        }


def append(path: Path, event: str, **fields: Any) -> None:
    row = {"ts": datetime.now(timezone.utc).isoformat(), "event": event, **fields}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(row, ensure_ascii=False), flush=True)


def parse_metrics(text: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    match = re.search(
        r"prompt eval time\s*=\s*[\d.]+ ms\s*/\s*(\d+) tokens\s*"
        r"\(\s*([\d.]+) ms per token,\s*([\d.]+) tokens per second\)",
        text,
    )
    if match:
        metrics["prompt_tokens"] = int(match.group(1))
        metrics["prompt_ms_per_token"] = float(match.group(2))
        metrics["prompt_tokens_per_second"] = float(match.group(3))
    match = re.search(
        r"eval time\s*=\s*[\d.]+ ms\s*/\s*(\d+) runs\s*"
        r"\(\s*([\d.]+) ms per token,\s*([\d.]+) tokens per second\)",
        text,
    )
    if match:
        metrics["decode_tokens"] = int(match.group(1))
        metrics["decode_ms_per_token"] = float(match.group(2))
        metrics["decode_tokens_per_second"] = float(match.group(3))
    match = re.search(r"total time\s*=\s*([\d.]+) ms\s*/\s*(\d+) tokens", text)
    if match:
        metrics["ar_total_ms"] = float(match.group(1))
        metrics["ar_total_tokens"] = int(match.group(2))

    match = re.search(
        r"total time:\s*([\d.]+)ms,\s*time per step:\s*([\d.]+)ms\s*"
        r"\((\d+) steps over (\d+) blocks",
        text,
    )
    if match:
        metrics["diffusion_total_ms"] = float(match.group(1))
        metrics["diffusion_step_ms"] = float(match.group(2))
        metrics["diffusion_steps"] = int(match.group(3))
        metrics["diffusion_blocks"] = int(match.group(4))
    match = re.search(
        r"throughput:\s*([\d.]+) tok/s\s*\((\d+) tok in ([\d.]+)ms\),\s*"
        r"in-step parallel\s*([\d.]+) tok/s",
        text,
    )
    if match:
        metrics["diffusion_canvas_tokens_per_second"] = float(match.group(1))
        metrics["diffusion_canvas_tokens"] = int(match.group(2))
        metrics["diffusion_canvas_total_ms"] = float(match.group(3))
        metrics["diffusion_in_step_parallel_tokens_per_second"] = float(match.group(4))
    return metrics


def count_output_words(stdout: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", stdout))


def download_model(repo: str, include: str, out_dir: Path, progress: Path, label: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ggufs = sorted(out_dir.rglob("*.gguf"))
    if not ggufs:
        append(progress, f"{label}_download_start", repo=repo, include=include)
        result = run(
            ["hf", "download", repo, "--local-dir", str(out_dir), "--include", include],
            timeout=3600,
        )
        append(
            progress,
            f"{label}_download_done",
            returncode=result.get("returncode"),
            elapsed_sec=result.get("elapsed_sec"),
        )
        if result.get("returncode") != 0:
            raise RuntimeError(f"{label} download failed: {result.get('stderr') or result.get('stdout')}")
        ggufs = sorted(out_dir.rglob("*.gguf"))
    if not ggufs:
        raise RuntimeError(f"{label} GGUF not found under {out_dir}")
    return ggufs[0]


def main() -> int:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = OUT_ROOT / f"{stamp}_l4_250_token_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    progress = out_dir / "progress.jsonl"
    meta: dict[str, Any] = {
        "prompt": PROMPT,
        "n_predict": int(N_PREDICT),
        "output_dir": str(out_dir),
        "route": "active-colab-cli-l4-session",
        "comparison_note": (
            "Both commands request 320 generated tokens. DiffusionGemma canvas "
            "throughput and regular Gemma 4 autoregressive decode tok/s are still "
            "not identical metrics."
        ),
    }
    append(progress, "start", **meta)

    try:
        meta["nvidia_smi"] = run(
            "nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader",
            timeout=30,
        )
        append(progress, "gpu", output=meta["nvidia_smi"].get("stdout", "").strip())

        for name, cmd, timeout in [
            ("pip_install_hf", f"{shlex.quote(sys.executable)} -m pip install -U 'huggingface_hub[cli]'", 300),
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

        append(progress, "checkout_start")
        meta["fetch_pr"] = run("git fetch origin pull/24423/head", timeout=600, cwd=LLAMA_DIR)
        meta["checkout"] = run("git checkout -B diffusiongemma FETCH_HEAD", timeout=120, cwd=LLAMA_DIR)
        meta["git_head"] = run("git rev-parse HEAD && git log -1 --oneline", timeout=30, cwd=LLAMA_DIR)
        append(
            progress,
            "checkout_done",
            fetch_returncode=meta["fetch_pr"].get("returncode"),
            checkout_returncode=meta["checkout"].get("returncode"),
            git_head=meta["git_head"].get("stdout", "").strip(),
        )
        if meta["fetch_pr"].get("returncode") != 0 or meta["checkout"].get("returncode") != 0:
            raise RuntimeError("checkout failed")

        need_build = not (LLAMA_DIR / "build/bin/llama-diffusion-cli").exists() or not (
            LLAMA_DIR / "build/bin/llama-completion"
        ).exists()
        if need_build:
            append(progress, "build_start")
            meta["cmake_config"] = run("cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release", timeout=600, cwd=LLAMA_DIR)
            meta["build"] = run(
                "cmake --build build -j --config Release --target llama-diffusion-cli llama-completion",
                timeout=1800,
                cwd=LLAMA_DIR,
            )
            append(progress, "build_done", returncode=meta["build"].get("returncode"), elapsed_sec=meta["build"].get("elapsed_sec"))
            if meta["cmake_config"].get("returncode") != 0 or meta["build"].get("returncode") != 0:
                raise RuntimeError("build failed")
        else:
            append(progress, "build_reuse")

        diffusion_model = download_model(
            "unsloth/diffusiongemma-26B-A4B-it-GGUF",
            "*Q4_K_M*",
            MODEL_ROOT / "unsloth__diffusiongemma-26B-A4B-it-GGUF",
            progress,
            "diffusiongemma",
        )
        gemma_model = download_model(
            "ggml-org/gemma-4-26B-A4B-it-GGUF",
            "*Q4_K_M*",
            MODEL_ROOT / "ggml-org__gemma-4-26B-A4B-it-GGUF",
            progress,
            "gemma4",
        )
        meta["models"] = {"diffusiongemma": str(diffusion_model), "gemma4": str(gemma_model)}

        commands = {
            "diffusiongemma": [
                str(LLAMA_DIR / "build/bin/llama-diffusion-cli"),
                "-m",
                str(diffusion_model),
                "-p",
                PROMPT,
                "-n",
                N_PREDICT,
                "--temp",
                "0.2",
            ],
            "gemma4": [
                str(LLAMA_DIR / "build/bin/llama-completion"),
                "-m",
                str(gemma_model),
                "-p",
                PROMPT,
                "-n",
                N_PREDICT,
                "--temp",
                "0.2",
                "-ngl",
                "99",
                "--no-warmup",
                "--jinja",
                "-no-cnv",
            ],
        }

        for label, cmd in commands.items():
            append(progress, f"{label}_inference_start", cmd=cmd)
            result = run(cmd, timeout=1200, tail=200000)
            meta[f"{label}_inference"] = result
            append(
                progress,
                f"{label}_inference_done",
                returncode=result.get("returncode"),
                elapsed_sec=result.get("elapsed_sec"),
                error=result.get("error"),
            )
            (out_dir / f"{label}.stdout.txt").write_text(result.get("stdout", ""), encoding="utf-8")
            (out_dir / f"{label}.stderr.txt").write_text(result.get("stderr", ""), encoding="utf-8")
            meta[f"{label}_metrics"] = parse_metrics((result.get("stdout") or "") + "\n" + (result.get("stderr") or ""))
            meta[f"{label}_output_words"] = count_output_words(result.get("stdout") or "")
            if result.get("returncode") != 0:
                raise RuntimeError(f"{label} inference failed")

        meta["status"] = "completed"
    except Exception as exc:  # noqa: BLE001
        meta["status"] = "failed"
        meta["error"] = repr(exc)
        append(progress, "failed", error=repr(exc))

    meta["completed_at"] = datetime.now(timezone.utc).isoformat()
    (out_dir / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = [
        "# L4 250+ Token Comparison",
        "",
        f"- Status: {meta['status']}",
        f"- N predict: {N_PREDICT}",
        f"- Prompt: {PROMPT}",
        f"- Git head: {meta.get('git_head', {}).get('stdout', '').strip()}",
        "",
        "## Metrics",
        "",
        "```json",
        json.dumps(
            {
                "diffusiongemma": meta.get("diffusiongemma_metrics"),
                "gemma4": meta.get("gemma4_metrics"),
                "output_words": {
                    "diffusiongemma": meta.get("diffusiongemma_output_words"),
                    "gemma4": meta.get("gemma4_output_words"),
                },
                "elapsed_sec": {
                    "diffusiongemma": meta.get("diffusiongemma_inference", {}).get("elapsed_sec"),
                    "gemma4": meta.get("gemma4_inference", {}).get("elapsed_sec"),
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        "```",
    ]
    (out_dir / "run_report.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"RESULT_DIR={out_dir}", flush=True)
    print(f"RESULT_STATUS={meta['status']}", flush=True)
    return 0 if meta["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
