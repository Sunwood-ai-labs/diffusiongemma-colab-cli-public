#!/usr/bin/env python3
"""Find a DiffusionGemma configuration that feels fast on Colab L4.

This is not a quality benchmark. It intentionally targets the model's visible
strength: a single 256-token canvas with fewer denoising steps, then compares
that generation-phase latency with regular Gemma 4.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


OUT_ROOT = Path("/content/diffusiongemma-colab-cli-public")
LLAMA_DIR = Path("/content/llama.cpp")
MODEL_ROOT = Path("/content/models")
PR_REF = "pull/24423/head"
N_PREDICT = "256"
PROMPT = (
    "In one compact but vivid technical field note, describe a Mars cafe robot "
    "barista calibrating espresso pressure, grind size, and crema quality under "
    "0.38g gravity. Keep it under 190 words and make it feel immediately useful."
)
DIFFUSION_STEPS = [8, 12, 16, 24, 48]


def run(cmd: str | list[str], timeout: int = 120, cwd: Path | None = None, tail: int = 200000) -> dict[str, Any]:
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


def parse_diffusion(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if m := re.search(r"total time:\s*([\d.]+)ms,\s*time per step:\s*([\d.]+)ms\s*\((\d+) steps over (\d+) blocks", text):
        out["total_ms"] = float(m.group(1))
        out["step_ms"] = float(m.group(2))
        out["steps"] = int(m.group(3))
        out["blocks"] = int(m.group(4))
    if m := re.search(r"throughput:\s*([\d.]+) tok/s\s*\((\d+) tok in ([\d.]+)ms\),\s*in-step parallel\s*([\d.]+) tok/s", text):
        out["tokens_per_second"] = float(m.group(1))
        out["tokens"] = int(m.group(2))
        out["token_ms"] = float(m.group(3))
        out["in_step_parallel_tokens_per_second"] = float(m.group(4))
    return out


def parse_gemma(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if m := re.search(r"Prompt:\s*([\d.]+) t/s\s*\|\s*Generation:\s*([\d.]+) t/s", text):
        out["prompt_tokens_per_second"] = float(m.group(1))
        out["generation_tokens_per_second"] = float(m.group(2))
    if m := re.search(r"eval time\s*=\s*[\d.]+ ms\s*/\s*(\d+) runs\s*\(\s*([\d.]+) ms per token,\s*([\d.]+) tokens per second\)", text):
        out["decode_tokens"] = int(m.group(1))
        out["decode_ms_per_token"] = float(m.group(2))
        out["decode_tokens_per_second"] = float(m.group(3))
    return out


def output_word_count(text: str) -> int:
    cleaned = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", text)
    return len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", cleaned))


def download_model(repo: str, include: str, out_dir: Path, progress: Path, label: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ggufs = sorted(out_dir.rglob("*.gguf"))
    if not ggufs:
        append(progress, f"{label}_download_start", repo=repo, include=include)
        result = run(["hf", "download", repo, "--local-dir", str(out_dir), "--include", include], timeout=3600)
        append(progress, f"{label}_download_done", returncode=result.get("returncode"), elapsed_sec=result.get("elapsed_sec"))
        if result.get("returncode") != 0:
            raise RuntimeError(f"{label} download failed")
        ggufs = sorted(out_dir.rglob("*.gguf"))
    if not ggufs:
        raise RuntimeError(f"{label} GGUF not found under {out_dir}")
    return ggufs[0]


def main() -> int:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = OUT_ROOT / f"{stamp}_l4_fast_feel_diffusiongemma"
    out_dir.mkdir(parents=True, exist_ok=True)
    progress = out_dir / "progress.jsonl"
    meta: dict[str, Any] = {
        "route": "cli-created-l4-diagnostic",
        "intent": "fast-feel DiffusionGemma experiment, not a quality benchmark",
        "prompt": PROMPT,
        "n_predict": int(N_PREDICT),
        "diffusion_step_sweep": DIFFUSION_STEPS,
        "output_dir": str(out_dir),
    }
    append(progress, "start", **meta)

    try:
        meta["nvidia_smi"] = run("nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader", timeout=30)
        append(progress, "gpu", output=meta["nvidia_smi"].get("stdout", "").strip())

        append(progress, "pip_install_hf_start")
        meta["pip_install_hf"] = run([sys.executable, "-m", "pip", "install", "-U", "huggingface_hub[cli]"], timeout=300)
        append(progress, "pip_install_hf_done", returncode=meta["pip_install_hf"].get("returncode"))

        if not LLAMA_DIR.exists():
            append(progress, "clone_start")
            meta["clone"] = run("git clone --depth 1 https://github.com/ggml-org/llama.cpp /content/llama.cpp", timeout=600)
            append(progress, "clone_done", returncode=meta["clone"].get("returncode"))
            if meta["clone"].get("returncode") != 0:
                raise RuntimeError("clone failed")

        append(progress, "checkout_start")
        meta["fetch_pr"] = run(f"git fetch origin {PR_REF}", timeout=600, cwd=LLAMA_DIR)
        meta["checkout"] = run("git checkout -B diffusiongemma FETCH_HEAD", timeout=120, cwd=LLAMA_DIR)
        meta["git_head"] = run("git rev-parse HEAD && git log -1 --oneline", timeout=30, cwd=LLAMA_DIR)
        append(progress, "checkout_done", git_head=meta["git_head"].get("stdout", "").strip())

        need_build = not all((LLAMA_DIR / p).exists() for p in ["build/bin/llama-diffusion-cli", "build/bin/llama-cli"])
        if need_build:
            append(progress, "build_start")
            meta["cmake_config"] = run("cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release", timeout=600, cwd=LLAMA_DIR)
            meta["build"] = run("cmake --build build -j --config Release --target llama-diffusion-cli llama-cli", timeout=1800, cwd=LLAMA_DIR)
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

        diffusion_results = []
        for max_steps in DIFFUSION_STEPS:
            label = f"diffusiongemma_eb{max_steps}"
            cmd = [
                str(LLAMA_DIR / "build/bin/llama-diffusion-cli"),
                "-m", str(diffusion_model),
                "-p", PROMPT,
                "-n", N_PREDICT,
                "--temp", "0.2",
                "-ngl", "99",
                "-cnv",
                "--diffusion-eb-max-steps", str(max_steps),
            ]
            append(progress, f"{label}_start", cmd=cmd)
            result = run(cmd, timeout=900)
            append(progress, f"{label}_done", returncode=result.get("returncode"), elapsed_sec=result.get("elapsed_sec"), error=result.get("error"))
            (out_dir / f"{label}.stdout.txt").write_text(result.get("stdout", ""), encoding="utf-8")
            (out_dir / f"{label}.stderr.txt").write_text(result.get("stderr", ""), encoding="utf-8")
            metrics = parse_diffusion((result.get("stdout") or "") + "\n" + (result.get("stderr") or ""))
            item = {
                "label": label,
                "max_steps": max_steps,
                "returncode": result.get("returncode"),
                "elapsed_sec": result.get("elapsed_sec"),
                "metrics": metrics,
                "output_words": output_word_count(result.get("stdout") or ""),
                "cmd": cmd,
            }
            diffusion_results.append(item)
            meta[label] = item
            if result.get("returncode") != 0:
                raise RuntimeError(f"{label} failed")

        gemma_cmd = [
            str(LLAMA_DIR / "build/bin/llama-cli"),
            "-m", str(gemma_model),
            "-p", PROMPT,
            "-n", N_PREDICT,
            "--temp", "0.2",
            "-ngl", "99",
            "-cnv",
            "-st",
        ]
        append(progress, "gemma4_start", cmd=gemma_cmd)
        gemma = run(gemma_cmd, timeout=900)
        append(progress, "gemma4_done", returncode=gemma.get("returncode"), elapsed_sec=gemma.get("elapsed_sec"), error=gemma.get("error"))
        (out_dir / "gemma4.stdout.txt").write_text(gemma.get("stdout", ""), encoding="utf-8")
        (out_dir / "gemma4.stderr.txt").write_text(gemma.get("stderr", ""), encoding="utf-8")
        meta["gemma4"] = {
            "returncode": gemma.get("returncode"),
            "elapsed_sec": gemma.get("elapsed_sec"),
            "metrics": parse_gemma((gemma.get("stdout") or "") + "\n" + (gemma.get("stderr") or "")),
            "output_words": output_word_count(gemma.get("stdout") or ""),
            "cmd": gemma_cmd,
        }
        if gemma.get("returncode") != 0:
            raise RuntimeError("gemma4 failed")

        best = max(
            diffusion_results,
            key=lambda x: x.get("metrics", {}).get("tokens_per_second", 0.0),
        )
        gemma_tps = meta["gemma4"]["metrics"].get("generation_tokens_per_second") or meta["gemma4"]["metrics"].get("decode_tokens_per_second")
        meta["summary"] = {
            "best_diffusion_label": best["label"],
            "best_diffusion_max_steps": best["max_steps"],
            "best_diffusion_tokens_per_second": best.get("metrics", {}).get("tokens_per_second"),
            "best_diffusion_generation_ms": best.get("metrics", {}).get("total_ms"),
            "gemma4_generation_tokens_per_second": gemma_tps,
            "speedup_vs_gemma4_generation_tps": (
                best.get("metrics", {}).get("tokens_per_second") / gemma_tps
                if gemma_tps else None
            ),
        }
        meta["status"] = "completed"
    except Exception as exc:  # noqa: BLE001
        meta["status"] = "failed"
        meta["error"] = repr(exc)
        append(progress, "failed", error=repr(exc))

    meta["completed_at"] = datetime.now(timezone.utc).isoformat()
    (out_dir / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rows = {
        "status": meta.get("status"),
        "n_predict": meta.get("n_predict"),
        "prompt": meta.get("prompt"),
        "git_head": meta.get("git_head", {}).get("stdout", "").strip(),
        "summary": meta.get("summary"),
        "diffusion": [
            meta.get(f"diffusiongemma_eb{s}") for s in DIFFUSION_STEPS if meta.get(f"diffusiongemma_eb{s}")
        ],
        "gemma4": meta.get("gemma4"),
    }
    (out_dir / "run_report.md").write_text(
        "# DiffusionGemma Fast-Feel L4 Experiment\n\n```json\n"
        + json.dumps(rows, ensure_ascii=False, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    print(f"RESULT_DIR={out_dir}", flush=True)
    print(f"RESULT_STATUS={meta.get('status')}", flush=True)
    return 0 if meta.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
