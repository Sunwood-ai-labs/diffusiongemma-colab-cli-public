#!/usr/bin/env python3
"""Colab-side DiffusionGemma GGUF probe for L4/T4 class GPUs.

Targets the Unsloth Q4_K_M GGUF route using the llama.cpp DiffusionGemma PR
runner. This is intentionally separate from the Transformers probe because the
full model needs >60 GiB VRAM while Q4_K_M is intended for 24 GiB GPUs.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MODEL_REPO = os.environ.get("MODEL_REPO", "unsloth/diffusiongemma-26B-A4B-it-GGUF")
GGUF_INCLUDE = os.environ.get("GGUF_INCLUDE", "*Q4_K_M*")
SESSION_LABEL = os.environ.get("SESSION_LABEL", "diffusiongemma-l4-gguf-q4")
PROMPT = os.environ.get(
    "PROMPT",
    "日本語で、DiffusionGemmaの特徴を3点だけ簡潔に説明してください。",
)
OUT_ROOT = Path("/content/diffusiongemma-colab-cli-public")


def run(cmd: str | list[str], timeout: int = 120, cwd: Path | None = None) -> dict[str, Any]:
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
            "stdout": proc.stdout[-12000:],
            "stderr": proc.stderr[-12000:],
            "elapsed_sec": round(time.time() - started, 3),
        }
    except Exception as exc:  # noqa: BLE001
        return {"cmd": cmd, "error": repr(exc), "elapsed_sec": round(time.time() - started, 3)}


def append(path: Path, event: str, **fields: Any) -> None:
    row = {"ts": datetime.now(timezone.utc).isoformat(), "event": event, **fields}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(row, ensure_ascii=False), flush=True)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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


def main() -> int:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = OUT_ROOT / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    progress = out_dir / "progress.jsonl"
    meta: dict[str, Any] = {
        "session_label": SESSION_LABEL,
        "model_repo": MODEL_REPO,
        "gguf_include": GGUF_INCLUDE,
        "prompt": PROMPT,
        "output_dir": str(out_dir),
        "drive_mounted": Path("/content/drive/MyDrive").exists(),
    }
    append(progress, "start", **meta)

    try:
        meta["gpu"] = gpu_snapshot()
        append(progress, "gpu", gpu=meta["gpu"].get("gpu_name"), vram=meta["gpu"].get("gpu_total_memory_gib"))

        steps = [
            ("apt_update", "apt-get update -y", 240),
            ("apt_install", "apt-get install -y cmake ninja-build git git-lfs build-essential", 360),
            ("pip_install", f"{shlex.quote(sys.executable)} -m pip install -U 'huggingface_hub[cli]'", 300),
        ]
        for name, cmd, timeout in steps:
            append(progress, f"{name}_start")
            res = run(cmd, timeout=timeout)
            meta[name] = res
            append(progress, f"{name}_done", returncode=res.get("returncode"), elapsed_sec=res.get("elapsed_sec"))
            if res.get("returncode") != 0:
                raise RuntimeError(f"{name} failed")

        llama_dir = Path("/content/llama.cpp")
        if not llama_dir.exists():
            append(progress, "clone_start")
            meta["clone"] = run("git clone --depth 1 https://github.com/ggml-org/llama.cpp /content/llama.cpp", timeout=600)
            append(progress, "clone_done", returncode=meta["clone"].get("returncode"))
            if meta["clone"].get("returncode") != 0:
                raise RuntimeError("clone failed")

        append(progress, "fetch_pr_start")
        meta["fetch_pr"] = run("git fetch origin pull/24423/head:diffusiongemma", timeout=600, cwd=llama_dir)
        append(progress, "fetch_pr_done", returncode=meta["fetch_pr"].get("returncode"))
        if meta["fetch_pr"].get("returncode") != 0:
            raise RuntimeError("fetch PR 24423 failed")

        append(progress, "checkout_start")
        meta["checkout"] = run("git checkout diffusiongemma", timeout=120, cwd=llama_dir)
        append(progress, "checkout_done", returncode=meta["checkout"].get("returncode"))
        if meta["checkout"].get("returncode") != 0:
            raise RuntimeError("checkout failed")

        append(progress, "cmake_config_start")
        meta["cmake_config"] = run("cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release", timeout=600, cwd=llama_dir)
        append(progress, "cmake_config_done", returncode=meta["cmake_config"].get("returncode"))
        if meta["cmake_config"].get("returncode") != 0:
            raise RuntimeError("cmake config failed")

        append(progress, "build_start")
        meta["build"] = run("cmake --build build -j --config Release --target llama-diffusion-cli", timeout=1800, cwd=llama_dir)
        append(progress, "build_done", returncode=meta["build"].get("returncode"), elapsed_sec=meta["build"].get("elapsed_sec"))
        if meta["build"].get("returncode") != 0:
            raise RuntimeError("build failed")

        model_dir = Path("/content/models") / MODEL_REPO.replace("/", "__")
        model_dir.mkdir(parents=True, exist_ok=True)
        append(progress, "download_start", repo=MODEL_REPO, include=GGUF_INCLUDE)
        meta["download"] = run(
            [
                "hf",
                "download",
                MODEL_REPO,
                "--local-dir",
                str(model_dir),
                "--include",
                GGUF_INCLUDE,
            ],
            timeout=2400,
        )
        append(progress, "download_done", returncode=meta["download"].get("returncode"), elapsed_sec=meta["download"].get("elapsed_sec"))
        if meta["download"].get("returncode") != 0:
            raise RuntimeError("hf download failed")

        ggufs = sorted(model_dir.rglob("*.gguf"))
        meta["gguf_files"] = [str(p) for p in ggufs]
        if not ggufs:
            raise RuntimeError("no GGUF file downloaded")
        model_path = ggufs[0]

        binary = llama_dir / "build/bin/llama-diffusion-cli"
        meta["help"] = run([str(binary), "--help"], timeout=60)
        append(progress, "inference_start", model=str(model_path))
        infer_cmd = [
            str(binary),
            "-m",
            str(model_path),
            "-p",
            PROMPT,
            "-n",
            "96",
            "--temp",
            "0.2",
        ]
        meta["inference_cmd"] = infer_cmd
        meta["inference"] = run(infer_cmd, timeout=900)
        (out_dir / "generation_stdout.txt").write_text(meta["inference"].get("stdout", ""), encoding="utf-8")
        (out_dir / "generation_stderr.txt").write_text(meta["inference"].get("stderr", ""), encoding="utf-8")
        append(progress, "inference_done", returncode=meta["inference"].get("returncode"), elapsed_sec=meta["inference"].get("elapsed_sec"))
        meta["status"] = "generated" if meta["inference"].get("returncode") == 0 else "inference_failed"
    except Exception as exc:  # noqa: BLE001
        meta["status"] = "failed"
        meta["error"] = repr(exc)
        append(progress, "failed", error=repr(exc))

    meta["completed_at"] = datetime.now(timezone.utc).isoformat()
    write_json(out_dir / "experiment_metadata.json", meta)
    summary = [
        "# DiffusionGemma GGUF L4/T4 Result",
        "",
        f"- Status: `{meta['status']}`",
        f"- Model repo: `{MODEL_REPO}`",
        f"- Include: `{GGUF_INCLUDE}`",
        f"- GPU: `{meta.get('gpu', {}).get('gpu_name', 'unknown')}`",
        f"- GPU memory GiB: `{meta.get('gpu', {}).get('gpu_total_memory_gib', 'unknown')}`",
        f"- Output dir: `{out_dir}`",
    ]
    (out_dir / "result_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"RESULT_DIR={out_dir}", flush=True)
    print(f"RESULT_STATUS={meta['status']}", flush=True)
    return 0 if meta["status"] == "generated" else 1


if __name__ == "__main__":
    raise SystemExit(main())
