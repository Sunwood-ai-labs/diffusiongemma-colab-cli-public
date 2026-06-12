#!/usr/bin/env python3
"""Colab-side DiffusionGemma GGUF visual capture probe.

This is the evidence-first rerun path: it executes llama-diffusion-cli in a
pseudo-terminal with visual diffusion mode enabled, streams the terminal output,
and saves the raw ANSI terminal transcript without truncating the inference
surface.
"""

from __future__ import annotations

import json
import os
import pty
import select
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MODEL_REPO = os.environ.get("MODEL_REPO", "unsloth/diffusiongemma-26B-A4B-it-GGUF")
GGUF_INCLUDE = os.environ.get("GGUF_INCLUDE", "*Q4_K_M*")
SESSION_LABEL = os.environ.get("SESSION_LABEL", "diffusiongemma-l4-gguf-visual")
PROMPT = os.environ.get(
    "PROMPT",
    "日本語で、DiffusionGemmaの特徴を3点だけ簡潔に説明してください。",
)
OUT_ROOT = Path("/content/diffusiongemma-colab-cli-public")


def run(cmd: str | list[str], timeout: int = 120, cwd: Path | None = None, tail: int = 12000) -> dict[str, Any]:
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


def run_pty(cmd: list[str], out_path: Path, timeout: int = 900, cwd: Path | None = None) -> dict[str, Any]:
    started = time.time()
    master_fd, slave_fd = pty.openpty()
    env = os.environ.copy()
    env.setdefault("TERM", "xterm-256color")
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdin=subprocess.DEVNULL,
        stdout=slave_fd,
        stderr=slave_fd,
        text=False,
        env=env,
        close_fds=True,
    )
    os.close(slave_fd)
    raw = bytearray()
    timed_out = False
    with out_path.open("wb") as handle:
        try:
            while True:
                if time.time() - started > timeout:
                    timed_out = True
                    proc.kill()
                    break
                ready, _, _ = select.select([master_fd], [], [], 0.2)
                if ready:
                    try:
                        chunk = os.read(master_fd, 8192)
                    except OSError:
                        break
                    if not chunk:
                        break
                    raw.extend(chunk)
                    handle.write(chunk)
                    handle.flush()
                    sys.stdout.write(chunk.decode("utf-8", "replace"))
                    sys.stdout.flush()
                if proc.poll() is not None:
                    while True:
                        ready, _, _ = select.select([master_fd], [], [], 0)
                        if not ready:
                            break
                        try:
                            chunk = os.read(master_fd, 8192)
                        except OSError:
                            break
                        if not chunk:
                            break
                        raw.extend(chunk)
                        handle.write(chunk)
                        sys.stdout.write(chunk.decode("utf-8", "replace"))
                        sys.stdout.flush()
                    break
        finally:
            os.close(master_fd)
    if proc.poll() is None:
        try:
            returncode = proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                returncode = proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                returncode = proc.wait(timeout=5)
    else:
        returncode = proc.returncode

    return {
        "cmd": cmd,
        "returncode": returncode,
        "timed_out": timed_out,
        "elapsed_sec": round(time.time() - started, 3),
        "ansi_log": str(out_path),
        "ansi_bytes": len(raw),
        "text_tail": raw.decode("utf-8", "replace")[-20000:],
    }


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
        "capture_mode": "pty_ansi_terminal_visual_diffusion",
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
            res = run(cmd, timeout=timeout, tail=20000)
            meta[name] = res
            append(progress, f"{name}_done", returncode=res.get("returncode"), elapsed_sec=res.get("elapsed_sec"))
            if res.get("returncode") != 0:
                raise RuntimeError(f"{name} failed")

        llama_dir = Path("/content/llama.cpp")
        if not llama_dir.exists():
            append(progress, "clone_start")
            meta["clone"] = run("git clone --depth 1 https://github.com/ggml-org/llama.cpp /content/llama.cpp", timeout=600, tail=20000)
            append(progress, "clone_done", returncode=meta["clone"].get("returncode"))
            if meta["clone"].get("returncode") != 0:
                raise RuntimeError("clone failed")

        append(progress, "fetch_pr_start")
        meta["fetch_pr"] = run("git fetch origin pull/24423/head", timeout=600, cwd=llama_dir, tail=20000)
        append(progress, "fetch_pr_done", returncode=meta["fetch_pr"].get("returncode"))
        if meta["fetch_pr"].get("returncode") != 0:
            raise RuntimeError("fetch PR 24423 failed")

        append(progress, "checkout_start")
        meta["checkout"] = run("git checkout -B diffusiongemma FETCH_HEAD", timeout=120, cwd=llama_dir, tail=20000)
        append(progress, "checkout_done", returncode=meta["checkout"].get("returncode"))
        if meta["checkout"].get("returncode") != 0:
            raise RuntimeError("checkout failed")

        append(progress, "cmake_config_start")
        meta["cmake_config"] = run("cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release", timeout=600, cwd=llama_dir, tail=20000)
        append(progress, "cmake_config_done", returncode=meta["cmake_config"].get("returncode"))
        if meta["cmake_config"].get("returncode") != 0:
            raise RuntimeError("cmake config failed")

        append(progress, "build_start")
        meta["build"] = run("cmake --build build -j --config Release --target llama-diffusion-cli", timeout=1800, cwd=llama_dir, tail=30000)
        append(progress, "build_done", returncode=meta["build"].get("returncode"), elapsed_sec=meta["build"].get("elapsed_sec"))
        if meta["build"].get("returncode") != 0:
            raise RuntimeError("build failed")

        model_dir = Path("/content/models") / MODEL_REPO.replace("/", "__")
        model_dir.mkdir(parents=True, exist_ok=True)
        append(progress, "download_start", repo=MODEL_REPO, include=GGUF_INCLUDE)
        meta["download"] = run(
            ["hf", "download", MODEL_REPO, "--local-dir", str(model_dir), "--include", GGUF_INCLUDE],
            timeout=2400,
            tail=30000,
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
        meta["help"] = run([str(binary), "--help"], timeout=60, tail=30000)
        visual_log = out_dir / "visual_terminal_capture.ansi"
        llama_log = out_dir / "llama_diffusion_visual.log"
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
            "--diffusion-visual",
            "--diffusion-visual-progress",
            "--diffusion-visual-interval",
            "1",
            "--log-file",
            str(llama_log),
        ]
        meta["inference_cmd"] = infer_cmd
        append(progress, "inference_visual_start", model=str(model_path), ansi_log=str(visual_log), llama_log=str(llama_log))
        meta["inference"] = run_pty(infer_cmd, visual_log, timeout=900)
        (out_dir / "generation_terminal_tail.txt").write_text(meta["inference"].get("text_tail", ""), encoding="utf-8")
        append(
            progress,
            "inference_visual_done",
            returncode=meta["inference"].get("returncode"),
            elapsed_sec=meta["inference"].get("elapsed_sec"),
            ansi_bytes=meta["inference"].get("ansi_bytes"),
        )
        meta["status"] = "generated_visual_capture" if meta["inference"].get("returncode") == 0 else "inference_failed"
    except Exception as exc:  # noqa: BLE001
        meta["status"] = "failed"
        meta["error"] = repr(exc)
        append(progress, "failed", error=repr(exc))

    meta["completed_at"] = datetime.now(timezone.utc).isoformat()
    write_json(out_dir / "experiment_metadata.json", meta)
    summary = [
        "# DiffusionGemma GGUF Visual Capture Result",
        "",
        f"- Status: `{meta['status']}`",
        f"- Model repo: `{MODEL_REPO}`",
        f"- Include: `{GGUF_INCLUDE}`",
        f"- GPU: `{meta.get('gpu', {}).get('gpu_name', 'unknown')}`",
        f"- GPU memory GiB: `{meta.get('gpu', {}).get('gpu_total_memory_gib', 'unknown')}`",
        f"- Output dir: `{out_dir}`",
        "- Capture mode: `PTY ANSI terminal output with --diffusion-visual`",
    ]
    (out_dir / "result_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"RESULT_DIR={out_dir}", flush=True)
    print(f"RESULT_STATUS={meta['status']}", flush=True)
    return 0 if meta["status"] == "generated_visual_capture" else 1


if __name__ == "__main__":
    raise SystemExit(main())
