#!/usr/bin/env python3
"""Diagnose why regular Gemma 4 GGUF did not finish on Colab L4."""

from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


OUT_ROOT = Path("/content/diffusiongemma-colab-cli-public")
LLAMA_DIR = Path("/content/llama.cpp")
MODEL_ROOT = Path("/content/models")
MODEL_REPO = "ggml-org/gemma-4-26B-A4B-it-GGUF"
MODEL_INCLUDE = "*Q4_K_M*"
PROMPT = "Write one short sentence about espresso on Mars."


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
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", "replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", "replace")
        return {
            "cmd": cmd,
            "error": "TimeoutExpired",
            "timeout": timeout,
            "stdout": stdout[-tail:],
            "stderr": stderr[-tail:],
            "elapsed_sec": round(time.time() - started, 3),
        }
    except Exception as exc:  # noqa: BLE001
        return {"cmd": cmd, "error": repr(exc), "elapsed_sec": round(time.time() - started, 3)}


def append(path: Path, event: str, **fields: Any) -> None:
    row = {"ts": datetime.now(timezone.utc).isoformat(), "event": event, **fields}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(row, ensure_ascii=False), flush=True)


def popen_capture(cmd: list[str], out_dir: Path, label: str, timeout: int) -> dict[str, Any]:
    stdout_path = out_dir / f"{label}.stdout.txt"
    stderr_path = out_dir / f"{label}.stderr.txt"
    monitor_path = out_dir / f"{label}.nvidia_smi.jsonl"
    started = time.time()
    with stdout_path.open("wb") as stdout_handle, stderr_path.open("wb") as stderr_handle:
        proc = subprocess.Popen(
            cmd,
            stdout=stdout_handle,
            stderr=stderr_handle,
            start_new_session=True,
            env={**os.environ, "TERM": "xterm-256color"},
        )
        timed_out = False
        samples: list[dict[str, Any]] = []
        while True:
            elapsed = time.time() - started
            if int(elapsed) % 10 == 0:
                smi = run(
                    "nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw --format=csv,noheader,nounits || true",
                    timeout=10,
                    tail=4000,
                )
                sample = {"elapsed_sec": round(elapsed, 1), "smi": smi.get("stdout", "").strip()}
                samples.append(sample)
                with monitor_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
                time.sleep(1.0)
            if proc.poll() is not None:
                break
            if elapsed >= timeout:
                timed_out = True
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                    proc.wait(timeout=10)
                except Exception:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except Exception:
                        pass
                    proc.wait(timeout=10)
                break
            time.sleep(0.25)
    stdout_text = stdout_path.read_text(encoding="utf-8", errors="replace")
    stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "timed_out": timed_out,
        "elapsed_sec": round(time.time() - started, 3),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "monitor_path": str(monitor_path),
        "stdout_tail": stdout_text[-20000:],
        "stderr_tail": stderr_text[-20000:],
        "stdout_bytes": len(stdout_text.encode("utf-8")),
        "stderr_bytes": len(stderr_text.encode("utf-8")),
        "monitor_samples": samples[-20:],
    }


def main() -> int:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = OUT_ROOT / f"{stamp}_gemma4_timeout_diagnosis"
    out_dir.mkdir(parents=True, exist_ok=True)
    progress = out_dir / "progress.jsonl"
    meta: dict[str, Any] = {"output_dir": str(out_dir), "prompt": PROMPT}
    append(progress, "start", **meta)
    try:
        meta["preflight"] = {
            "nvidia_smi": run("nvidia-smi || true", timeout=30),
            "free": run("free -h || true", timeout=30),
            "df": run("df -h /content || true", timeout=30),
        }
        append(progress, "preflight_done")

        for name, cmd, timeout in [
            ("apt_update", "apt-get update -y", 240),
            ("apt_install", "apt-get install -y cmake ninja-build git git-lfs build-essential coreutils", 360),
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

        append(progress, "checkout_pr_start")
        meta["fetch_pr"] = run("git fetch origin pull/24423/head", timeout=600, cwd=LLAMA_DIR)
        meta["checkout"] = run("git checkout -B diffusiongemma FETCH_HEAD", timeout=120, cwd=LLAMA_DIR)
        meta["git_head"] = run("git rev-parse HEAD && git log -1 --oneline", timeout=30, cwd=LLAMA_DIR)
        append(progress, "checkout_pr_done", fetch_returncode=meta["fetch_pr"].get("returncode"), checkout_returncode=meta["checkout"].get("returncode"))

        append(progress, "build_start")
        meta["cmake_config"] = run("cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release", timeout=600, cwd=LLAMA_DIR)
        meta["build"] = run("cmake --build build -j --config Release --target llama-cli", timeout=1800, cwd=LLAMA_DIR)
        append(progress, "build_done", returncode=meta["build"].get("returncode"), elapsed_sec=meta["build"].get("elapsed_sec"))
        if meta["build"].get("returncode") != 0:
            raise RuntimeError("build failed")

        model_dir = MODEL_ROOT / MODEL_REPO.replace("/", "__")
        model_dir.mkdir(parents=True, exist_ok=True)
        append(progress, "download_start", repo=MODEL_REPO, include=MODEL_INCLUDE)
        meta["download"] = run(["hf", "download", MODEL_REPO, "--local-dir", str(model_dir), "--include", MODEL_INCLUDE], timeout=3600)
        append(progress, "download_done", returncode=meta["download"].get("returncode"), elapsed_sec=meta["download"].get("elapsed_sec"))
        if meta["download"].get("returncode") != 0:
            raise RuntimeError("download failed")
        ggufs = sorted(model_dir.rglob("*.gguf"))
        if not ggufs:
            raise RuntimeError("no GGUF downloaded")
        model = ggufs[0]
        meta["model"] = str(model)
        meta["model_stat"] = run(f"ls -lh {shlex.quote(str(model))} && du -h {shlex.quote(str(model))}", timeout=30)

        binary = LLAMA_DIR / "build/bin/llama-cli"
        common = [str(binary), "-m", str(model), "-p", PROMPT, "-n", "1", "--temp", "0.2", "--no-warmup", "--log-verbosity", "5"]

        tests = [
            ("ngl_99_n1", [*common, "-ngl", "99"], 180),
            ("ngl_0_n1", [*common, "-ngl", "0"], 180),
        ]
        results = {}
        for label, cmd, timeout in tests:
            append(progress, "inference_probe_start", label=label, timeout=timeout, cmd=cmd)
            results[label] = popen_capture(cmd, out_dir, label, timeout)
            append(
                progress,
                "inference_probe_done",
                label=label,
                returncode=results[label].get("returncode"),
                timed_out=results[label].get("timed_out"),
                elapsed_sec=results[label].get("elapsed_sec"),
                stdout_bytes=results[label].get("stdout_bytes"),
                stderr_bytes=results[label].get("stderr_bytes"),
            )
        meta["probe_results"] = results
        meta["status"] = "completed"
    except Exception as exc:  # noqa: BLE001
        meta["status"] = "failed"
        meta["error"] = repr(exc)
        append(progress, "failed", error=repr(exc))

    meta["completed_at"] = datetime.now(timezone.utc).isoformat()
    (out_dir / "diagnosis_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Gemma 4 Timeout Diagnosis",
        "",
        f"- Status: {meta.get('status')}",
        f"- Model: {meta.get('model')}",
        f"- Output dir: {out_dir}",
        "",
        "## Probe Results",
        "",
    ]
    for label, result in meta.get("probe_results", {}).items():
        lines.extend(
            [
                f"### {label}",
                "",
                f"- timed_out: {result.get('timed_out')}",
                f"- returncode: {result.get('returncode')}",
                f"- elapsed_sec: {result.get('elapsed_sec')}",
                f"- stdout_bytes: {result.get('stdout_bytes')}",
                f"- stderr_bytes: {result.get('stderr_bytes')}",
                f"- stdout: {Path(result.get('stdout_path')).name}",
                f"- stderr: {Path(result.get('stderr_path')).name}",
                f"- monitor: {Path(result.get('monitor_path')).name}",
                "",
            ]
        )
    (out_dir / "diagnosis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"RESULT_DIR={out_dir}", flush=True)
    print(f"RESULT_STATUS={meta.get('status')}", flush=True)
    return 0 if meta.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
