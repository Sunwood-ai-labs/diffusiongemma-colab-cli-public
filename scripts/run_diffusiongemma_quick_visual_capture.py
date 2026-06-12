#!/usr/bin/env python3
"""Colab-side quick visual capture using an already built llama-diffusion-cli."""

from __future__ import annotations

import json
import os
import pty
import select
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROMPT = os.environ.get(
    "QUICK_PROMPT",
    "Write a concise 3-bullet lab note about a robot barista calibrating espresso shots on a tiny Mars cafe.",
)
SESSION_LABEL = os.environ.get("QUICK_SESSION_LABEL", "diffusiongemma-l4-quick-visual")
OUT_ROOT = Path("/content/diffusiongemma-colab-cli-public")
LLAMA_DIR = Path("/content/llama.cpp")
MODEL_DIR = Path("/content/models/unsloth__diffusiongemma-26B-A4B-it-GGUF")


def append(path: Path, event: str, **fields: Any) -> None:
    row = {"ts": datetime.now(timezone.utc).isoformat(), "event": event, **fields}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(row, ensure_ascii=False), flush=True)


def run_pty(cmd: list[str], out_path: Path, timeout: int = 900) -> dict[str, Any]:
    started = time.time()
    master_fd, slave_fd = pty.openpty()
    env = os.environ.copy()
    env.setdefault("TERM", "xterm-256color")
    proc = subprocess.Popen(
        cmd,
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
                    break
        finally:
            os.close(master_fd)
    if proc.poll() is None:
        proc.terminate()
        returncode = proc.wait(timeout=5)
    else:
        returncode = proc.returncode
    return {
        "returncode": returncode,
        "timed_out": timed_out,
        "elapsed_sec": round(time.time() - started, 3),
        "ansi_bytes": len(raw),
        "text_tail": raw.decode("utf-8", "replace")[-20000:],
    }


def main() -> int:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = OUT_ROOT / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    progress = out_dir / "progress.jsonl"
    binary = LLAMA_DIR / "build/bin/llama-diffusion-cli"
    ggufs = sorted(MODEL_DIR.rglob("*.gguf"))
    if not binary.exists():
        raise FileNotFoundError(f"missing binary: {binary}")
    if not ggufs:
        raise FileNotFoundError(f"missing GGUF under: {MODEL_DIR}")

    model_path = ggufs[0]
    visual_log = out_dir / "visual_terminal_capture.ansi"
    llama_log = out_dir / "llama_diffusion_visual.log"
    cmd = [
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
    meta: dict[str, Any] = {
        "session_label": SESSION_LABEL,
        "model_repo": "unsloth/diffusiongemma-26B-A4B-it-GGUF",
        "gguf_include": "*Q4_K_M*",
        "model_path": str(model_path),
        "prompt": PROMPT,
        "output_dir": str(out_dir),
        "runner": "llama-diffusion-cli",
        "capture_mode": "pty_ansi_terminal_visual_diffusion",
        "inference_cmd": cmd,
    }
    append(progress, "start", **meta)
    append(progress, "inference_visual_start", ansi_log=str(visual_log), llama_log=str(llama_log))
    meta["inference"] = run_pty(cmd, visual_log)
    (out_dir / "generation_terminal_tail.txt").write_text(meta["inference"]["text_tail"], encoding="utf-8")
    append(
        progress,
        "inference_visual_done",
        returncode=meta["inference"]["returncode"],
        elapsed_sec=meta["inference"]["elapsed_sec"],
        ansi_bytes=meta["inference"]["ansi_bytes"],
    )
    meta["status"] = "generated_visual_capture" if meta["inference"]["returncode"] == 0 else "inference_failed"
    meta["completed_at"] = datetime.now(timezone.utc).isoformat()
    (out_dir / "experiment_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "result_summary.md").write_text(
        "\n".join(
            [
                "# DiffusionGemma Quick Visual Capture Result",
                "",
                f"- Status: {meta['status']}",
                "- Runner: llama-diffusion-cli",
                "- Hardware: Colab L4 session",
                "- Model repo: unsloth/diffusiongemma-26B-A4B-it-GGUF",
                "- Quant: Q4_K_M",
                f"- Prompt: {PROMPT}",
                "- Capture mode: PTY ANSI terminal output with --diffusion-visual",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"RESULT_DIR={out_dir}", flush=True)
    print(f"RESULT_STATUS={meta['status']}", flush=True)
    return 0 if meta["status"] == "generated_visual_capture" else 1


if __name__ == "__main__":
    raise SystemExit(main())
