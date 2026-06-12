#!/usr/bin/env python3
"""Colab-side DiffusionGemma probe.

This script is intentionally conservative: it verifies environment, Drive
output, model metadata, and GPU memory before attempting to load the model.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MODEL_ID = "google/diffusiongemma-26B-A4B-it"
DRIVE_ROOT = Path("/content/drive/MyDrive/diffusiongemma-colab-cli-public")
MIN_VRAM_GIB_FOR_LOAD = 60.0
PROMPT = "日本語で、DiffusionGemmaの特徴を3点だけ簡潔に説明してください。"


def run(cmd: list[str], timeout: int = 120) -> dict[str, Any]:
    started = time.time()
    try:
        proc = subprocess.run(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        return {
            "cmd": cmd,
            "returncode": proc.returncode,
            "stdout": proc.stdout[-8000:],
            "stderr": proc.stderr[-8000:],
            "elapsed_sec": round(time.time() - started, 3),
        }
    except Exception as exc:  # noqa: BLE001 - serialize diagnostic evidence
        return {
            "cmd": cmd,
            "error": repr(exc),
            "elapsed_sec": round(time.time() - started, 3),
        }


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_progress(path: Path, event: str, **fields: Any) -> None:
    row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **fields,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(row, ensure_ascii=False), flush=True)


def ensure_deps(progress_path: Path) -> None:
    append_progress(progress_path, "install_start")
    result = run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-U",
            "transformers",
            "accelerate",
            "huggingface_hub",
            "psutil",
        ],
        timeout=900,
    )
    append_progress(progress_path, "install_done", returncode=result.get("returncode"))
    if result.get("returncode") != 0:
        raise RuntimeError(f"pip install failed: {result}")


def gpu_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {"nvidia_smi": run(["bash", "-lc", "nvidia-smi || true"])}
    try:
        import torch

        snapshot["torch_version"] = torch.__version__
        snapshot["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            snapshot["gpu_name"] = props.name
            snapshot["gpu_total_memory_gib"] = round(props.total_memory / 1024**3, 3)
    except Exception as exc:  # noqa: BLE001
        snapshot["torch_error"] = repr(exc)
    return snapshot


def drive_mount_snapshot() -> dict[str, Any]:
    return {
        "mount_grep": run(["bash", "-lc", "mount | grep /content/drive || true"]),
        "drive_exists": Path("/content/drive").exists(),
        "mydrive_exists": Path("/content/drive/MyDrive").exists(),
    }


def output_dir() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if Path("/content/drive/MyDrive").exists():
        base = DRIVE_ROOT
    else:
        base = Path("/content/diffusiongemma-colab-cli-public")
    target = base / timestamp
    target.mkdir(parents=True, exist_ok=True)
    return target


def hf_model_info(progress_path: Path) -> dict[str, Any]:
    append_progress(progress_path, "hf_model_info_start", model_id=MODEL_ID)
    from huggingface_hub import model_info

    info = model_info(MODEL_ID)
    payload = {
        "model_id": MODEL_ID,
        "sha": info.sha,
        "last_modified": str(info.last_modified),
        "license": (info.card_data or {}).get("license") if isinstance(info.card_data, dict) else None,
        "tags": info.tags,
        "siblings_count": len(info.siblings or []),
    }
    append_progress(progress_path, "hf_model_info_done", sha=payload["sha"])
    return payload


def try_generation(out_dir: Path, progress_path: Path, gpu: dict[str, Any]) -> dict[str, Any]:
    if not Path("/content/drive/MyDrive").exists() and os.environ.get("ALLOW_NO_DRIVE_MODEL_LOAD") != "1":
        return {
            "status": "blocked_drive_not_mounted",
            "reason": "Drive is not mounted. Skipping long model load so the run does not rely on ephemeral /content outputs.",
        }

    total_gib = float(gpu.get("gpu_total_memory_gib") or 0.0)
    if total_gib < MIN_VRAM_GIB_FOR_LOAD:
        return {
            "status": "blocked_insufficient_vram",
            "reason": f"GPU memory {total_gib:.3f} GiB is below {MIN_VRAM_GIB_FOR_LOAD:.1f} GiB load threshold.",
        }

    append_progress(progress_path, "model_load_start", model_id=MODEL_ID)
    import torch
    from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion

    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        MODEL_ID,
        dtype="auto",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    append_progress(progress_path, "model_load_done")

    messages = [{"role": "user", "content": PROMPT}]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    append_progress(progress_path, "generation_start", max_new_tokens=128)
    started = time.time()
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=128)
    elapsed = time.time() - started
    text = processor.decode(output[0], skip_special_tokens=True)
    (out_dir / "generation.txt").write_text(text, encoding="utf-8")
    append_progress(progress_path, "generation_done", elapsed_sec=round(elapsed, 3))
    return {"status": "generated", "elapsed_sec": round(elapsed, 3), "output_chars": len(text)}


def main() -> int:
    out_dir = output_dir()
    progress_path = out_dir / "progress.jsonl"
    append_progress(progress_path, "start", output_dir=str(out_dir), model_id=MODEL_ID)

    metadata: dict[str, Any] = {
        "model_id": MODEL_ID,
        "prompt": PROMPT,
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "working_directory": os.getcwd(),
        "output_dir": str(out_dir),
        "colab": {
            "content_exists": Path("/content").exists(),
            "drive_snapshot_before": drive_mount_snapshot(),
        },
    }

    try:
        ensure_deps(progress_path)
        metadata["gpu"] = gpu_snapshot()
        metadata["drive_snapshot_after_deps"] = drive_mount_snapshot()
        metadata["hf_model_info"] = hf_model_info(progress_path)
        metadata["generation_attempt"] = try_generation(out_dir, progress_path, metadata["gpu"])
        status = metadata["generation_attempt"]["status"]
    except Exception as exc:  # noqa: BLE001
        status = "failed"
        metadata["error"] = repr(exc)
        append_progress(progress_path, "failed", error=repr(exc))

    metadata["status"] = status
    metadata["completed_at"] = datetime.now(timezone.utc).isoformat()
    write_json(out_dir / "experiment_metadata.json", metadata)

    summary = [
        "# DiffusionGemma Colab CLI Result",
        "",
        f"- Status: `{status}`",
        f"- Model: `{MODEL_ID}`",
        f"- Output dir: `{out_dir}`",
        f"- GPU: `{metadata.get('gpu', {}).get('gpu_name', 'unknown')}`",
        f"- GPU memory GiB: `{metadata.get('gpu', {}).get('gpu_total_memory_gib', 'unknown')}`",
        f"- Drive mounted: `{bool(metadata.get('drive_snapshot_after_deps', {}).get('mydrive_exists'))}`",
        "",
        "See `experiment_metadata.json` and `progress.jsonl` for raw evidence.",
    ]
    (out_dir / "result_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    metadata["files"] = sorted(p.name for p in out_dir.iterdir())
    write_json(out_dir / "experiment_metadata.json", metadata)
    append_progress(progress_path, "done", status=status, files=metadata["files"])
    print(f"RESULT_DIR={out_dir}", flush=True)
    print(f"RESULT_STATUS={status}", flush=True)
    return 0 if status in {"generated", "blocked_insufficient_vram", "blocked_drive_not_mounted"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
