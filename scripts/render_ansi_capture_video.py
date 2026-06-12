#!/usr/bin/env python3
"""Render a captured ANSI terminal transcript into MP4 frames.

The input is a raw terminal capture produced by
run_diffusiongemma_gguf_visual_capture.py. This script does not invent token
states; it replays the captured terminal control stream through pyte and renders
the resulting terminal screen snapshots.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path

import pyte
from PIL import Image, ImageDraw, ImageFont


SUNWOOD = {
    "felt": (7, 63, 56),
    "felt_deep": (4, 31, 27),
    "felt_dark": (5, 38, 33),
    "cream": (255, 242, 212),
    "cream_dim": (245, 223, 180),
    "gold": (242, 182, 67),
    "thread_red": (217, 31, 47),
    "thread_green": (22, 130, 77),
    "thread_blue": (55, 183, 200),
}
FG = {
    "default": SUNWOOD["cream"],
    "black": SUNWOOD["felt_deep"],
    "red": SUNWOOD["thread_red"],
    "green": SUNWOOD["thread_green"],
    "yellow": SUNWOOD["gold"],
    "blue": SUNWOOD["thread_blue"],
    "magenta": (184, 140, 255),
    "cyan": SUNWOOD["thread_blue"],
    "white": SUNWOOD["cream"],
}
BG = SUNWOOD["felt_deep"]
STEP_RE = re.compile(r"diffusion step:\s+(\d+)/(\d+)")


def find_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc",
        "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
        "/System/Library/Fonts/ヒラギノ丸ゴ ProN W4.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/Monaco.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def screen_text(screen: pyte.Screen) -> str:
    return "\n".join(screen.display)


def cover_resize(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    target_w, target_h = size
    scale = max(target_w / image.width, target_h / image.height)
    new_size = (round(image.width * scale), round(image.height * scale))
    resized = image.resize(new_size, Image.Resampling.LANCZOS)
    left = (resized.width - target_w) // 2
    top = (resized.height - target_h) // 2
    return resized.crop((left, top, left + target_w, top + target_h))


def make_canvas(size: tuple[int, int], background: Image.Image | None) -> Image.Image:
    solid = Image.new("RGB", size, BG)
    if background is None:
        return solid
    bg = cover_resize(background.convert("RGB"), size)
    return Image.blend(solid, bg, 0.30)


def render_screen(screen: pyte.Screen, out: Path, title: str, font: ImageFont.ImageFont, canvas: Image.Image) -> None:
    margin_x = 42
    margin_y = 88
    cell_w = 11
    cell_h = 22
    width = 1280
    height = 720
    image = canvas.copy()
    draw = ImageDraw.Draw(image)

    draw.rectangle((0, 0, width, 58), fill=SUNWOOD["felt"])
    draw.rectangle((0, 56, width, 60), fill=SUNWOOD["gold"])
    draw.rectangle((0, 60, width, 64), fill=SUNWOOD["thread_red"])
    draw.text((26, 18), "Sunwood AI Labs / " + title, fill=SUNWOOD["cream"], font=font)
    draw.text((width - 520, 18), "llama-diffusion-cli --diffusion-visual", fill=SUNWOOD["cream_dim"], font=font)
    draw.rectangle((42, 672, 494, 708), fill=SUNWOOD["felt"])
    draw.rectangle((42, 672, 494, 708), outline=SUNWOOD["gold"], width=2)
    draw.text((58, 681), "Runner: llama-diffusion-cli", fill=SUNWOOD["gold"], font=font)

    for y, line in enumerate(screen.display):
        text = line.rstrip().replace("\ufffd", "…")
        if not text:
            continue
        color = SUNWOOD["cream"]
        if "diffusion step:" in text:
            color = SUNWOOD["gold"]
        elif text.lstrip().startswith("Draft:") or "Draft:" in text:
            color = SUNWOOD["cream_dim"]
        elif "total time:" in text or "throughput:" in text:
            color = SUNWOOD["thread_blue"]
        draw.text((margin_x, margin_y + y * cell_h), text, fill=color, font=font)

    image.save(out)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("ansi", type=Path)
    parser.add_argument("out_mp4", type=Path)
    parser.add_argument("--frames-dir", type=Path, default=None)
    parser.add_argument("--ffmpeg", type=Path, default=Path("videos/l4-quant-generation-flow/node_modules/ffmpeg-static/ffmpeg"))
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--background-image", type=Path, default=None)
    args = parser.parse_args()

    frames_dir = args.frames_dir or args.out_mp4.with_suffix("").parent / (args.out_mp4.stem + "_frames")
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    args.out_mp4.parent.mkdir(parents=True, exist_ok=True)

    raw = args.ansi.read_bytes().decode("utf-8", "replace")
    screen = pyte.Screen(110, 23)
    stream = pyte.Stream(screen)
    font = find_font(18)
    background = Image.open(args.background_image) if args.background_image else None
    canvas = make_canvas((1280, 720), background)

    frame_index = 0
    last_step = None
    last_text = ""

    def save_frame(repeat: int = 1) -> None:
        nonlocal frame_index
        title = "llama-diffusion-cli x DiffusionGemma L4 Q4_K_M"
        for _ in range(repeat):
            render_screen(screen, frames_dir / f"frame_{frame_index:05d}.png", title, font, canvas)
            frame_index += 1

    # Pre-roll.
    save_frame(8)
    for line in raw.splitlines(keepends=True):
        stream.feed(line)
        text = screen_text(screen)
        match = STEP_RE.search(text)
        step = int(match.group(1)) if match else None
        if step is not None and step != last_step:
            save_frame(8)
            last_step = step
            last_text = text
            continue
        if "total time:" in text and text != last_text:
            save_frame(24)
            last_text = text

    save_frame(30)
    cmd = [
        str(args.ffmpeg),
        "-y",
        "-framerate",
        str(args.fps),
        "-i",
        str(frames_dir / "frame_%05d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(args.out_mp4),
    ]
    subprocess.check_call(cmd)
    print(f"frames={frame_index}")
    print(f"mp4={args.out_mp4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
