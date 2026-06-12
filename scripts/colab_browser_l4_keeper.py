#!/usr/bin/env python3
"""Open a browser-owned Colab L4 runtime and keep the Chrome surface alive.

This helper intentionally creates the runtime through the Colab browser UI.
It does not run `colab new`. After it sees an L4 assignment through the
Colab CLI server-side assignment list, use `adopt_browser_colab_assignment.py`
with a temporary config file.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.parse
from pathlib import Path
from typing import Any

import requests
import websocket


CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
DEFAULT_PROFILE = str(Path.home() / ".cache" / "diffusiongemma-colab-cli" / "chrome-colab-cdp")


class CDP:
    def __init__(self, ws_url: str):
        self.ws = websocket.create_connection(ws_url, timeout=30)
        self.next_id = 1

    def call(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        msg_id = self.next_id
        self.next_id += 1
        self.ws.send(json.dumps({"id": msg_id, "method": method, "params": params or {}}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("id") == msg_id:
                if "error" in msg:
                    raise RuntimeError(f"{method} failed: {msg['error']}")
                return msg.get("result", {})

    def eval(self, expression: str) -> Any:
        result = self.call(
            "Runtime.evaluate",
            {"expression": expression, "returnByValue": True, "awaitPromise": True},
        )
        return result.get("result", {}).get("value")

    def click(self, x: int, y: int) -> None:
        for event_type in ("mousePressed", "mouseReleased"):
            self.call(
                "Input.dispatchMouseEvent",
                {
                    "type": event_type,
                    "x": x,
                    "y": y,
                    "button": "left",
                    "clickCount": 1,
                },
            )
            time.sleep(0.08)

    def screenshot(self, path: Path) -> None:
        data = self.call("Page.captureScreenshot", {"format": "png", "fromSurface": True})["data"]
        import base64

        path.write_bytes(base64.b64decode(data))


def wait_http(url: str, timeout: float = 30) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            requests.get(url, timeout=2)
            return
        except requests.RequestException:
            time.sleep(0.5)
    raise TimeoutError(f"Chrome CDP did not become ready: {url}")


def browser_text(cdp: CDP) -> str:
    value = cdp.eval("document.body ? document.body.innerText : ''")
    return value or ""


def click_text(cdp: CDP, text: str) -> bool:
    script = f"""
(() => {{
  const needle = {json.dumps(text)};
  const nodes = [...document.querySelectorAll('button, div[role="button"], span, a')];
  const n = nodes.find(el => (el.innerText || el.textContent || '').trim().includes(needle));
  if (!n) return false;
  n.scrollIntoView({{block: 'center', inline: 'center'}});
  const r = n.getBoundingClientRect();
  const x = Math.round(r.left + r.width / 2);
  const y = Math.round(r.top + r.height / 2);
  return {{x, y, text: (n.innerText || n.textContent || '').trim().slice(0, 80)}};
}})()
"""
    point = cdp.eval(script)
    if not point:
        return False
    cdp.click(int(point["x"]), int(point["y"]))
    return True


def close_gemini_panel(cdp: CDP) -> None:
    script = """
(() => {
  const candidates = [...document.querySelectorAll('div[role="button"], button')];
  const close = candidates.find(el => {
    const label = (el.getAttribute('aria-label') || el.innerText || el.textContent || '').trim();
    const r = el.getBoundingClientRect();
    return (label === 'Close' || label === '閉じる' || label === 'close') && r.left > window.innerWidth * 0.7;
  });
  if (!close) return false;
  const r = close.getBoundingClientRect();
  return {x: Math.round(r.left + r.width / 2), y: Math.round(r.top + r.height / 2), label: close.getAttribute('aria-label') || close.innerText || close.textContent};
})()
"""
    point = cdp.eval(script)
    if point:
        cdp.click(int(point["x"]), int(point["y"]))
        time.sleep(1.0)


def list_l4_assignments() -> list[str]:
    from colab_cli.common import state

    endpoints: list[str] = []
    for assignment in state.client.list_assignments():
        if assignment.accelerator.value.upper() == "L4":
            endpoints.append(assignment.endpoint)
    return endpoints


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=9224)
    parser.add_argument("--profile-dir", default=DEFAULT_PROFILE)
    parser.add_argument("--screenshot", default="/tmp/colab-browser-l4-keeper.png")
    parser.add_argument("--status-json", default="/tmp/colab-browser-l4-keeper.json")
    parser.add_argument("--ready-timeout", type=int, default=420)
    parser.add_argument("--keepalive", action="store_true")
    args = parser.parse_args()

    profile_dir = Path(args.profile_dir)
    profile_dir.mkdir(parents=True, exist_ok=True)
    url = "https://colab.research.google.com/#create=true"
    chrome_cmd = [
        CHROME,
        "--headless=new",
        "--disable-gpu",
        f"--user-data-dir={profile_dir}",
        "--profile-directory=Profile 1",
        f"--remote-debugging-port={args.port}",
        "--remote-allow-origins=*",
        "--window-size=1280,720",
        "--no-first-run",
        "--no-default-browser-check",
        url,
    ]
    env = os.environ.copy()
    env.setdefault("DISPLAY", "")
    proc = subprocess.Popen(chrome_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
    status_path = Path(args.status_json)

    try:
        wait_http(f"http://127.0.0.1:{args.port}/json/version")
        create_url = f"http://127.0.0.1:{args.port}/json/new?{urllib.parse.quote(url, safe='')}"
        try:
            requests.put(create_url, timeout=5)
        except requests.RequestException:
            pass
        targets = requests.get(f"http://127.0.0.1:{args.port}/json/list", timeout=5).json()
        colab_pages = [
            t
            for t in targets
            if t.get("type") == "page" and "colab.research.google.com" in t.get("url", "")
        ]
        if not colab_pages:
            raise RuntimeError(f"No top-level Colab page target found: {targets!r}")
        target = colab_pages[-1]
        cdp = CDP(target["webSocketDebuggerUrl"])
        cdp.call("Page.enable")
        cdp.call("Runtime.enable")
        cdp.call("Emulation.setDeviceMetricsOverride", {"width": 1280, "height": 720, "deviceScaleFactor": 1, "mobile": False})

        deadline = time.time() + args.ready_timeout
        connected_text_seen = False
        l4_endpoints: list[str] = []
        phase = "load"
        while time.time() < deadline:
            text = browser_text(cdp)
            if "Gemini" in text and ("ご用件をお聞かせください" in text or "Maki さん" in text):
                cdp.click(1240, 139)
                time.sleep(1.0)
                close_gemini_panel(cdp)
                text = browser_text(cdp)

            if "ログイン" in text and not ("接続" in text or "ランタイム" in text):
                click_text(cdp, "ログイン")
                phase = "login-clicked"
                time.sleep(8)
                continue

            if phase in {"load", "login-clicked"} and ("ランタイム" in text or "Runtime" in text):
                cdp.click(310, 50)
                time.sleep(1.2)
                cdp.click(400, 400)
                time.sleep(2.0)
                phase = "runtime-dialog"
                continue

            if phase == "runtime-dialog":
                # Existing verified Colab coordinates for the L4 radio and Save button
                # on a 1280x720 desktop viewport.
                cdp.click(424, 340)
                time.sleep(0.8)
                cdp.click(879, 606)
                time.sleep(3.0)
                phase = "l4-saved"
                continue

            if phase == "l4-saved":
                cdp.click(700, 84)
                time.sleep(5.0)
                phase = "connect-clicked"
                continue

            if "接続先" in text or "Connected" in text or "RAM" in text:
                connected_text_seen = True

            l4_endpoints = list_l4_assignments()
            if connected_text_seen and l4_endpoints:
                phase = "ready"
                break
            time.sleep(3)

        cdp.screenshot(Path(args.screenshot))
        status = {
            "phase": phase,
            "connected_text_seen": connected_text_seen,
            "l4_endpoints": l4_endpoints,
            "screenshot": args.screenshot,
            "pid": proc.pid,
            "port": args.port,
        }
        status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(status, ensure_ascii=False), flush=True)
        if phase != "ready":
            return 2

        if args.keepalive:
            while proc.poll() is None:
                try:
                    _ = browser_text(cdp)
                    print("KEEPER_BEAT", time.strftime("%Y-%m-%dT%H:%M:%S%z"), flush=True)
                except Exception as exc:  # noqa: BLE001
                    print(f"KEEPER_ERROR {exc!r}", flush=True)
                    break
                time.sleep(30)
        return 0
    finally:
        if not args.keepalive and proc.poll() is None:
            proc.terminate()


if __name__ == "__main__":
    raise SystemExit(main())
