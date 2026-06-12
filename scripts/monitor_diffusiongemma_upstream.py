#!/usr/bin/env python3
"""Collect upstream DiffusionGemma performance discussion snapshots.

This intentionally does not run model inference. It records GitHub-side issue/PR
state so speed/offload regressions around llama-diffusion-cli can be checked
twice a day without relying on chat history.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = "ggml-org/llama.cpp"
PRS = [24423, 24427]
KEYWORDS = [
    "DiffusionGemma",
    "diffusiongemma",
    "llama-diffusion-cli",
    "throughput",
    "tok/s",
    "tokens per second",
    "time per step",
    "slow",
    "slower",
    "performance",
    "L4",
    "offload",
    "-ngl",
    "gpu",
]


def run_gh(args: list[str]) -> Any:
    proc = subprocess.run(
        ["gh", *args],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"gh {' '.join(args)} failed with {proc.returncode}: {proc.stderr.strip()}"
        )
    text = proc.stdout.strip()
    return json.loads(text) if text else None


def gh_text(args: list[str]) -> str:
    proc = subprocess.run(
        ["gh", *args],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"gh {' '.join(args)} failed with {proc.returncode}: {proc.stderr.strip()}"
        )
    return proc.stdout


def compact_body(body: str, limit: int = 1600) -> str:
    body = re.sub(r"\r\n?", "\n", body or "")
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body[:limit] + ("..." if len(body) > limit else "")


def is_relevant(body: str) -> bool:
    low = body.lower()
    return any(keyword.lower() in low for keyword in KEYWORDS)


def search_existing() -> list[dict[str, Any]]:
    query = "DiffusionGemma speed performance slow llama-diffusion-cli"
    return run_gh(
        [
            "search",
            "issues",
            query,
            "--repo",
            REPO,
            "--state",
            "open",
            "--include-prs",
            "--limit",
            "30",
            "--json",
            "number,title,url,state,isPullRequest,updatedAt,commentsCount",
        ]
    )


def collect_pr(number: int) -> dict[str, Any]:
    view = run_gh(
        [
            "pr",
            "view",
            str(number),
            "-R",
            REPO,
            "--json",
            "title,state,isDraft,url,headRefName,updatedAt,comments",
        ]
    )
    comments = run_gh(
        [
            "api",
            f"repos/{REPO}/issues/{number}/comments",
            "--paginate",
        ]
    )
    relevant_comments = []
    for comment in comments or []:
        body = comment.get("body", "")
        if is_relevant(body):
            relevant_comments.append(
                {
                    "user": comment.get("user", {}).get("login"),
                    "created_at": comment.get("created_at"),
                    "updated_at": comment.get("updated_at"),
                    "url": comment.get("html_url"),
                    "body_excerpt": compact_body(body),
                }
            )
    return {
        "number": number,
        "view": view,
        "relevant_comments": relevant_comments,
    }


def write_outputs(snapshot: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "snapshot.json").write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# DiffusionGemma upstream watch",
        "",
        f"- Collected at: {snapshot['collected_at']}",
        f"- Repository: {REPO}",
        "",
        "## Open search results",
    ]
    for item in snapshot["open_search_results"]:
        kind = "PR" if item.get("isPullRequest") else "issue"
        lines.append(
            f"- {kind} #{item['number']}: {item['title']} ({item['state']}) - {item['url']}"
        )
    lines.extend(["", "## Relevant PR comments"])
    for pr in snapshot["prs"]:
        view = pr["view"]
        lines.extend(
            [
                "",
                f"### PR #{pr['number']}: {view['title']}",
                "",
                f"- State: {view['state']}",
                f"- Draft: {view['isDraft']}",
                f"- Updated: {view['updatedAt']}",
                f"- URL: {view['url']}",
                "",
            ]
        )
        for comment in pr["relevant_comments"]:
            excerpt = comment["body_excerpt"].replace("\n", " ")
            lines.append(
                f"- {comment['created_at']} @{comment['user']}: {excerpt} ({comment['url']})"
            )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = repo_root / "monitoring" / "upstream" / stamp
    snapshot = {
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "repo": REPO,
        "keywords": KEYWORDS,
        "open_search_results": search_existing(),
        "prs": [collect_pr(number) for number in PRS],
    }
    write_outputs(snapshot, out_dir)

    latest = repo_root / "monitoring" / "upstream" / "latest.md"
    latest.parent.mkdir(parents=True, exist_ok=True)
    latest.write_text(
        f"# Latest DiffusionGemma upstream watch\n\n"
        f"- Snapshot: `{out_dir.relative_to(repo_root)}`\n"
        f"- Collected at: {snapshot['collected_at']}\n\n"
        f"See `{out_dir.relative_to(repo_root)}/summary.md`.\n",
        encoding="utf-8",
    )
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
