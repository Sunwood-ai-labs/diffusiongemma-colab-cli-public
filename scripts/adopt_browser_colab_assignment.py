#!/usr/bin/env python3
"""Adopt a browser-owned Colab assignment into an isolated colab-cli config.

This is for the reverse-hybrid workflow:

1. Open/connect the runtime in the real Colab browser UI first.
2. Confirm `colab sessions` shows a server-side orphan assignment as `[?]`.
3. Run this helper against a temporary `--config` path.
4. Use `colab --config <temp-config> ...` for upload/exec/download.

The runtime proxy token is written only to the chosen temp config file. It is
never printed. Do not point this at the default sessions.json.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from colab_cli.auth import AuthProvider
from colab_cli.common import state
from colab_cli.state import SessionState, StateStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Isolated session state JSON path.")
    parser.add_argument("--session", default="browser-owned", help="Local session name to write.")
    parser.add_argument("--accelerator", default=None, help="Required accelerator, for example L4 or T4.")
    parser.add_argument(
        "--client-oauth-config",
        default=os.path.expanduser("~/.colab-cli-oauth-config.json"),
        help="Same OAuth client config used by colab-cli.",
    )
    parser.add_argument("--auth", default="oauth2", choices=["oauth2", "adc"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).expanduser()
    default_path = Path("~/.config/colab-cli/sessions.json").expanduser()
    if config_path == default_path:
        raise SystemExit("Refusing to write the default colab-cli sessions.json.")

    state.client_oauth_config = args.client_oauth_config
    state.config_path = str(config_path)
    state.auth_provider = AuthProvider(args.auth)

    assignments = state.client.list_assignments()
    if args.accelerator:
        assignments = [
            a for a in assignments if a.accelerator.value.upper() == args.accelerator.upper()
        ]
    if not assignments:
        accel = f" accelerator={args.accelerator}" if args.accelerator else ""
        raise SystemExit(f"No browser-owned Colab assignment found{accel}.")
    if len(assignments) > 1:
        endpoints = ", ".join(a.endpoint for a in assignments)
        raise SystemExit(f"Multiple matching assignments found: {endpoints}")

    assignment = assignments[0]
    StateStore(str(config_path)).add(
        SessionState(
            name=args.session,
            token=assignment.runtime_proxy_info.token,
            url=assignment.runtime_proxy_info.url,
            endpoint=assignment.endpoint,
            variant=assignment.variant.name,
            accelerator=assignment.accelerator.value,
        )
    )

    print(
        "adopted "
        f"session={args.session} endpoint={assignment.endpoint} "
        f"accelerator={assignment.accelerator.value} variant={assignment.variant.name}"
    )
    print(f"config={config_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
