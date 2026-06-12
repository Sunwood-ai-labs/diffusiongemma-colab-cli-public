#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
label="com.sunwood.diffusiongemma.upstream-monitor"
src="${repo_root}/automation/${label}.plist"
dst="${HOME}/Library/LaunchAgents/${label}.plist"

mkdir -p "${HOME}/Library/LaunchAgents" "${repo_root}/monitoring/upstream"
cp "${src}" "${dst}"
chmod 644 "${dst}"

launchctl bootout "gui/$(id -u)" "${dst}" >/dev/null 2>&1 || true
launchctl bootstrap "gui/$(id -u)" "${dst}"
launchctl enable "gui/$(id -u)/${label}"
launchctl kickstart -k "gui/$(id -u)/${label}" || true

echo "${dst}"
