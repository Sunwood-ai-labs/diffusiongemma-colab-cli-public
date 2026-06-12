# DiffusionGemma upstream monitoring

This repository includes the helper script used by the Codex Automation that
checks the upstream llama.cpp DiffusionGemma PR discussion twice per day and
stores snapshots under `monitoring/upstream/`.

## Targets

- ggml-org/llama.cpp PR #24423: DiffusionGemma
- ggml-org/llama.cpp PR #24427: Add diffusion-gemma block-diffusion support

The monitor searches for speed and offload signals such as `throughput`,
`tok/s`, `time per step`, `slow`, `L4`, `-ngl`, and `llama-diffusion-cli`.

## Manual run

```bash
python3 scripts/monitor_diffusiongemma_upstream.py
```

Outputs:

- `monitoring/upstream/<UTC timestamp>/snapshot.json`
- `monitoring/upstream/<UTC timestamp>/summary.md`
- `monitoring/upstream/latest.md`

## Codex Automation

Live Codex Automation entrypoint:

- `/Users/admin/.codex/automations/diffusiongemma-upstream-speed-monitor/automation.toml`

Schedule:

- 09:15 JST
- 21:15 JST

The automation runs Codex against this repository, calls the helper script, and
then appends a concise local note under `monitoring/upstream/codex-runs/`.

This is intentionally a Codex Automation, not a macOS `launchd` or cron job.

## Scope

This automation does not run Colab or model inference. It only records upstream
GitHub issue and PR discussion state so local benchmark follow-up can be planned
from current upstream evidence.
