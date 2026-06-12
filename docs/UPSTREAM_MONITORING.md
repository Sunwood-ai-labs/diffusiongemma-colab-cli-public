# DiffusionGemma upstream monitoring

This repository checks the upstream llama.cpp DiffusionGemma PR discussion twice
per day and stores snapshots under `monitoring/upstream/`.

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

## Local automation

Install or refresh the macOS launchd job:

```bash
bash scripts/install_upstream_monitor_launchd.sh
```

Schedule:

- 09:15 JST
- 21:15 JST

LaunchAgent:

- `~/Library/LaunchAgents/com.sunwood.diffusiongemma.upstream-monitor.plist`

Logs:

- `monitoring/upstream/launchd.stdout.log`
- `monitoring/upstream/launchd.stderr.log`

## Scope

This automation does not run Colab or model inference. It only records upstream
GitHub issue and PR discussion state so local benchmark follow-up can be planned
from current upstream evidence.
