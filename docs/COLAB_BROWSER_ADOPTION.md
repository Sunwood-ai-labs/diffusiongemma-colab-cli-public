# Colab Browser-Owned Runtime Adoption

This repository uses the browser-owned Colab route for long or account-backed
GPU experiments.

## Required rule

Do not replace a browser-owned experiment with a `colab new` CLI-created
session. A CLI-created session can be used for bounded diagnostics, but it does
not prove that the browser-opened Colab runtime was used.

Before any long run, label the active route in the experiment log as exactly
one of:

- `browser-owned-adopted`
- `cli-created`
- `diagnostic-only`

If the goal is a browser-owned experiment, the run may start only after all
items below are true:

- the real browser UI is visible and connected to Colab;
- `colab sessions` shows the server-side `[?]` assignment or matching endpoint;
- `scripts/adopt_browser_colab_assignment.py` has written an isolated temp
  config;
- `colab --config "$colab_config" status -s <session>` succeeds.

If any item is missing, stop and record the missing proof surface. Do not fall
back to `colab new` unless the run is explicitly relabeled `diagnostic-only`.

## Procedure

1. Open the notebook in the intended Chrome profile and connect the requested
   accelerator in the Colab UI.
2. Verify the actual browser surface is usable and connected. A URL or tab
   title alone is not enough.
3. Run `colab sessions` and confirm a server-side assignment appears, usually
   as `[?]`.
4. Adopt that assignment into an isolated temporary config:

   colab_config="$(mktemp /tmp/colab-browser-owned.XXXXXX.json)"
   python3 scripts/adopt_browser_colab_assignment.py \
     --config "$colab_config" \
     --session browser-owned-l4 \
     --accelerator L4

5. Use the temp config for all CLI actions:

   colab --config "$colab_config" status -s browser-owned-l4
   colab --config "$colab_config" exec -s browser-owned-l4 path/to/script.py

6. Delete the temp config after download/cleanup.

## Proof surfaces

- Browser/UI: screenshot or DOM evidence that Colab is connected.
- CLI/server: `colab sessions` shows the same active server assignment.
- GPU: `nvidia-smi` from inside the adopted runtime shows the requested GPU.
- Run artifacts: benchmark metadata and logs recovered from the adopted
  runtime.

## Current blocker note

On 2026-06-11, the rerun could not proceed through this route because the Mac
screen was at the lock screen and `colab sessions` showed no active server-side
assignment. Without an active browser-owned assignment, there is nothing for the
CLI to adopt.
