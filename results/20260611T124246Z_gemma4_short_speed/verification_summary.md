# Speed Verification Summary

Date: 2026-06-11

Question: Was the DiffusionGemma run slower than regular Gemma 4, and why did
the regular Gemma 4 comparison time out?

## Result

The claim was not confirmed, but the checked L4 setup was not the proper
browser-owned Colab workflow requested for the final comparison.

DiffusionGemma completed measurable L4 runs:

- Robot-barista visual run: 56.3 tok/s for 256 canvas tokens in 4551.07 ms.
- Earlier explainer visual run: 27.6 tok/s for 256 canvas tokens in 9289.70 ms.

Regular Gemma 4 26B-A4B Q4_K_M did not provide a usable apples-to-apples
throughput number in the same L4 validation path:

- A 256-token regular Gemma 4 comparison run disappeared with the Colab CLI
  session before completion.
- A shorter 32-token regular Gemma 4 run on `ggml-org/gemma-4-26B-A4B-it-GGUF`
  hit the 300 second subprocess limit.

## Root Cause

The 300 second result was not evidence that Gemma 4 needed 300 seconds to
generate 32 tokens.

The diagnostic rerun showed that `llama-cli` auto-enabled conversation mode
from the Gemma 4 chat template. With the original command, it generated the
requested token, then stayed in an interactive prompt loop instead of exiting.
That loop printed repeated `>` prompts to stdout. In the diagnosis:

- `-ngl 99`, `-n 1`: 794,793,602 bytes of stdout in 184.718 seconds.
- `-ngl 0`, `-n 1`: 844,409,922 bytes of stdout in 183.919 seconds.
- The stderr timing showed prompt eval did complete:
  - `-ngl 99`: 25 prompt tokens at 72.76 tokens/sec.
  - `-ngl 0`: 25 prompt tokens at 21.08 tokens/sec.

So the immediate bug was the benchmark command, not a proven Gemma 4 speed
failure. The command needed `-no-cnv` / no-conversation mode.

## Corrected Attempt

A corrected `-no-cnv` run was started with 128 generated tokens. It no longer
hit the interactive `>` prompt loop, but the Colab CLI session was pruned before
the run completed and before artifacts could be recovered.

This is a separate Colab CLI keep-alive/session-lifetime failure, not the same
cause as the earlier 300 second timeout.

## Browser-Owned Route Audit

The prior comparison used CLI-created sessions. That was a workflow error for
the requested Colab experiment route.

The required rerun path is:

- open/connect the L4 runtime in the real Colab browser UI first;
- confirm `colab sessions` shows the browser-owned server-side assignment,
  usually as `[?]`;
- adopt that assignment with an isolated temporary `--config`;
- run the corrected Gemma 4 command with `-no-cnv` through that adopted runtime.

Current rerun status:

- Normal Chrome `Profile 1` had a Colab URL open.
- `colab sessions` returned no active server-side assignments.
- A screen capture showed the Mac lock screen, so the Colab UI could not be
  operated and no browser-owned runtime could be connected.
- The adoption helper and procedure are recorded in:
  `scripts/adopt_browser_colab_assignment.py` and
  `docs/COLAB_BROWSER_ADOPTION.md`.

## Browser-Owned L4 Rerun

A corrected rerun did reach the required browser-owned route:

- Colab was opened through the real browser/CDP surface.
- The Colab UI was logged in.
- Runtime type was changed through the UI to `L4 GPU`.
- `colab sessions` showed a browser-owned `[?]` assignment:
  `gpu-l4-s-kkb-ass1a2-1f2bnfvyz6h9w | Hardware: L4 | Variant: GPU`.
- The assignment was adopted with a temporary `--config` as `browser-owned-l4`.
- Inside the adopted runtime, GPU proof reported:
  `NVIDIA L4, 23034 MiB`.

The corrected Gemma 4 benchmark then started with:

- `-n 32`
- `-ngl 99`
- `--no-warmup`
- `-no-cnv`

It completed setup, build, and model download, then reached inference start.
During inference, the remote WebSocket was lost and the assignment list became
empty. No final decode throughput or benchmark artifact was produced.

Evidence is stored at:

- `results/20260612T0006Z_browser_owned_l4_gemma4_nocnv_attempt/run_report.md`
- `results/20260612T0006Z_browser_owned_l4_gemma4_nocnv_attempt/browser_owned_l4_connected.png`

## Caveat

DiffusionGemma's reported throughput is canvas-token throughput over diffusion
steps. It is not identical to autoregressive Gemma 4 decode tok/s. The practical
claim that can be made from these runs is narrower:

Under this Colab L4 + development llama.cpp PR setup, DiffusionGemma completed
and reported speed. The original regular Gemma 4 comparison was invalid because
conversation mode kept `llama-cli` alive after generation. The corrected
regular Gemma 4 comparison still does not have a final throughput number because
the browser-owned L4 runtime disappeared during corrected inference. Therefore
this evidence does not support
saying that the posted DiffusionGemma experiment was slower than regular Gemma
4.
