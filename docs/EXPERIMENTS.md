# Experiments

## 2026-06-11 DiffusionGemma Colab CLI smoke

Goal: Try `google/diffusiongemma-26B-A4B-it` through Google Colab CLI and record whether the assigned Colab GPU can run a minimal generation.

Expected blocker: The official Google AI docs say the Hugging Face notebook requires a GPU with more than 60 GB of memory, such as NVIDIA G4 or H100. T4/L4 sessions should be treated as environment probes only.

Proof surfaces to capture:

- Colab CLI version and session creation output.
- `nvidia-smi` from inside Colab.
- Drive mount proof from inside Colab.
- Direct result files under `/content/drive/MyDrive/diffusiongemma-colab-cli-public/`.
- `experiment_metadata.json` status.

Current local result: pending live Colab CLI run.

Live run notes:

- `diffusiongemma-h100-smoke`: H100 assignment failed with Colab `Service Unavailable`.
- `diffusiongemma-g4-smoke`: G4 assignment succeeded, but Drive mount failed during credential propagation with HTTP 400 and `ValueError: mount failed`.
- G4 hardware evidence from inside Colab: `NVIDIA RTX PRO 6000 Blackwell Server Edition`, 94.971 GiB VRAM, CUDA available.
- Hugging Face model reachability succeeded for `google/diffusiongemma-26B-A4B-it`, SHA `0f28bc42f588fbd8f71e08102b1c3960298a1358`, last modified `2026-06-10 13:45:13+00:00`, tag `license:apache-2.0`.
- Safety rule applied: because Drive direct save was not established, the probe must stop before long model weight download/load unless explicitly run with `ALLOW_NO_DRIVE_MODEL_LOAD=1`.

Recovered local artifacts:

- `results/20260611T043735Z/experiment_metadata.json`
- `results/20260611T043735Z/progress.jsonl`
- `results/20260611T043735Z/result_summary.md`

## 2026-06-11 L4/T4 quantized pivot

Reason: Full `google/diffusiongemma-26B-A4B-it` is too large for ordinary L4/T4 experiments.

Selected route:

- Model repo: `unsloth/diffusiongemma-26B-A4B-it-GGUF`
- Quant: `Q4_K_M`
- Runner: DiffusionGemma-specific `llama-diffusion-cli` from llama.cpp PR `24423`
- Target accelerator: L4 first, T4 fallback only if L4 is unavailable

Why not NVFP4 first:

- `nvidia/diffusiongemma-26B-A4B-it-NVFP4` and `RedHatAI/diffusiongemma-26B-A4B-it-NVFP4` are real quantized candidates, but their cards route through vLLM and emphasize Blackwell/Hopper/B200-class evidence. That is not the conservative first choice for T4/L4.

Live L4 result:

- Session: `diffusiongemma-l4-quant-smoke`
- Hardware: `NVIDIA L4`, 22.034 GiB VRAM
- Build: llama.cpp PR `24423`, target `llama-diffusion-cli`, 575.346 sec
- Download: `diffusiongemma-26B-A4B-it-Q4_K_M.gguf`, 65.315 sec
- Inference: return code 0, 29.424 sec
- Status: `generated`
- Quality note: generated text was not a reliable explanation of DiffusionGemma; it conflated text diffusion with image diffusion. Treat this as a runtime proof, not a quality pass.
- Drive note: Drive was not mounted for this short smoke; artifacts were downloaded from `/content` before session stop.

Recovered local artifacts:

- `results/20260611T044221Z/experiment_metadata.json`
- `results/20260611T044221Z/progress.jsonl`
- `results/20260611T044221Z/result_summary.md`
- `results/20260611T044221Z/generation_stdout.txt`
- `results/20260611T044221Z/generation_stderr.txt`

## 2026-06-11 L4 visual terminal capture

Reason: A reconstructed animation is not sufficient evidence for how
DiffusionGemma's canvas changes during generation. Rerun the L4/T4 GGUF path
with llama.cpp visual diffusion mode and capture the actual terminal surface.

Session:

- `diffusiongemma-l4-gguf-visual`
- Hardware: `NVIDIA L4`, 22.034 GiB VRAM
- Model repo: `unsloth/diffusiongemma-26B-A4B-it-GGUF`
- Quant: `Q4_K_M`
- Runner: `llama-diffusion-cli`

Visual command additions:

- `--diffusion-visual`
- `--diffusion-visual-progress`
- `--diffusion-visual-interval 1`
- `--log-file /content/diffusiongemma-colab-cli-public/<timestamp>/llama_diffusion_visual.log`

Result:

- Build reuse in the same Colab session worked; rebuild step completed in 0.713 sec after the initial 565.684 sec build.
- Inference terminal capture reached visible `diffusion step: 0/48` through `20/48`.
- Final terminal output reported `total time: 9289.70ms, time per step: 442.37ms (21 steps over 1 blocks, entropy-bound)`.
- The local script incorrectly marked the first captured run as `inference_failed` because the PTY process wait path returned `-9` after the final terminal output had already been captured. The script was patched afterward to wait for process exit before classifying return code.
- Colab session was stopped after artifact recovery; `colab sessions` reported no active sessions.

Recovered local artifacts:

- `results/20260611T080801Z_visual_capture/visual_terminal_capture.ansi`
- `results/20260611T080801Z_visual_capture/llama_diffusion_visual.log`
- `results/20260611T080801Z_visual_capture/generation_terminal_tail.txt`
- `results/20260611T080801Z_visual_capture/progress.jsonl`
- `results/visual_capture_local/colab_exec_visual_retry2_20260611T080748Z.typescript`
- `videos/l4-quant-generation-flow/renders/diffusiongemma-l4-q4-real-terminal-visual-capture.mp4`
- `videos/l4-quant-generation-flow/final/diffusiongemma-l4-q4-real-terminal-visual-capture-sunwood.mp4`
- `videos/l4-quant-generation-flow/final/qa-009s.png`
- `videos/l4-quant-generation-flow/final/qa-019s.png`

Sunwood AI Labs color rerender:

- The ANSI replay renderer was restyled with the local Sunwood AI Labs palette:
  deep green felt background, cream text, gold diffusion-step emphasis, cyan
  timing metrics, and red/gold header rules.
- The canonical MP4 was overwritten with the Sunwood version:
  `videos/l4-quant-generation-flow/renders/diffusiongemma-l4-q4-real-terminal-visual-capture.mp4`
- A trace copy was also kept:
  `videos/l4-quant-generation-flow/renders/diffusiongemma-l4-q4-real-terminal-visual-capture-sunwood.mp4`
- Public final artifacts were copied into:
  `videos/l4-quant-generation-flow/final/`
- QA frames extracted from the final MP4:
  `videos/l4-quant-generation-flow/final/qa-009s.png`
  and
  `videos/l4-quant-generation-flow/final/qa-019s.png`
- Visual QA result: 9s and 19s frames showed the Sunwood colors and readable
  Japanese text, with no missing-glyph boxes.

Public post:

- X/Twitter: https://x.com/haru_maki_ch/status/2064990568846127528?s=46

## 2026-06-11 L4 visual terminal capture - robot barista example

Reason: The first visual demo used the DiffusionGemma explanation prompt. A
separate example was run so the video is not just another model-explainer clip.

Session:

- `diffusiongemma-l4-llama-diffusion-alt`
- Hardware: `NVIDIA L4`, 22.034 GiB VRAM
- Model repo: `unsloth/diffusiongemma-26B-A4B-it-GGUF`
- Quant: `Q4_K_M`
- Runner: `llama-diffusion-cli`

Prompt:

- `Write a concise 3-bullet lab note about a robot barista calibrating espresso shots on a tiny Mars cafe.`

Result:

- The existing Colab build and downloaded GGUF were reused.
- The corrected quick script ran `llama-diffusion-cli --diffusion-visual`.
- The terminal capture reached visible `diffusion step: 0/48` through `20/48`.
- Final terminal output reported `total time: 4551.07ms, time per step: 216.72ms (21 steps over 1 blocks, entropy-bound)`.
- A previous quick rerun accidentally reused the old `PROMPT` environment variable from the Colab kernel. The script now reads `QUICK_PROMPT` for quick examples to prevent that recurrence.
- Colab session was stopped after artifact recovery; `colab sessions` reported no active sessions.

Recovered local artifacts:

- `results/20260611T113411Z_robot_barista_visual_capture/visual_terminal_capture.ansi`
- `results/20260611T113411Z_robot_barista_visual_capture/llama_diffusion_visual.log`
- `results/20260611T113411Z_robot_barista_visual_capture/generation_terminal_tail.txt`
- `results/20260611T113411Z_robot_barista_visual_capture/progress.jsonl`
- `results/20260611T113411Z_robot_barista_visual_capture/experiment_metadata.json`
- `results/20260611T113411Z_robot_barista_visual_capture/run_report.md`
- `videos/l4-quant-generation-flow/final/diffusiongemma-l4-q4-robot-barista-llama-diffusion-cli.mp4`
- `videos/l4-quant-generation-flow/final/diffusiongemma-robot-barista-ios-compatible.mp4`
- `videos/l4-quant-generation-flow/final/robot-barista-qa-009s.png`
- `videos/l4-quant-generation-flow/final/robot-barista-qa-019s.png`
- `videos/l4-quant-generation-flow/final/robot-barista-mobile-preview-screenshot.png`

## 2026-06-11 L4 speed check - DiffusionGemma vs regular Gemma 4

Question: The posted experiment raised a concern that DiffusionGemma might be
slower than regular Gemma 4. A direct L4 comparison was attempted.

Posted URL:

- https://x.com/haru_maki_ch/status/2064990568846127528?s=46

DiffusionGemma evidence already available:

- Robot-barista visual run: `total time: 4551.07ms`, `throughput: 56.3 tok/s`
  for 256 canvas tokens over 21 entropy-bound steps.
- Earlier explainer visual run: `total time: 9289.70ms`, `throughput: 27.6 tok/s`
  for 256 canvas tokens over 21 entropy-bound steps.
- Caveat: DiffusionGemma canvas throughput is not the same metric as regular
  autoregressive decode tok/s.

Direct comparison attempt:

- Session: `diffusiongemma-vs-gemma4-l4-speed`
- Hardware: `NVIDIA L4`, 22.034 GiB VRAM
- Checkout: llama.cpp PR `24423`, branch `diffusiongemma`
- Build: `llama-cli` and `llama-diffusion-cli`, 622.450 sec
- Downloaded `unsloth/diffusiongemma-26B-A4B-it-GGUF` Q4_K_M and
  `unsloth/gemma-4-26B-A4B-it-GGUF` UD-Q4_K_M.
- DiffusionGemma inference completed before the regular Gemma 4 run.
- Regular Gemma 4 256-token run did not complete before the Colab CLI session
  disappeared from the server/local session list.

Short regular Gemma 4 rerun:

- Session: `gemma4-short-l4-speed`
- Hardware: `NVIDIA L4`, 22.034 GiB VRAM
- Checkout: llama.cpp PR `24423`, branch `diffusiongemma`
- Model: `ggml-org/gemma-4-26B-A4B-it-GGUF`, Q4_K_M
- Command target: 32 generated tokens with `llama-cli`
- Result: timed out after 307.251 sec, with no llama.cpp eval throughput
  reported.
- Recovered artifacts:
  - `results/20260611T124246Z_gemma4_short_speed/gemma4_short_benchmark_metadata.json`
  - `results/20260611T124246Z_gemma4_short_speed/gemma4_short_speed_report.md`
  - `results/20260611T124246Z_gemma4_short_speed/progress.jsonl`

Root-cause diagnosis:

- Diagnostic run:
  `results/20260611T132153Z_gemma4_timeout_diagnosis/diagnosis_metadata.json`
- The original regular Gemma 4 command let `llama-cli` auto-enable conversation
  mode from the Gemma 4 chat template.
- The run was not stuck in model loading. It generated the requested token, then
  stayed alive in an interactive prompt loop and printed repeated `>` prompts.
- Evidence:
  - `-ngl 99`, `-n 1`: 794,793,602 bytes of stdout in 184.718 sec.
  - `-ngl 0`, `-n 1`: 844,409,922 bytes of stdout in 183.919 sec.
  - stderr reported prompt eval timing:
    - `-ngl 99`: 25 prompt tokens at 72.76 tokens/sec.
    - `-ngl 0`: 25 prompt tokens at 21.08 tokens/sec.
- Therefore the 300 sec "timeout" was primarily a benchmark command bug:
  missing `-no-cnv` / no-conversation mode.

Corrected attempt:

- Script: `scripts/benchmark_gemma4_nocnv_l4.py`
- Command added `-no-cnv`.
- Session: `gemma4-nocnv-speed-l4`
- The corrected run no longer showed the interactive prompt loop in the local
  transcript, but the Colab CLI session was pruned during the 128-token
  inference before the script could write final artifacts.
- This is a separate Colab CLI keep-alive/session-lifetime problem.

Conclusion:

- The claim "DiffusionGemma was slower than regular Gemma 4" was not confirmed
  by this L4 check.
- The original regular-Gemma timeout was caused by the benchmark command
  entering conversation mode, not by proven model slowness.
- After fixing conversation mode with `-no-cnv`, the Colab CLI runtime was
  pruned before a final 128-token throughput could be collected.
- This comparison did not follow the required browser-owned Colab route. It
  used CLI-created sessions, so it is a command/session diagnosis only, not a
  browser-opened Colab experiment.
- A proper regular-Gemma baseline must use `-no-cnv`, first connect the runtime
  from the real Colab browser UI, then adopt the `[?]` browser-owned assignment
  through an isolated temporary `--config` file before running CLI commands.

## 2026-06-11 browser-owned Colab rerun audit

Reason:

- The user pointed out that the documented workflow says to open Colab in the
  browser and have the CLI grab that runtime. The prior speed check did not do
  that.

What was checked:

- Existing normal Chrome `Profile 1` process was present with a Colab URL:
  `https://colab.research.google.com/notebooks/empty.ipynb?create=true&language=python3&accelerator=GPU&gpuType=L4`
- HTB proxy Chrome was also present on remote-debugging port `9223`; it was not
  touched.
- `colab sessions` returned: `[colab] No active sessions found on server.`
- A direct screen capture showed the Mac was at the lock screen, not an
  operable Colab notebook UI.
- Therefore no browser-owned server-side assignment existed for CLI adoption.

Operational correction:

- Do not use `colab new` for this rerun.
- First unlock/operate the real Chrome Colab UI, connect the requested L4
  runtime, verify the browser UI is connected, and only then adopt the
  server-side `[?]` assignment with:
  `scripts/adopt_browser_colab_assignment.py`.
- The required procedure is recorded in `docs/COLAB_BROWSER_ADOPTION.md`.

Current status:

- Proper browser-owned Gemma 4 baseline is still blocked by browser/UI access:
  the Mac is locked and `colab sessions` has no active assignment to adopt.
- The earlier timeout root cause remains valid: missing `-no-cnv` caused
  conversation mode to print repeated prompts after generation.

## 2026-06-12 browser-owned L4 Gemma 4 no-conversation attempt

Route:

- Route label: `browser-owned-adopted`
- Browser surface: Chrome CDP profile `Chrome-Colab-CDP`
- Runtime request: L4 GPU
- Adopted local session name: `browser-owned-l4`
- Assignment endpoint: `gpu-l4-s-kkb-ass1a2-1f2bnfvyz6h9w`

Preflight gate:

- The Colab browser page was opened through CDP.
- The logged-in Colab UI was confirmed by screenshot.
- Runtime type was changed through the Colab UI to `L4 GPU`.
- `colab sessions` showed:
  `[?] gpu-l4-s-kkb-ass1a2-1f2bnfvyz6h9w | Hardware: L4 | Variant: GPU`
- The orphan assignment was adopted with
  `scripts/adopt_browser_colab_assignment.py`.
- `colab --config ... status -s browser-owned-l4` showed the adopted L4 session.
- A GPU proof script inside the adopted runtime reported:
  `torch_gpu: NVIDIA L4` and `nvidia_smi: NVIDIA L4, 23034 MiB`.

Benchmark attempt:

- Script: `scripts/benchmark_gemma4_nocnv_l4_32.py`
- Underlying script: `scripts/benchmark_gemma4_nocnv_l4.py`
- Generated tokens: `32`
- Command included `-no-cnv`.
- Build completed in 614.935 sec.
- Model download completed in 53.424 sec.
- Inference started with:
  `llama-cli -m ...gemma-4-26B-A4B-it-Q4_K_M.gguf ... -n 32 ... -ngl 99 --no-warmup -no-cnv`

Result:

- No final Gemma 4 decode throughput was collected.
- During inference, the remote WebSocket was lost at 2026-06-12 00:27:32 JST.
- A later `colab sessions` call returned `[colab] No active sessions found on server.`
- This means the browser-owned L4 runtime disappeared during inference before
  benchmark artifacts could be written.

Recovered evidence:

- `results/20260612T0006Z_browser_owned_l4_gemma4_nocnv_attempt/browser_owned_l4_connected.png`
- `results/20260612T0006Z_browser_owned_l4_gemma4_nocnv_attempt/run_report.md`

Conclusion:

- The browser-owned route was successfully used this time.
- The previous missing-`-no-cnv` command bug was not repeated.
- The regular Gemma 4 speed comparison remains incomplete because the L4
  runtime disappeared during corrected inference.

## 2026-06-12 browser-owned L4 Gemma 4 completion baseline

Route:

- Route label: `browser-owned-adopted`
- Browser surface: Colab UI connected to L4 through the existing Chrome CDP
  profile.
- CLI surface: server-side `[?]` L4 assignment adopted into a temporary
  `--config`.
- Assignment endpoint: `gpu-l4-s-kkb-ass1a2-2ue03uoev4dhq`
- GPU proof: `NVIDIA L4`, 22.034 GiB from inside Colab.

What changed:

- `llama-cli -no-cnv` was confirmed to be the wrong executable for a clean
  non-conversation baseline on this PR branch. It entered prompt mode and
  produced repeated `>` prompts until timeout.
- `llama-cli -no-cnv -st` exited, but printed:
  `--no-conversation is not supported by llama-cli; please use llama-completion instead`.
- `llama-completion` then failed once without `--jinja` because Gemma4's custom
  chat template requires Jinja handling.
- Final working command used `llama-completion --jinja -no-cnv`.

Final command:

```text
/content/llama.cpp/build/bin/llama-completion \
  -m /content/models/ggml-org__gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-Q4_K_M.gguf \
  -p "Write a concise 3-bullet lab note about a robot barista calibrating espresso shots on a tiny Mars cafe." \
  -n 8 --temp 0.2 -ngl 99 --no-warmup --jinja -no-cnv
```

Result:

- Build target: `llama-completion`
- Build time: 575.407 sec
- Model download time: 51.292 sec
- Prompt eval: 24 tokens, 13.35 ms/token, 74.91 tok/s
- Decode eval: 7 runs, 18.86 ms/token, 53.01 tok/s
- Total timing: 462.40 ms / 31 tokens
- Full command elapsed wall time: 12.192 sec including model load/process
  overhead.

Comparison:

- Existing DiffusionGemma robot-barista L4 result:
  4551.07 ms total, 216.72 ms/step, 56.3 tok/s over a 256-token canvas.
- Existing DiffusionGemma Sunwood-note L4 result:
  4754.52 ms total, 216.11 ms/step, 53.8 tok/s over a 256-token canvas.
- Existing earlier DiffusionGemma Japanese explainer L4 result:
  9289.70 ms total, 442.37 ms/step, 27.6 tok/s over a 256-token canvas.

Conclusion:

- The claim that DiffusionGemma was slower than regular Gemma4 is not supported
  by this local browser-owned L4 check when comparing the reported tok/s values:
  the closest robot-barista DiffusionGemma run was 56.3 tok/s and regular Gemma4
  completion decode was 53.01 tok/s.
- The comparison is not perfectly apples-to-apples. DiffusionGemma reports
  diffusion/canvas throughput over 256 tokens and steps, while regular Gemma4
  reports sequential decode timing for a short 8-token completion.

Evidence:

- `results/20260612T0208Z_browser_owned_l4_gemma4_completion_jinja_n8/metadata.json`
- `results/20260612T0208Z_browser_owned_l4_gemma4_completion_jinja_n8/stdout.txt`
- `results/20260612T0208Z_browser_owned_l4_gemma4_completion_jinja_n8/stderr.txt`
- `results/20260612T0208Z_browser_owned_l4_gemma4_completion_jinja_n8/run_report.md`
