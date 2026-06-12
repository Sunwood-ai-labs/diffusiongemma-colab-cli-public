# Browser-Owned L4 Gemma 4 No-Conversation Attempt

Date: 2026-06-12 JST

## Route

- Route label: `browser-owned-adopted`
- Browser surface: Chrome CDP profile `Chrome-Colab-CDP`
- Runtime request: L4 GPU
- Adopted session name: `browser-owned-l4`
- Assignment endpoint: `gpu-l4-s-kkb-ass1a2-1f2bnfvyz6h9w`

## Gate Results

- Browser UI login and Colab page: passed.
- Runtime type set through Colab UI: passed.
- `colab sessions` browser-owned orphan assignment: passed.
- Temporary `--config` adoption: passed.
- `colab --config ... status`: passed.
- GPU check inside adopted runtime: passed.

GPU proof:

```json
{
  "cuda_available": true,
  "torch_gpu": "NVIDIA L4",
  "nvidia_smi": "NVIDIA L4, 23034 MiB"
}
```

## Benchmark Attempt

Script:

- `/content/scripts/benchmark_gemma4_nocnv_l4.py`
- wrapper: `/content/scripts/benchmark_gemma4_nocnv_l4_32.py`

Command target:

- `llama-cli`
- model: `ggml-org/gemma-4-26B-A4B-it-GGUF`
- quant: `Q4_K_M`
- generated tokens: `32`
- flags: `-ngl 99 --no-warmup -no-cnv`

Progress observed:

- `apt_update_done`: 6.416 sec
- `apt_install_done`: 6.036 sec
- `pip_install_done`: 4.623 sec
- `clone_done`: success
- `checkout_done`: success
- `build_done`: 614.935 sec
- `download_done`: 53.424 sec
- `inference_start`: reached with `-no-cnv`

## Result

No final Gemma 4 decode throughput was collected.

At 2026-06-12 00:27:32 JST, the remote WebSocket connection was lost during
the 32-token inference. A later assignment check returned an empty assignment
list, so the browser-owned L4 runtime disappeared before the script could write
final benchmark artifacts.

## Current Interpretation

- The browser-owned Colab route was successfully used this time.
- The benchmark command was corrected with `-no-cnv`.
- The previous command bug was not repeated.
- The regular Gemma 4 speed value is still not measured because the browser-
  owned L4 runtime disappeared during inference.
- Therefore the claim that DiffusionGemma was slower than regular Gemma 4
  remains unverified.
