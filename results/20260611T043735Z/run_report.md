# Run Report: 20260611T043735Z

Status: `blocked_drive_not_mounted`

What ran:

- `colab new -s diffusiongemma-h100-smoke --gpu H100`
- `colab new -s diffusiongemma-g4-smoke --gpu G4`
- `colab drivemount -s diffusiongemma-g4-smoke`
- `colab exec -s diffusiongemma-g4-smoke --timeout 1800 -f scripts/run_diffusiongemma_probe.py`
- `colab stop -s diffusiongemma-g4-smoke`

Observed:

- H100 allocation failed with Colab `Service Unavailable`.
- G4 allocation succeeded.
- Actual assigned GPU: `NVIDIA RTX PRO 6000 Blackwell Server Edition`.
- GPU memory: `94.971 GiB`.
- Hugging Face model metadata lookup for `google/diffusiongemma-26B-A4B-it` succeeded.
- Drive mount failed during OAuth credential propagation with HTTP 400 / `ValueError: mount failed`.

Decision:

The run did not load DiffusionGemma weights or generate text. This is intentional: Drive direct save was not established, so the script stopped before long model download/load to avoid relying on ephemeral `/content` outputs.

Next step:

Repair Colab Drive OAuth/account propagation, rerun `colab drivemount`, verify `/content/drive/MyDrive`, then rerun the same probe without changing the model-load guard.
