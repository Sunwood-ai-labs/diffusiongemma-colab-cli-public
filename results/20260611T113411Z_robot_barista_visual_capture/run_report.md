# Run Report: Robot Barista Visual Capture

Date: 2026-06-11

## Summary

- Status: generated visual capture
- Session: diffusiongemma-l4-llama-diffusion-alt
- Hardware: Google Colab L4 session
- Model: unsloth/diffusiongemma-26B-A4B-it-GGUF
- Quant: Q4_K_M
- Runner: llama-diffusion-cli
- Capture mode: PTY ANSI terminal replay with --diffusion-visual
- Prompt: Write a concise 3-bullet lab note about a robot barista calibrating espresso shots on a tiny Mars cafe.

## Evidence

- Raw ANSI capture: visual_terminal_capture.ansi
- llama-diffusion-cli log: llama_diffusion_visual.log
- Terminal tail: generation_terminal_tail.txt
- Progress log: progress.jsonl
- Metadata: experiment_metadata.json
- Local transcript: ../visual_capture_local/colab_exec_robot_barista_20260611T113410Z.typescript

## Rendered Artifacts

- Final MP4: ../../videos/l4-quant-generation-flow/final/diffusiongemma-l4-q4-robot-barista-llama-diffusion-cli.mp4
- iOS-compatible MP4: ../../videos/l4-quant-generation-flow/final/diffusiongemma-robot-barista-ios-compatible.mp4
- QA 9s frame: ../../videos/l4-quant-generation-flow/final/robot-barista-qa-009s.png
- QA 19s frame: ../../videos/l4-quant-generation-flow/final/robot-barista-qa-019s.png
- Mobile-width preview screenshot: ../../videos/l4-quant-generation-flow/final/robot-barista-mobile-preview-screenshot.png

## Notes

- A prior quick rerun reused the old PROMPT environment variable inside the Colab kernel. The quick script was corrected to read QUICK_PROMPT instead, then rerun with the robot-barista prompt.
- Colab CLI keep-alive had earlier 403 USER_PROJECT_DENIED evidence, so this is treated as a short-run visual capture, not a robust long-running Colab workflow proof.
- The Colab session was stopped after artifact recovery; colab sessions reported no active sessions.
