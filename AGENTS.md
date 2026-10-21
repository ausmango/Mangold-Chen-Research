# Repository instructions

- Purpose: research local construction-scene analysis on NVIDIA Jetson Orin Nano.
- No sudo access is available. Do not use sudo or plan work that requires it.
- Preserve the working PyTorch, CUDA, TorchAO, Transformers, and model dependencies.
  Do not install vLLM, upgrade the stack, or change model dependencies for this benchmark.
- Before experiment work, read `docs/research-log/` and `docs/context.md` for
  experiment history, current implementation, and known limitations.
- Preserve prior scripts, benchmarks, raw logs, and results. Use a unique result
  folder for each attempt, including failed attempts; never overwrite evidence.
- Record the exact model ID/revision, prompt, generation settings, model size
  (identify checkpoint size versus loaded footprint), load time, inference
  latency, capture-to-answer delay, and RAM usage. Record frame age and CUDA
  allocation/peak separately when available. Use tegrastats for whole-system RAM.
- Never describe generated hazard claims as verified accuracy. Independent live
  images do not establish temporal event detection; state limitations explicitly.
- Document completed work and failures using actual saved evidence. Distinguish
  CPU-only checks from hardware runs; never invent measurements or outputs.
- Keep the concurrent experiment bounded: one inference worker, a latest-frame
  slot instead of a frame queue, and OpenCV windows on the main thread. Preserve
  the pinned revision, loading workaround, and working CSI pipeline.
- Do not launch camera capture or GPU inference until the user confirms camera
  readiness. After a run, inspect saved logs/results and update the dated
  research note and `docs/context.md` with measurements, failures, and next steps.
- Do not commit or push unless the user requests it.
