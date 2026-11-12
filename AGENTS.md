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

## USB Scout & Inspector scope (November 12, 2026)

- New development uses `main.py` and `src/scout_inspector/`; preserve the legacy
  pinned Moondream/CSI scripts and results as historical baselines.
- The authorized architectural pivot uses a USB Scout worker and one serial
  sub-1B Inspector worker, with `queue.Queue(maxsize=1)` replacing stale pending
  triggered frames. OpenCV remains on the main thread.
- Follow `docs/scout-inspector-architecture.md`. Treat 30 FPS, present-second
  answers and a sub-400 MB model runtime as unverified targets until measured.
- Optional adapters/export recipes do not authorize dependency installation,
  stack upgrades, model downloads or hardware runs. Preserve the existing stack.
