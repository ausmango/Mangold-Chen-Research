# Meeting evidence — prepared October 21 for October 22, 2026

> **October 22 completion update:** All 30 new CSI-captured samples were
> subsequently analyzed successfully in
> `results/camera_analysis_20261022T163529601772802` (exit 0). Mean inference
> **4.4612 s**, **46.41% lower** than October 5's 8.3241 s; peak RAM 5401 MB,
> footprint 1.8969 GiB, peak CUDA 2.4977 GiB. Sequential capture then inference;
> no concurrent/live response success. Outputs are terse and quality unverified.
> Full current report: [October 22 comparison](2026-10-22-camera-benchmark-comparison.md).
> The material below is earlier experiment history; this update supersedes its
> statement that only the three-image control had completed.


## Outcome

Camera-only recording completed near 30 FPS. After the user quit Codex, the
latest saved-image control completed all three queries: mean inference 7.5390 s,
29.79% lower latency than the same three images in the October 5 baseline.
Concurrent CSI capture and inference still has no successful run. Closing Codex
is associated with more available RAM and restored loading; shorter generation
settings prevent attributing the latency reduction to Codex alone.

| Experiment | Completed work | Measurements / outcome |
|---|---|---|
| October 5 sequential baseline | 30 images, five videos | Mean inference 8.3241 s; model footprint 1.8969 GiB; peak whole-system RAM 5400 MB |
| October 21 camera-only recording | 596 frames | Measured 29.7559 FPS; first-to-last delivery span 19.9960 s; max delivery gap 0.2023 s; peak whole-system RAM 3456 MB (62 samples) |
| October 21 model-first concurrent attempts | No timed images/answers | Model loaded, but Argus CaptureSession failed; latest such run peak RAM 5083 MB (95 samples) |
| October 21 camera-first concurrent attempt | Camera initially delivered; no analyzed images | Argus InsufficientMemory during model load; process stalled; camera-first ordering did not resolve the issue |
| Earlier October 21 saved-image control, camera closed | 0/3 inference tasks | Model load failed with CUDA allocator NVML_SUCCESS assertion; peak system RAM 4393 MB (53 samples) |
| Latest control with Codex closed | 3/3 inference tasks | Mean 7.5390 s; total 22.6169 s; peak RAM 5383 MB; load 25.2017 s |

## Evidence

- Baseline: `results/camera_v2-2~2026-10-05/analysis.json`,
  `analysis_tegrastats.txt`, `report.md`.
- Camera-only: `results/camera_setup_20261021T202315853338873_20s/analysis.json`,
  `camera.avi`, `capture_timestamps.csv`, `tegrastats.txt`.
- Latest model-first failure:
  `results/camera_live_20261021T203203283424919_60s/analysis.json` and sibling
  `_terminal.log`. Load 26.0985 s; model footprint 1.8969 GiB; no inference.
- Camera-first partial run:
  `results/camera_first_20261021T203530886106301_60s/analysis.json`,
  `tegrastats.txt`, and sibling `_terminal.log`. Partial status `model_ready`
  does not mean experiment success. Model load 21.0089 s; footprint 1.8969 GiB.
  Final graceful summary unavailable at the time of inspection.
- Saved-image control:
  `results/saved_control_20261021T204359206238508/control.json`,
  `tegrastats.txt`, copied `frames/`, and sibling `_terminal.log`.
  `analysis.json` was not created because loading failed before the baseline
  analyzer's first results write. The control manifest records the failure.

## Interpretation

The saved-image control retained the pinned model revision, established loading
workaround, and dependencies. It never opened the camera and still failed with:
`NVML_SUCCESS == r INTERNAL ASSERT FAILED`, `CUDACachingAllocator.cpp:1131`,
preceded by NvMap allocation errors. This prevents attributing all failures to
camera buffering or concurrency. The assertion alone does not establish the
underlying cause. Memory availability/fragmentation, driver/allocator behavior,
and unreleased resources from an earlier stalled process remain possibilities;
full host process state was not observable from the assistant sandbox.

Control telemetry began at RAM 2376/7620 MB, `lfb 87x4MB`, and ended at
4393/7620 MB, `lfb 3x1MB`; swap rose from 71 to 94 MB. This is evidence of
changing allocatable blocks, not proof of a sole root cause or exhausted total
RAM. Runtime approximately 27.06 s is failed-attempt wall time, **not** successful
model load time or inference latency.

## Scope and limitations

Model `moondream/moondream-2b-2025-04-14-4bit`, revision
`a89c59223ef8b5bb7826780728eeec172727ca84`. New experiments use max_tokens=32,
temperature=0 and the exact prompt preserved in run JSON:

> Identify clearly visible construction hazards in one short sentence. If none are clear, say so. Do not infer movement or hidden hazards.

No dependencies were upgraded. The original scripts and benchmark evidence were
preserved. The single-worker/latest-frame implementation passed CPU-only tests;
those tests do not demonstrate hardware concurrency. Generated hazard claims
from older benchmarks remain unverified. Independent images do not establish
temporal event detection. Camera-only performance does not predict combined
performance; host delivery timestamps do not measure exposure latency.

## Next session

Stop unchanged benchmark retries. First establish the failed/stalled process is
gone using host-side process inspection, and capture startup/allocator state.
Investigate model-only loading against the known-working environment before
returning to concurrent capture. Bound native blocking operations with external
supervision; the current thread-based graceful-stop handler cannot interrupt
blocked driver calls. Do not change dependencies or present a successful
concurrent benchmark without a completed hardware run.


## Fresh-restart control failed; host snapshot identifies headroom candidate

Inspected `results/postrestart_control_20261021T205228343334195/control.json`
and raw telemetry. Model-only load failed again; exit status 1, zero answers,
peak RAM 4421 MB (94 samples). Uptime at launch ~1 minute; startup RAM available
5191 MiB, swap unused. Tegrastats first/final lfb 6x4MB / 4x4MB. This run does
not show a monotonic fall to 1MB blocks; earlier fragmentation evidence must not
be assumed to describe every failure.

Host startup snapshot lists codex processes RSS 516920, 382160, and 94612 KiB
(~970 MiB summed RSS, not unique/reclaimable memory because sharing can overlap),
plus desktop/background services. Thus a reboot did not produce an app-free
benchmark environment. Codex is a concrete memory consumer worth excluding;
this does not prove it caused allocation failure. Ollama is present with ~32MB
RSS; this is not evidence of a loaded large model.

Primary-source check: PyTorch v2.8.0 upstream CUDACachingAllocator.cpp calls
`reportProcessMemoryInfo` after allocation attempts fail with
`cudaErrorMemoryAllocation`. That diagnostic helper asserts `NVML_SUCCESS == r`
while querying compute processes. Consequently a failed NVML diagnostic can
mask an underlying memory-allocation failure. This is an inference from matching
upstream code, not a verified source-line mapping for NVIDIA's installed build
(the installed assertion line differs). Source:
https://raw.githubusercontent.com/pytorch/pytorch/v2.8.0/c10/cuda/CUDACachingAllocator.cpp

Recommended isolated comparison, only if the user wants another attempt: save
this context, fully quit the Codex app normally, keep a standalone system terminal
open, and run `bash experiments/run_saved_control_after_restart.sh` with camera
closed. Do not reopen Codex until the control finishes. Same dependencies and
settings; unique outputs. This changes a specific observed memory consumer
rather than repeating a restart. Success is not guaranteed. Assistant has not
closed any app, killed processes, launched GPU work, or changed dependencies.


## Successful saved-image control — October 21, 20:57 UTC

Confirmed actual saved evidence in
`results/postrestart_control_20261021T205726708981754/`: `control.json` and
`analysis.json` both `finished`; sibling `_exit_status.txt` is 0. All three
requested images completed. This supersedes earlier statements that no new
inference succeeded today. The intervening `postrestart_control_20261021T205501961838419`
failed with BrokenPipeError before model loading; it is a distinct failed attempt.

| Frame | Original video position | Inference | Exact returned answer |
|---|---:|---:|---|
| 1 | 5.0272 s | 11.8325 s | A small excavator is stuck in muddy soil. |
| 3 | 25.0224 s | 4.9682 s | A small excavator is stuck in a ditch. |
| 5 | 45.0183 s | 5.8162 s | The excavator is in a precarious position on a slope, indicating potential instability or uneven terrain. |

Original video positions/capture timestamps refer to October 5 historical
images, not live observations today. Outputs above are generated claims, not
manually verified accuracy. Image SHA-256 comparisons confirm copied inputs
are byte-identical to baseline Video 1 frames 1, 3 and 5.

### Performance and memory

- Completed: 3/3; mean inference 7.538950823 s; total inference 22.616852469 s.
- Median 5.8162 s; range 4.9682–11.8325 s. First query includes first-use costs.
- Mean of the two subsequent queries 5.3922 s; only two samples, not an
  established steady-state estimate.
- Model load 25.2017 s; load plus summed inference 47.8186 s.
- Whole control lifetime from recorded creation/completion: 73.4934 s, including
  imports/setup/telemetry/report overhead; not capture-to-answer latency.
- Loaded footprint 1.8968705833 GiB.
- CUDA-visible free before/after load 5.1544/3.3874 GiB; free difference
  1.7670 GiB is not an isolated model allocation measurement.
- Per-query current CUDA allocation 2.2566 GiB; peak CUDA allocation 2.4976544380 GiB.
- Whole-system sampled peak RAM 5383/7620 MB (~70.64%), 145 tegrastats samples.
  Peak swap 2 MB. Unified memory figures overlap; do not add them.
- Seven NvMap allocation-error pairs remain in the terminal log despite success;
  do not label the run warning-free. TF32 deprecation and pyvips informational
  messages also appear; no fatal exception in this successful attempt.

### Comparisons

| Metric | October 5 baseline | Successful control | Interpretation |
|---|---:|---:|---|
| Mean latency, all original 30 images | 8.3241 s | 7.5390 s on 3 images | 9.43% lower, different sample mix |
| Mean latency, matching frames 1/3/5 | 10.7375 s | 7.5390 s | 29.79% lower on identical inputs |
| Frame 1 latency | 14.6789 s | 11.8325 s | 19.39% lower |
| Frame 3 latency | 8.4386 s | 4.9682 s | 41.13% lower |
| Frame 5 latency | 9.0949 s | 5.8162 s | 36.05% lower |
| Model load | 25.1305 s | 25.2017 s | Essentially unchanged (~0.28% longer) |
| Model footprint | 1.8969 GiB | 1.8969 GiB | Unchanged |
| Peak CUDA allocation | 2.4977 GiB | 2.4977 GiB | Identical recorded value |
| Peak system RAM | 5400 MB | 5383 MB | 17 MB (~0.31%) lower; not meaningful evidence of an efficiency gain |

The baseline prompt was substantially longer and max_tokens=96; the new short
prompt uses max_tokens=32, temperature=0. Actual generated token counts, tokens/s,
energy per answer and formal answer quality are not measured. A 29.79% reduction
in latency is descriptive, not a controlled causal effect of Codex removal.
The same-prompt attempt with Codex failed, so it supplies no successful inference
latency for measuring Codex's speed effect.

### Codex/headroom evidence

Successful startup snapshot: used RAM 1385 MiB, available 5982 MiB, no codex in
the saved top-24-process list. Earlier fresh-restart failed control at 20:52:
used 2175 MiB, available 5191 MiB, codex combined RSS ~970 MiB (shared pages can
overlap). Available RAM increased ~791 MiB; used RAM decreased ~790 MiB. Both
startup snapshots report ~1 minute uptime and unset allocator overrides.
User reports quitting Codex. Successful model loading with more headroom strongly
supports Codex-associated memory pressure as an operational blocker, but this is
one before/after comparison and does not establish it was the only contributor.

### Reproducibility and scope

Model `moondream/moondream-2b-2025-04-14-4bit`, revision
`a89c59223ef8b5bb7826780728eeec172727ca84`; PyTorch 2.8.0, Transformers 4.57.6,
TorchAO 0.11.0. Exact prompt:

> Identify clearly visible construction hazards in one short sentence. If none are clear, say so. Do not infer movement or hidden hazards.

Settings max_tokens=32, temperature=0. Model dependencies and loading workaround
unchanged. This is successful **saved-image inference**, not a concurrent live
camera benchmark. Frame age and capture-to-answer delay remain unavailable for
this control. Concurrent capture/inference still needs a successful hardware run;
temporal event detection and accuracy remain unverified. Assistant only read
artifacts/calculated comparisons and updated documentation; no new hardware run.
