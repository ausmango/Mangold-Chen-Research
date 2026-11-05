# Research context — updated 2026-11-05

> **Expanded November 5 meeting script:**
> `docs/meeting-script-2026-11-05-live-concurrency.md` provides a ~10–13-minute
> read-aloud explanation and sourced Moondream/LFM2-VL-450M/SmolVLM2 comparison.
> User prioritizes concurrency above further offline benchmark tuning. Smaller
> models may sacrifice capability; size does not determine domain accuracy.
> Recommend validating supervised latest-frame capture first, then one candidate
> at a time, while preserving Moondream as baseline. No new model performance,
> working concurrency, hardware tests or dependency changes claimed.


> **November 5 research-only update:** User had 30 minutes before leaving;
> no new benchmarks/camera/GPU inference, downloads, installs or dependency
> changes were performed. Read `docs/meeting-brief-2026-11-05.md` and
> `docs/research-log/2026-11-05-live-inference-research.md`.
> Latest measured result remains October 22: 4.4612 s mean, 46.41% lower than
> October 5 with changed task/input conditions. 87.81% is image/prefix processing;
> this includes visual encoding and text prefill, not a pure vision timer.
> Static review found blocking native camera calls and unbounded total join wait;
> proposal is independently supervised capture/UI and one inference process with
> a latest-frame mailbox. Process separation does not solve physical RAM limits
> by itself and has not been implemented/validated as a fix.
> Model shortlist: original LiquidAI/LFM2-VL-450M (card supports Transformers
> 4.57+, 86M encoder, visual token controls) first live-image candidate;
> HuggingFaceTB/SmolVLM2-500M-Video-Instruct for multi-frame/video capability;
> SmolVLM2-256M lower-capacity control. Newer LFM2.5-VL-450M requires 5.1+ per
> card, so defer without changing the working 4.57.6 stack. All untested locally.
> Cached Moondream source builds BF16 ordinary vision linear layers; “4-bit”
> does not imply every vision operation is INT4. Existing KV caching is not new.
> Proposed next milestone is a small-model supervised two-minute LIVE trial with
> at least five useful answers, not another record-first primary benchmark.
> No commits/pushes. Old benchmark evidence remains unchanged.


> **COMPLETED October 22 inference:** User-run
> `results/camera_analysis_20261022T163529601772802` finished 30/30 compact16
> queries, exit 0, using preserved October 22 CSI images (all hashes verified).
> Mean 4.4612 s vs October 5 8.3241 s: **46.41% lower latency**, **1.866x speed**
> (**86.59% greater serial throughput**). Total 133.8347 s vs 249.7241 s.
> Mean encoding 3.9174 s (87.81%); query 0.5437 s (12.19%). First query
> 11.3289 s; remaining 29 mean 4.2243 s. Load 25.3014 s, footprint 1.8969 GiB,
> peak CUDA 2.4977 GiB, peak whole-system RAM 5401 MB (358 samples).
> New shorter outputs are often object labels, not verified hazard explanations;
> fresh captures/different prompt prevent controlled causal attribution.
> Final report with all 30 answers:
> `docs/research-log/2026-10-22-camera-benchmark-comparison.md`.
> Capture was earlier, inference resumed later: actual capture-to-answer mean
> 930.32 s includes offline waiting, NOT real-time responsiveness. Original
> timed-out attempt preserved. Concurrent allocation remains unresolved.


> **Next authorized user action:** User wants to finish inference on today's
> saved 30 images. Prepared `bash experiments/analyze_recorded_frames.sh`.
> Close Codex and run in a standalone terminal; no restart or video playback
> required. Copies only saved images/timestamps/manifest into a fresh
> `results/camera_analysis_<UTC>/` attempt, preserves original recordings and
> timed-out analysis. Loads pinned model, compact16 only, 900-second timeout
> independent of expired noon deadline. Syntax/source validation passed;
> assistant did not launch model or camera. Inspect new results then update
> October 22 report with actual answers/performance percentages.


> **Latest verified outcome, October 22:** Run
> `results/camera_fast_20261022T161952378124366` completed all five recordings,
> 4500 frames and 30 JPEG samples; capture exit 0, weighted FPS 30.0012.
> Inference did NOT complete: analysis exit 124 at 12:24 while model loading,
> `analysis_fast.json` frames empty and stale `initializing`. Only ~46 seconds
> remained for analysis after captures. Do not claim a new latency gain or 30
> completed answers. Report: `docs/research-log/2026-10-22-camera-benchmark-comparison.md`.
> Valid prior 3-image control improvement is 29.79% lower matched-image latency,
> not a new 30-image result. Reuse captured images later rather than refilming;
> preserve original partial results. No new hardware run launched by assistant.


> **Urgent October 22 12:17 update:** User accepted October 5 sequential workflow
> after concurrent Argus allocation failed again. Capture stalled at old terminal
> metadata prompts; removed all metadata entry by automatically supplying the
> five baseline names/source/durations. `run_camera_fast_benchmark.sh` now uses
> foreground timeout for capture and **compact16 only (30 queries)** for remaining
> time. Run `bash experiments/run_camera_fast_benchmark.sh 12:24` after stopping
> old attempt. Only press S in preview per video. Reference short32 comparison
> can be run later on the saved images; this urgent run is not paired. Syntax,
> shell syntax and automatic-five-preset validation passed; no hardware launched.


> **October 22 corrected active task:** User clarified actual concurrent CSI
> recording/preview and latest-frame inference, not capture-then-analyze.
> Use `bash experiments/run_live_prompt_compare.sh 12:23` after restart with
> Codex closed. Five familiar clips; one serial worker compares yesterday's
> short32 with compact16 on the same latest frame, encoding once and reusing
> existing same-image prefix cache. No new KV cache enabling or cross-frame reuse.
> Optional second argument `candidate-only` measures freshest single-prompt mode.
> Runbook: `docs/live-prompt-compare-runbook.md`. External timeouts guard native
> hangs, no automatic retries. All 13 CPU-only tests and syntax checks pass.
> No hardware run launched or performance gain measured. Log measured findings
> only after user confirms actual results. Older offline runbook is superseded
> for this task; six completed observations per short clip are not guaranteed.



> **Latest result:** `results/postrestart_control_20261021T205726708981754`
> succeeded, 3/3 saved images, exit 0. Mean inference 7.5390 s; matched-image
> baseline 10.7375 s (29.79% lower). Load 25.2017 s; footprint 1.8969 GiB;
> peak CUDA 2.4977 GiB; peak system RAM 5383 MB. User reports Codex closed;
> startup available RAM was 791 MiB higher than the failed control. Shorter
> prompt/32-token cap confound latency attribution. See detailed successful-run
> section below. Camera-only succeeded separately; concurrent operation remains
> unproven. No further hardware run authorized or launched by assistant.


## Purpose and working environment

Local construction-scene analysis on NVIDIA Jetson Orin Nano, with approximately
8 GB unified memory and a CSI camera filming construction scenes/video playback.
No sudo access. Preserve the installed PyTorch/CUDA/model dependencies.
Read [repository instructions](../AGENTS.md) and [research history](research-log/)
before further experiment work. Do not commit or push without a request.

## Baseline evidence

The October 5 sequential capture/analysis benchmark completed **30 frames** from
five videos. Mean inference: **8.3241 seconds**; loaded model footprint:
**1.8969 GiB**; sampled peak whole-system RAM: **5400 MB** of 7620 MB.
Model load: 25.1305 s; peak PyTorch CUDA allocation: 2.4977 GiB.

- History: `docs/research-log/2026-09-21-moondream-benchmarks.md`,
  `2026-09-28-moondream-live-camera.md`, and
  `2026-10-05-moondream-camera-benchmark.md` in the same directory.
- Sequential implementation: `experiments/moondream_4bit_camera_v2.py`.
- Baseline artifacts: `results/camera_v2-2~2026-10-05/analysis.json`,
  `analysis_log.txt`, `analysis_tegrastats.txt`, `capture.json`, `report.md`,
  and `video_01/` through `video_05/`.
- Previous failures include repetitive unconstrained generation, camera/memory
  allocation warnings, and a CUDA allocator `NVML_SUCCESS` assertion. Preserve
  their raw logs and partial results.

## Current concurrent implementation

`experiments/moondream_4bit_camera_live.py` is separate from previous experiments.
User-run hardware attempts on October 21 have now been inspected: one model-load
failure and one camera-session failure after model load; no analyzed images yet.
Do not launch another capture or inference run without user readiness confirmation.

Model: `moondream/moondream-2b-2025-04-14-4bit`.
Revision: `a89c59223ef8b5bb7826780728eeec172727ca84`.
The existing `caching_allocator_warmup` no-op and CUDA device mapping are retained.
CSI pipeline matches v2: sensor 0 at 1280×720/30 FPS, converted to 640×360 BGR,
with `appsink drop=true max-buffers=1 sync=false`.

Main thread owns capture and OpenCV windows. Exactly one worker owns the model
and all queries. A condition-protected slot retains only the newest delivered
frame, timestamp, and sequence; one selected frame can be in flight separately.
There is no accumulated queue and no concurrent query on mutable model caches.

Prompt:

> Identify clearly visible construction hazards in one short sentence. If none are clear, say so. Do not infer movement or hidden hazards.

Settings: `max_tokens=32`, `temperature=0`. No extra caption or warmup inference.
KV caches already exist in the pinned model's cached remote code; they are not
newly enabled. See the exact source references in
`docs/research-log/2026-10-21-moondream-concurrent-camera.md`.

The worker saves selected PNG frames before inference, atomically updates
`analysis.json`, and saves each completed answer in a JSON sidecar. Main writes
`capture_timestamps.csv`. Metadata records exact model/revision/prompt/settings,
model load time/footprint, CUDA allocation/peak, query latency, frame age at query
start, and delivery-to-completed-answer delay. Query rows record capture sequence
counts at start/completion to verify that capture advances during inference.

The script itself launches `tegrastats --interval 500` before loading, saves raw
output to `tegrastats.txt`, and terminates only its own telemetry subprocess after
worker drain. Final JSON summarizes whole-system peak RAM and sample count.
Missing telemetry is flagged explicitly. No separate telemetry command or sudo
is needed.

## Checks and review

CPU-only checks: `experiments/tests/test_moondream_camera_live.py`.
Syntax compilation, CLI help, and tests pass; these do not establish hardware
concurrency or performance. Review corrected a JSON write inside the inference
timer, premature initialization-stop return status, and cleanup handling that
could prevent final output preservation. Regression checks cover these failure
paths as well as latest-frame replacement, timing identity, frame/result
persistence, and RAM parsing.

## Known limitations and interpretation

- Hardware initialization measurements and failures now exist (see below), but
  no inference outputs or timed camera capture measurements exist yet.
- Periodic live image analysis is not verified temporal event detection.
  Generated hazard claims are not verified accuracy. Manual image review and
  ground truth remain necessary; brief hazards may be missed entirely.
- Frame timestamps measure host delivery after `camera.read`, not exposure;
  sensor/ISP/buffering delay is unmeasured. Inference includes image encoding
  and decoding; frame age includes PNG persistence and PIL preparation.
- First query includes first-use overhead. Shorter prompt/token cap and different
  scenes mean baseline comparisons cannot isolate the effect of concurrency.
- With `--wait-for-start`, alignment preview follows model load; press S, then
  immediately start video playback. Duration begins at the first delivered timed
  frame after S. Alignment frames are neither analyzed nor recorded. Without
  this option, duration begins at the first delivered frame after model load. Q, Ctrl+C,
  SIGTERM, or duration expiry stop capture and new work, then drain a current
  load/query. A 60-second capture can have a longer total process lifetime.
- Blocking camera/driver/GPU operations are not cancellable with a hard timeout.
  Memory pressure and native warnings must be assessed on actual hardware.
- RAM is whole-system, sampled every 500 ms. CUDA/model/system memory overlap
  on Jetson; never add them as separate physical pools.
- Working graphical display is required for preview. No dependency changes,
  vLLM installation, or system-stack upgrades are authorized.

## Next test — only after camera readiness confirmation

From `/home/charlotte/research`, in Bash with a working display:

```bash
set -o pipefail
run="results/camera_live_$(date -u +%Y%m%dT%H%M%S%N)_60s"
python3 -u experiments/moondream_4bit_camera_live.py --run "$run" --duration 60 --wait-for-start --source-label "Video 1 - Red Machine (October 5 baseline)" 2>&1 | tee "${run}_terminal.log"
```

This includes internal tegrastats monitoring at 500 ms. The timestamp produces
a unique result folder; existing directories are rejected. The sibling terminal
log preserves Python and native Argus/CUDA messages. Use familiar Video 1,
“Red Machine” (60 seconds), from the October 5 baseline. Pause it at the start,
wait for model load and preview, align the camera, press S in preview, then
immediately start playback. Timer starts at the next delivered frame, with a
small unmeasured manual playback offset. Q stops capture and drains the query.

After the test, inspect `analysis.json`, all selected PNG/JSON pairs,
`capture_timestamps.csv`, `tegrastats.txt`, and the terminal log. Report completed
counts, actual answers, loading/footprint, inference latency, frame age,
capture-to-answer delay, CUDA allocation/peak, whole-system RAM, capture gaps,
warnings, and whether capture sequences advanced during queries. Preserve
failures/partial results and update this file plus the dated research note.


## Metrics audit and camera-only recording check

The audit added explicit `camera_capture.measured_fps`, delivery count/span,
maximum delivery gap, and mean camera read time, calculated from saved timed
CSV rows as `(N - 1) / (last_timestamp - first_timestamp)`. Rates exclude alignment,
model loading, and worker drain. `capture_wall_seconds` now excludes camera cleanup.
Each answer sidecar also includes exact model/revision/prompt/settings.

Saved fields in `analysis.json`:

| Requirement | Field |
|---|---|
| Loaded model footprint | `model_footprint_gib` |
| Model loading time | `load_seconds` |
| Per-query inference latency | `frames[].inference_seconds` |
| Frame age at query start | `frames[].frame_age_at_inference_start_seconds` |
| Delivery-to-answer delay | `frames[].capture_to_completed_answer_seconds` |
| Actual host-delivered camera rate | `camera_capture.measured_fps` |
| CUDA peak allocation, including load | `cuda_peak_allocated_gib` and per-query peaks |
| Sampled whole-system RAM peak | `system_ram.peak_used_mb`, with raw `tegrastats.txt` |

Timing fields also have mean/min/max summaries. `frames/` holds each selected
lossless PNG and completed JSON sidecar with timestamp/sequence/answer. Top-level
JSON retains source label, model revision, prompt, and generation settings.

For a 20-second camera/preview/recording check without model loading:

```bash
set -o pipefail
run="results/camera_setup_$(date -u +%Y%m%dT%H%M%S%N)_20s"
python3 -u experiments/moondream_4bit_camera_live.py --camera-only --wait-for-start --run "$run" --duration 20 --source-label "Video 1 - Red Machine (camera setup check)" 2>&1 | tee "${run}_terminal.log"
```

This has the same pipeline and internal tegrastats sampling. Press S after
alignment, then start playback. It saves `camera.avi` (MJPEG, nominal 30 FPS),
`setup_first_frame.png`, timestamp CSV, `analysis.json`, and telemetry. Recording
statistics report submitted frames and mean/max write times; writer calls do
not prove encoded-frame integrity. Review the AVI/still for focus, glare,
framing, readable details and flicker. Compare measured delivery FPS with the
requested 30 FPS and inspect gaps/errors for stalls. AVI playback uses nominal
30 FPS, so use the CSV for actual time if capture is slower. Camera-only success
does not establish performance under simultaneous model memory/compute load.
It has no model footprint/load/inference/CUDA metrics because it never loads
Moondream. The normal inference mode saves selected PNGs, not a full recording.

Six CPU-only tests now pass, including a stand-in camera-only alignment/recording
run that verifies the model worker is never invoked. Syntax/help checks pass.
Those checks launched no hardware. Subsequent user-run hardware attempts are
documented below; the assistant has not launched a hardware test.


## Actual hardware attempts inspected — October 21, 2026

- `results/camera_setup_20261021T201423867363688_20s/analysis.json`
  and sibling `_terminal.log`: camera-only Argus producer connected, then run
  stopped during alignment (`alignment_stopped`). No timed frames/recording;
  peak system RAM 2740 MB from 38 telemetry samples. This establishes camera
  initialization/connection, not measured recording throughput or image quality.
- `results/camera_live_20261021T201509182834357_60s/analysis.json`:
  failed during model loading with PyTorch `NVML_SUCCESS == r` allocator assertion;
  no model load time/footprint or inference results. Peak system RAM 3606 MB,
  61 samples. This resembles a previously documented allocator failure.
- `results/camera_live_20261021T201900166974762_60s/analysis.json`
  and sibling `_terminal.log`: model loaded in 26.3344 s, footprint 1.8969 GiB,
  CUDA allocated/peak 1.9081 GiB. CUDA-visible free memory 4.4866 GiB before
  load, 2.2897 GiB afterward. Repeated NvMap allocation errors preceded Argus
  `Failed to create CaptureSession`; alignment read failed. Zero timed frames,
  zero answers; inference/age/delay/FPS unavailable, not zero. Peak whole-system
  RAM 5206/7620 MB (96 samples). Initial telemetry `lfb 5x4MB`; final
  `lfb 28x1MB`, with swap 11/8192 MB at final sample.

Memory pressure/fragmentation is a plausible contributor, not a confirmed root
cause. NVIDIA documents lfb as largest allocatable contiguous physical blocks;
it is distinct from total free RAM. Reference:
https://docs.nvidia.com/jetson/archives/r36.5/DeveloperGuide/AT/JetsonLinuxDevelopmentTools/TegrastatsUtility.html

Successful October 5 sequential inference began with 5.1728 GiB CUDA-visible
free and retained 3.3684 GiB after load; latest failed combined run had less
headroom. Its 5206 MB RAM peak does not mean all camera allocations could succeed.
Host process inspection from the assistant's sandbox exposed only sandbox
processes, so it cannot rule out other host camera clients or memory consumers.

Next: close unused applications/camera clients manually, then run the existing
20-second camera-only command above, press S, and complete the recording. Inspect
AVI, timed FPS/gaps and telemetry. If camera-only also fails, investigate Argus
state/camera connection independently; a normal desktop restart may help reset
state but is not a confirmed fix. If camera-only succeeds but the combined run
fails, investigate initialization ordering/memory headroom next while preserving
model dependencies and pipeline. No code/dependency changes made for this diagnosis.


## Camera-first revision after repeated combined failure

Actual completed camera-only run:
`results/camera_setup_20261021T202315853338873_20s/analysis.json`:
596 frames in 19.9960 s between first/last deliveries, 29.7559 FPS, max gap
0.2023 s, mean recording write 0.0170 s, peak RAM 3456/7620 MB (62 samples).
This establishes timed camera/recording throughput without the model, not image
quality or inference performance.

Repeated combined run:
`results/camera_live_20261021T203203283424919_60s/analysis.json`:
model loaded in 26.0985 s, footprint 1.8969 GiB; CUDA-visible free memory
4.5308 GiB before load and 2.4161 GiB after load. Peak whole-system RAM
5083 MB (95 samples). Argus CaptureSession creation failed after NvMap
allocation errors; zero delivered timed frames/answers. The user reports two
restarts and no other applications open. Background apps are not established
as the cause. Sequential baseline did not keep the model resident with capture.

Changed initialization to verify actual CSI frame delivery **before** starting
the model worker, reserving camera resources first. Main continues camera reads
and preview while loading, then optional S-to-start alignment. Pre-test frames
are not published to inference and do not start the timer. Revision metadata
records `initialization_order=camera_verified_before_model`, camera verification
UTC, and frames delivered during loading. Pipeline, model/revision, dependencies,
query settings, single-worker/latest-frame design, and output metrics remain
unchanged. The previous model-first allocation order was an implementation
choice that did not work on the observed combined attempts.

The camera-first change is a focused hypothesis test, not a demonstrated fix.
Seven CPU-only tests, syntax checks, and whitespace checks pass, including
camera verification before model work and preventing model loading if the camera
fails verification. No new camera or model run was launched. Use the same
60-second command above; pause Video 1, wait for camera-first model loading and
alignment, press S, then start playback. If loading or inference fails, retain
that partial attempt; do not spend the remaining session repeating restarts.
The completed camera-only recording and documented combined failures are valid
meeting evidence, but not a successful concurrent inference benchmark.


## Camera-first attempt stalled — do not repeat unchanged

Run `results/camera_first_20261021T203530886106301_60s` verified the camera,
then its terminal log reported Argus `InsufficientMemory` in image initialization
and `copyToNvBuffer`, followed by `IImageNativeBuffer not supported by Image`
and repeated NvMap allocation failures. Camera-first ordering did not solve the
combined allocation problem. At inspection, partial `analysis.json` said
`model_ready`, load 21.0089 s, footprint 1.8969 GiB, and no analyzed frames.
This is a stale intermediate status, not successful completion. Telemetry at
inspection had 379 RAM samples with an observed peak of 5043 MB; telemetry was
still potentially growing, so this is not a finalized run peak.

User reports repeated Ctrl+C could not end the process. The signal handler only
sets a stop flag; blocked native camera reads/cleanup and worker join are not
bounded. A forced stop targeting this experiment and its tegrastats child was
provided to the user; the assistant cannot see host PIDs from its sandbox and
did not execute a kill. Already saved logs remain; graceful final metrics may
be unavailable. Do not overwrite this partial run or treat it as a completed
60-second test. No inference latency/event accuracy was measured.

Current practical outcome: completed camera-only 29.7559 FPS benchmark plus
reproducible combined initialization failures. Further concurrent tests need
bounded native-operation supervision and investigation of camera/model shared
allocation pressure; no further unchanged retries or dependency changes are
recommended. Camera-first is an unsuccessful attempted mitigation, not a fix.


## Time-bounded saved-image control prepared

To obtain a current inference measurement without the failed camera/model
combination, prepared `experiments/moondream_4bit_saved_control.py`. It copies
baseline Video 1 samples 1, 3, and 5 to a unique folder and reuses the unchanged
v2 loading/query path, with the live prompt and max_tokens=32/temperature=0.
It starts tegrastats, saves each completed inference via v2, and writes
`control.json`/`analysis.json`/`report.md`. Original image timestamps are provenance;
frame age and capture-to-answer delay are unavailable for this offline control.
This is not a replacement claim of concurrent benchmark success.

Preparation-only validation copied three real saved images in a temporary folder;
syntax and help checks passed. No camera, model inference, or telemetry was
launched by the assistant. Before running, stop the hung live process and its
tegrastats child. Command from repository root:

```bash
set -o pipefail
run="results/saved_control_$(date -u +%Y%m%dT%H%M%S%N)"
timeout --signal=TERM --kill-after=5s 180s python3 -u experiments/moondream_4bit_saved_control.py --run "$run" 2>&1 | tee "${run}_terminal.log"
```

The external deadline bounds the attempt to approximately 185 seconds; timeout
termination can skip final summaries, but completed frame results and raw logs
already written remain. Inspect actual output before reporting any new result.
No new measured inference performance exists until the user runs this control.


## Saved-image control failed too — latest conclusion

Inspected `results/saved_control_20261021T204359206238508/control.json`, raw
telemetry, and the user-provided traceback. Model loading failed without opening
the camera: NvMap errors followed by CUDA allocator `NVML_SUCCESS == r` assertion
at `CUDACachingAllocator.cpp:1131`. No inference completed; `analysis.json` was
never created. Peak RAM 4393/7620 MB from 53 samples. Initial/final lfb
87x4MB / 3x1MB; swap 71/94 MB. Approximate 27.06 s is failed-attempt wall time,
not model load or inference time. Do not infer that concurrency alone caused
all failures or that the assertion proves exhausted total RAM.

Prepared `docs/research-log/2026-10-22-meeting-summary.md` from actual artifacts:
completed camera-only throughput, prior sequential baseline, all combined
failures, and model-only failure. No new inference outputs exist today. Stop
unchanged retries for now; next investigation must verify host process/resource
state and model-only loading before attempting concurrency. No extra hardware
run or dependency change was performed by the assistant.


## Final post-restart attempt handoff

User explicitly chose one last run after restart; earlier recommendations to
stop testing are superseded by that choice. Do not run it on the user's behalf
or launch the live camera. No success is guaranteed by restarting.

Launcher: `experiments/run_saved_control_after_restart.sh`.
Validation: `bash -n` passed; saved-control Python syntax passed. The launcher
was not executed, so no model/camera/tegrastats run was launched by the assistant.
It preserves existing environment settings and dependencies; it does not tune
allocators, clear system caches, change swap, or use sudo.

After restarting, open a terminal, leave camera previews/video playback closed,
and run:

```bash
cd /home/charlotte/research
bash experiments/run_saved_control_after_restart.sh
```

Outputs use unique `results/postrestart_control_<UTC timestamp>` names:

- Sibling `_startup.txt`: uptime, RAM/meminfo, host process names/RSS, selected
  allocator/device environment settings; collected before model import.
- Sibling `_terminal.log` and `_exit_status.txt`: full output and Python/timeout
  status (0 only when the control reports all three frames finished).
- Run folder: copied images, `capture.json`, `control.json`, raw `tegrastats.txt`,
  and, if load succeeds, `analysis.json`/`report.md` with incremental answers.

`timeout` sends TERM at 180 seconds and KILL five seconds later if needed;
uninterruptible kernel/driver operations may still delay process exit. A forced
termination may leave stale `running` status or omit final summaries; inspect
exit status and completed rows rather than trusting status alone.

Next assistant: read this top handoff and actual newest outputs, report completed
frame count, load time/footprint, latency/answers and RAM/CUDA metrics that exist,
then update the dated research note and meeting summary. If it fails, document
its exact stage/error and startup evidence; do not invent successful results or
silently change dependencies. Do not commit/push. Original target remains a
concurrent CSI/Moondream benchmark; this control first establishes whether model
loading/inference can work after restart without camera contention.


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
