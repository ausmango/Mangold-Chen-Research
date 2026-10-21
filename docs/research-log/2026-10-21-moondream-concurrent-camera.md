# Concurrent CSI camera and Moondream — 2026-10-21

## Status

Implementation and CPU-only checks completed. Subsequent user-run hardware
attempts failed before timed capture/inference; actual saved evidence is recorded
below. No inference latency, throughput, or model-output results exist yet.
The assistant has not launched a hardware test; further runs require readiness.

## Baseline and scope

October 5: 30 saved frames from five videos, sequential acquisition and analysis;
mean inference 8.3241 s, model footprint 1.8969 GiB, peak whole-system RAM
5,400/7,620 MB. See [baseline note](2026-10-05-moondream-camera-benchmark.md).

The separate [live experiment](../../experiments/moondream_4bit_camera_live.py)
preserves existing scripts/results and dependencies. It retains model
`moondream/moondream-2b-2025-04-14-4bit`, revision
`a89c59223ef8b5bb7826780728eeec172727ca84`, CUDA device mapping, and the
`caching_allocator_warmup` no-op loading workaround. The CSI pipeline is the same
as v2: Argus sensor 0, 1280×720 at 30 FPS, conversion to 640×360 BGR,
`appsink drop=true max-buffers=1 sync=false`.

Main owns camera reads, preview, key handling, and OpenCV cleanup. One worker
loads the model and performs all queries serially. A condition-protected slot
holds only the latest delivered frame plus sequence, host monotonic timestamp,
and UTC timestamp. The worker retains its selected frame during processing;
new arrivals replace the pending slot. Intermediate frames are deliberately
skipped. No history of images or encoded-image caches is supplied to a query.

Exact prompt:

> Identify clearly visible construction hazards in one short sentence. If none are clear, say so. Do not infer movement or hidden hazards.

Exact settings: `{"max_tokens": 32, "temperature": 0}`. No caption or warmup query
is issued. The first measured query includes first-use costs. Changing the prompt
and token cap from the baseline prevents attributing latency changes solely to
concurrency.

## Existing KV cache inspection

Inspected local cached source under
`~/.cache/huggingface/modules/transformers_modules/moondream/moondream_hyphen_2b_hyphen_2025_hyphen_04_hyphen_14_hyphen_4bit/a89c59223ef8b5bb7826780728eeec172727ca84/`:

- `moondream.py:47`: `KVCache` allocates K/V buffers and updates positions in place.
- `moondream.py:139`: `_setup_caches` attaches a cache to each text block.
- `moondream.py:207`: `encode_image` prefills image tokens and clones prefix caches.
- `moondream.py:356`: `query` encodes the image, restores its prefix cache, then generates.
- `text.py:41`: attention calls `kv_cache.update`.

KV caching already exists in this model. This experiment does not newly enable
it or introduce cross-frame cache reuse. Mutable model caches are one reason to
use exactly one inference worker.

## Timing and artifacts

A frame timestamp is recorded immediately after `camera.read()` returns. It is
**not a sensor exposure timestamp**; upstream buffering and sensor/ISP latency
are unmeasured. All elapsed measurements use the host monotonic clock.

- Inference latency: synchronized query start to synchronized completion;
  includes model image encoding and decoding, excludes frame saving/PIL conversion.
- Frame age at inference start: query start minus delivery timestamp; includes
  selected-frame persistence and image preparation.
- Capture-to-completed-answer delay: synchronized completion minus delivery
  timestamp. Excludes subsequent result-file writes.

The selected image is saved losslessly before its query. `analysis.json` is
atomically replaced at load, selection, each completion, worker
failure, and finalization. Each completed answer also has a frame JSON sidecar.
`capture_timestamps.csv` is flushed per delivered frame and permits capture-rate
and gap analysis. Query rows contain start/completion capture sequence counts,
allowing the live test to establish whether capture progressed during inference.

The script starts its own `tegrastats --interval 500` subprocess before loading
and stops only that child after draining the worker. `tegrastats.txt` retains raw
output. The final JSON includes sample count and sampled peak whole-system RAM,
model load time/footprint, CUDA allocation after loading and per query, and peak
CUDA allocation including load and inference. CUDA and system RAM overlap on
Jetson unified memory; do not add these values. Telemetry without RAM samples
is explicitly flagged as missing.

Duration starts at the first delivered frame, after model loading/camera open.
Q, Ctrl+C, SIGTERM, and duration expiry stop acquisition/new work and wait for
an in-flight load/query to finish, preserving its result. Worker errors print a
traceback and return failure; selected frames and prior results remain saved.
The duration is not a hard process timeout: blocking camera/driver calls or a
hung GPU operation cannot be cancelled safely by this thread-based prototype.
The final query can finish after capture has stopped. Whole-system RAM covers
loading, capture, and this drain period.

## Checks completed

- `python3 -m py_compile experiments/moondream_4bit_camera_live.py experiments/tests/test_moondream_camera_live.py` — passed.
- `python3 experiments/moondream_4bit_camera_live.py --help` — passed; imports no camera/GPU libraries.
- `python3 -m unittest discover -s experiments/tests -v` — 6 passed.
- `git diff --check` — passed.

CPU-only stand-ins exercised a blocked query while 99 newer frames arrived:
next selected sequence was 100, with 98 skipped frames between queries.
Checks also covered separate timing identities, persistence before completion,
query failure/traceback preservation, and tegrastats peak parsing. These checks
are implementation validation, not evidence of hardware concurrency.

## Short live test command (only after readiness)

Run from `/home/charlotte/research`, in a terminal with a working graphical
display. Pause baseline Video 1 (Red Machine) at its beginning. After model
loading, align the preview, press S, and immediately start playback. Use a new `--run` path for every attempt; existing run directories
are rejected. This command generates a unique run name:

```bash
set -o pipefail
run="results/camera_live_$(date -u +%Y%m%dT%H%M%S%N)_60s"
python3 -u experiments/moondream_4bit_camera_live.py --run "$run" --duration 60 --wait-for-start --source-label "Video 1 - Red Machine (October 5 baseline)" 2>&1 | tee "${run}_terminal.log"
```

The terminal log captures Python output and native CUDA/Argus warnings. No
external tegrastats invocation is needed. Q closes capture and drains the query.

## Post-test work still required

Inspect `analysis.json`, frame PNG/JSON files, `capture_timestamps.csv`, terminal
log, and `tegrastats.txt`. Update this dated note with actual completed/failed
counts, model loading/footprint, CUDA allocation/peak, system RAM peak/sample
count, capture rate/gaps, per-query latency/age/delay, actual answers, and warning
messages. Check that capture sequences advance during queries; inspect image
claims manually and preserve any failed/partial attempt. Treat a missing metric
as unavailable, never as zero. Make baseline comparisons descriptive because
prompt, token limit, scenes, and background memory differ.

This is periodic live image analysis, **not verified temporal event detection**.
It can miss brief hazards between analyzed frames, infer unsupported details,
and truncate answers. Accuracy, false positives/negatives, temporal reasoning,
and event recall remain unmeasured. Continuous preview and bounded buffering
alone do not establish real-time hazard detection.


## Pre-test review follow-up

Created root `AGENTS.md` and `docs/context.md`; neither previously existed.
Review corrected a JSON write inside the inference timing interval, a premature
return that could mask an initialization worker failure, and OpenCV cleanup
exceptions that could prevent final artifact preservation. Missing telemetry
files now produce explicit unavailable-memory metadata. Worker final persistence
errors are printed, stop the run, and always release the readiness wait.
Five CPU-only regression checks pass, including initialization failure exit status
and cleanup/telemetry failures. No capture or model inference was launched.


## Metrics audit and setup-test addition

Added explicit camera delivery FPS/count/span/read-time/gap summary using the
saved timestamp CSV, excluding alignment and shutdown. Corrected capture wall
time to end before cleanup. Per-frame answer sidecars now repeat exact model
revision/prompt/settings. Added optional S-to-start alignment and a source label
so the first familiar-video test can begin after framing is settled. Press S,
then start baseline Video 1 (Red Machine, 60 seconds); timer begins on the next
delivered frame. Playback alignment remains manual and approximate.

Added `--camera-only` to check actual MJPEG recording/preview with no model load,
using the same pipeline and internal telemetry. See `docs/context.md` for the
20-second command, output list, and interpretation. Submitted writer frames do
not prove video integrity; inspect the AVI and still. The inference benchmark
does not record full video. Six CPU-only tests and syntax/help checks pass.
No hardware test has run, and no new performance or hazard-accuracy claims exist.


## User-run hardware evidence: initialization failures

Inspected the saved JSON, terminal logs, and raw tegrastats for three attempts:

| Run suffix (all under `results/`) | Outcome | Peak RAM |
|---|---|---:|
| `camera_setup_20261021T201423867363688_20s` | Argus producer connected; stopped during alignment, before timed capture | 2740 MB / 38 samples |
| `camera_live_20261021T201509182834357_60s` | Model load failed: PyTorch CUDA allocator `NVML_SUCCESS == r` assertion | 3606 MB / 61 samples |
| `camera_live_20261021T201900166974762_60s` | Model loaded; Argus CaptureSession creation and alignment read failed | 5206 MB / 96 samples |

Latest run: load 26.3344 s; footprint 1.8969 GiB; CUDA allocated/peak 1.9081 GiB;
CUDA-visible free memory before/after load 4.4866/2.2897 GiB. Terminal log contains
repeated `NvMapMemAllocInternalTagged ... error 12`, then
`Failed to create CaptureSession`, then `CSI camera stopped during alignment`.
Zero timed frames and zero answers. Inference latency, frame age, delivery-to-answer
delay, and camera FPS are unavailable. The 60-second timer never started.

Raw telemetry begins with `RAM 2497/7620MB (lfb 5x4MB)` and ends with
`RAM 5206/7620MB (lfb 28x1MB) SWAP 11/8192MB`. This is consistent with pressure
and reduced contiguous-block availability, not proof of the exact failed
allocation or sole root cause. NVIDIA's [tegrastats documentation](https://docs.nvidia.com/jetson/archives/r36.5/DeveloperGuide/AT/JetsonLinuxDevelopmentTools/TegrastatsUtility.html)
defines lfb separately from total RAM. Camera/service state remains a possible
contributor. CUDA-visible post-load headroom was lower than the successful
October 5 sequential baseline (3.3684 GiB).

Next: complete a timed camera-only recording after closing unused applications
and other camera clients. Its previous stopped alignment attempt is not a FPS
benchmark. If it succeeds and combined initialization fails again, investigate
camera-first initialization/memory headroom as a separate, measured change.
No dependencies or camera pipeline changed, no processes killed, and no new
hardware run launched during diagnosis. Context updated with actual evidence.


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


## Final post-restart control prepared at user request

User chose one final attempt after restart. Prepared
`experiments/run_saved_control_after_restart.sh`: snapshots startup RAM/process
state and selected allocator/device settings, checks for existing Moondream
Python processes, then launches the three-image camera-closed control under
`timeout --signal=TERM --kill-after=5s 180s`. Unique result names preserve all
previous failures. The launcher saves startup/terminal/exit-status files beside
the run directory. No environment/dependency changes or sudo. Shell syntax and
Python syntax checks passed; launcher not executed. Context top handoff updated.
This is a pending model-only control; concurrent operation is still unproven.


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
