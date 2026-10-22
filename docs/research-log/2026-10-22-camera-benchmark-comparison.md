# CSI camera benchmark comparison — October 22, 2026

## Final verified outcome

**All five CSI recordings, all 30 sampled images, and all 30 inference queries completed.**
The initial 12:24 analysis attempt timed out during loading; the user subsequently
reran analysis on the preserved camera images without refilming. This final report
supersedes the earlier incomplete-inference summary while retaining that failure as history.

Capture run: `results/camera_fast_20261022T161952378124366` (capture exit 0).
Completed analysis: `results/camera_analysis_20261022T163529601772802`
(`status=finished`, 30/30 queries, exit 0). Thirty per-answer JSON sidecars are
present, and SHA-256 comparisons confirm analyzed images match the newly captured
October 22 images byte-for-byte. They are fresh captures, not the October 5 input files.

## Measured performance increases versus October 5

Latency reduction = `(old - new) / old × 100`. Throughput increase =
`(old latency / new latency - 1) × 100`; these percentages are different.

| Measurement | October 5 | October 22 | Change |
|---|---:|---:|---|
| Mean image inference, 30 samples | 8.3241 s | 4.4612 s | **46.41% lower latency** |
| Serial image throughput | 0.1201 images/s | 0.2242 images/s | **86.59% higher; 1.866× speed** |
| Summed inference, 30 samples | 249.7241 s | 133.8347 s | **115.8894 s saved; 46.41% less time** |
| Mean excluding first image | 8.1050 s | 4.2243 s | **47.88% lower**; separate first-use exclusion |
| Model loading | 25.1305 s | 25.3014 s | **0.68% longer**, effectively unchanged |
| Loaded model footprint | 1.8969 GiB | 1.8969 GiB | **0% change** at reported precision |
| Peak CUDA allocation | 2.4977 GiB | 2.4977 GiB | **0% change**, identical recorded value |
| Peak inference-phase whole-system RAM | 5400 MB | 5401 MB | **0.02% higher**, effectively unchanged |
| Weighted camera delivery FPS | 30.0078 | 30.0012 | **0.022% lower**, effectively unchanged |

This is a substantial measured reduction in inference time across all five videos,
but **not a controlled speedup on identical images or an accuracy improvement**.
Both sessions sampled six frames per source video through the camera; recording
alignment, framing and captured content differ. The prompt and requested output
also changed from detailed descriptions to short labels. No same-image short32
comparison was performed in this final compact16-only run.

Yesterday’s separate three-image control measured **29.79% lower latency on
identical October 5 images**, or **9.43% lower versus the full original 30-image
mean**. Those are distinct comparisons; do not add 29.79% and 46.41%.

## Per-video results

| Video | Completed | October 5 mean | October 22 mean | Latency reduction |
|---|---:|---:|---:|---:|
| 1 | 6/6 | 9.1645 s | 5.4841 s | **40.16%** |
| 2 | 6/6 | 8.6596 s | 4.1902 s | **51.61%** |
| 3 | 6/6 | 8.3916 s | 4.1829 s | **50.15%** |
| 4 | 6/6 | 7.8535 s | 4.2475 s | **45.92%** |
| 5 | 6/6 | 7.5515 s | 4.2012 s | **44.37%** |

## Where inference time is spent

| Stage | Mean | Share of measured inference |
|---|---:|---:|
| Image encoding and image-prefix prefill | **3.9174 s** | **87.81%** |
| Prompt processing and text generation | **0.5437 s** | **12.19%** |
| Total | **4.4612 s** | 100% |

Encoding includes first-use setup: the first image took **11.3289 s**, including
**10.4990 s encoding** and **0.8300 s query**. The remaining 29 images averaged
**4.2243 s**. Overall range was 4.0908–11.3289 s. No hidden warmup query was
issued or excluded from the primary average. CUDA synchronization brackets the
stages; PIL loading and result-file writes are outside the inference interval.

Generation is now the smaller cost. Even eliminating the whole query stage
would reduce the observed total by only **12.19%**, assuming encoding unchanged;
realistic gains from further prompt shortening would be smaller. Image encoding
is the next performance target, not a newly enabled text cache. This theoretical
bound is arithmetic from this run, not a measured optimization.

## What changed, and what stayed fixed

- Same official model: `moondream/moondream-2b-2025-04-14-4bit`, revision
  `a89c59223ef8b5bb7826780728eeec172727ca84`.
- Same working dependencies: PyTorch 2.8.0, Transformers 4.57.6, TorchAO 0.11.0,
  Pillow 11.3.0. Same CUDA mapping and allocator-warmup no-op workaround.
- Same capture pipeline: Argus sensor 0, requested 1280×720/30 FPS input,
  conversion to 640×360 BGR, dropping stale appsink buffers.
- Same successful October 5 scheduling: record first, release camera, load model,
  analyze saved images serially. No concurrent queries.
- Automatic source labels/durations replaced terminal metadata entry; preview S
  still starts each camera recording, followed by manually starting playback.
- Codex closed during benchmark to preserve memory headroom. This is associated
  with successful model-only execution, not an isolated measure of its speed effect.
- Shorter prompt and **16-token cap**, temperature=0, replacing the October 5
  detailed prompt and 96-token cap. Maximum token budget is **83.33% lower**,
  but this does not mean 83.33% lower actual generation or overall latency.
- Added separate encoding/query timing, incremental JSON/sidecars, phase-specific
  tegrastats, capture FPS/gaps and explicit timeout/exit-status evidence.

Exact compact16 prompt:

> Name the clearest visible construction hazard in at most six words. If unclear, say 'none clear'. Do not infer motion or hidden hazards.

Settings: `{"max_tokens": 16, "temperature": 0}`. Only this profile ran; the
generic stored design string mentions paired queries, but the profiles/results
establish that this was **one compact16 query per image**, with no paired arms.

## Cache and current memory behavior

The model already uses text KV caches and EncodedImage prefix caches. They were
**not newly enabled**. Splitting `encode_image` from `query(encoded, ...)` exposes
encoding cost; it does not avoid the required first encoding of each new image.
This single-profile run encoded each image once and did not reuse stale encoded
frames or share image caches across changing scenes. The separate live comparison
can encode one selected image once for two serial prompts, but that is not what
produced these 30 results. No cache speedup was isolated experimentally.

The inspected crop code already selects one local tile plus a global tile for
640×360 inputs. Smaller input files therefore do not automatically reduce GPU
crop count. Any future crop/model change needs quality checks.

- CUDA-visible free memory before/after load: **5.1733 / 3.3153 GiB**.
- CUDA allocated after load: **1.9081 GiB**; after each answer: **2.2566 GiB**.
- Peak CUDA allocation: **2.4977 GiB**, including loading and inference.
- Peak inference-phase whole-system RAM: **5401/7620 MB**, **358 samples**.
- Earlier capture-only peak: **1909 MB**, **395 samples**; a different phase,
  not a model memory reduction. These unified-memory figures overlap; do not add.
- **Eight NvMap allocation-error pairs** remain in the successful terminal log.
  They did not prevent completion; the run is not warning-free.

Full analysis attempt lifetime was **181.3844 s**, including imports, loading,
queries and output overhead. Model load plus summed inference was **159.1361 s**.
Neither number is a live camera-to-answer measurement.

## Camera recordings and timing interpretation

All five recordings completed: **4500 recorded frames**, **30/30 samples**,
150 seconds requested source playback, 150.1269 s summed capture wall time.
Delivery FPS by video: 29.9839, 30.0145, 30.0122, 30.0109, 30.0120. Maximum
delivery gap was 0.1114 s in Video 1; other clip maxima were about 0.036–0.037 s.
Host timestamps do not measure sensor exposure or exact source playback position.

Because inference was resumed later, the recorded mean frame age at inference
start is **925.86 s** and mean capture-to-answer is **930.32 s** (~15.5 minutes).
These include the recording/session break and analysis restart. They are **not**
4.46-second live response measurements. At 0.224 images/s, even a stable
concurrent version would be periodic scene analysis, not 30 FPS video understanding.

## All 30 returned outputs

The table trims leading whitespace for readability; JSON preserves raw answers.
Video times are approximate positions relative to camera-recording start.

| Video | Sample | Video time | Inference | Generated output |
|---|---:|---:|---:|---|
| 1 | 1 | 5.03 s | 11.3289 s | Unstable excavator |
| 1 | 2 | 15.03 s | 4.1923 s | Digging |
| 1 | 3 | 25.03 s | 4.3655 s | Unstable excavator |
| 1 | 4 | 35.02 s | 4.3777 s | Uneven ground |
| 1 | 5 | 45.01 s | 4.3603 s | Unstable excavator |
| 1 | 6 | 55.00 s | 4.2798 s | Orange excavator |
| 2 | 1 | 2.93 s | 4.1837 s | Ladder |
| 2 | 2 | 8.76 s | 4.1881 s | Ladder |
| 2 | 3 | 14.59 s | 4.1959 s | Ladder |
| 2 | 4 | 20.42 s | 4.1862 s | Ladder |
| 2 | 5 | 26.25 s | 4.1844 s | Ladder |
| 2 | 6 | 32.12 s | 4.2026 s | Ladder |
| 3 | 1 | 1.60 s | 4.1677 s | Holes |
| 3 | 2 | 4.76 s | 4.1898 s | Unclear |
| 3 | 3 | 7.93 s | 4.1824 s | Unclear |
| 3 | 4 | 11.09 s | 4.1922 s | Concrete |
| 3 | 5 | 14.26 s | 4.2743 s | Unstable structure |
| 3 | 6 | 17.42 s | 4.0908 s | Tree |
| 4 | 1 | 2.10 s | 4.1982 s | Metal beam |
| 4 | 2 | 6.26 s | 4.2774 s | Person on beam |
| 4 | 3 | 10.43 s | 4.1822 s | Metal beams |
| 4 | 4 | 14.59 s | 4.2791 s | Person on beam |
| 4 | 5 | 18.76 s | 4.3641 s | Person on a beam |
| 4 | 6 | 22.92 s | 4.1838 s | Metal beams |
| 5 | 1 | 0.93 s | 4.1880 s | Forklift |
| 5 | 2 | 2.76 s | 4.1784 s | Unclear |
| 5 | 3 | 4.60 s | 4.2885 s | Tractor overturned |
| 5 | 4 | 6.43 s | 4.1857 s | Tires |
| 5 | 5 | 8.26 s | 4.1799 s | Tires |
| 5 | 6 | 10.09 s | 4.1865 s | Unclear |

### Output-quality tradeoffs

Four of 30 outputs (**13.33%**) say “Unclear.” All six Video 2 outputs say
“Ladder.” Other outputs include “Digging,” “Orange excavator,” “Concrete,” “Tree,”
“Metal beams,” “Forklift,” and “Tires.” These name objects or activities without
explaining a safety concern. Several outputs do name potential conditions, such
as “Unstable excavator,” “Uneven ground,” and “Tractor overturned,” but none has
been validated against images/ground truth here. Faster brief labels are not
evidence of equal hazard usefulness, improved accuracy, or reliable event detection.

Retokenized answer lengths were 1–4 tokens and no possible-cap flag was set.
These are approximate answer retokenizations, not exact generation-token counts;
the current evidence does not suggest the 16-token cap forced long truncations.
The prompt’s demand for brevity, not only the maximum cap, changed output length.

## Failed attempts retained as experiment history

The original recording run’s analysis exited **124** at 12:24 while initializing,
with zero answers. It remains intact. The later user-run analysis copied only
saved images/timestamps into a unique result folder and completed successfully.
No video was re-recorded for the completed analysis.

The earlier live attempt `results/live_compare_20261022T160402845773289` failed
with Argus InsufficientMemory/buffer-allocation errors despite Codex being absent.
That confirms concurrent camera/model resource handling remains unresolved; it
does not invalidate the completed sequential benchmark.

## Route toward NVIDIA-style live analytics

The NVIDIA video the user watched has not been identified, so its hardware,
models, output task and published FPS cannot be compared with this experiment.
A defensible reference is NVIDIA's documented DeepStream architecture: GStreamer
video processing, TensorRT inference and object tracking are combined for video
analytics. Its `nvinfer` element executes TensorRT inference. These are different
workloads from generating free-form Moondream answers. See
[NVIDIA DeepStream overview](https://github.com/NVIDIA/DeepStream) and
[NVIDIA nvinfer documentation](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvinfer.html).

Recommended next steps, **not implemented performance gains**:

1. Compact16 inference on today's saved 30 images is now complete. Run short32
   on these identical inputs in a fresh attempt for a paired comparison.
2. Measure encoder versus generation time and examine output completeness.
   This identifies whether shorter answers can materially reduce total latency.
3. Resolve concurrent camera/model allocation independently, preserving the
   working environment. Use process supervision/bounded native-operation tests
   and capture/allocator evidence; repeated restarts alone have not solved it.
4. For frequent updates, evaluate a lightweight detector/tracker for narrowly
   defined observable conditions, invoking Moondream selectively for explanation.
   That is an architectural proposal requiring separate accuracy/latency tests,
   not proof that these construction hazards are already detectable reliably.
5. If concurrent VLM operation becomes stable, retain a latest-frame-only slot,
   one serial model worker, main-thread preview, and measure actual delivery-to-
   answer delay and missed-event behavior. Paired prompts slow update frequency;
   single-prompt deployment-style tests should be measured separately.

No TensorRT/DeepStream installation or stack migration was performed. Such a
migration needs a separate compatibility plan rather than changing the working
PyTorch/CUDA installation during a timed benchmark session.



## Next steps informed by the completed run

1. Review these 30 images against their terse answers; label unsupported/missed
   hazards before trading more context for speed.
2. Run short32 on these **same saved October 22 images** in a fresh result folder
   for a controlled paired prompt comparison; no refilming needed.
3. Prioritize encoder cost and a smaller/specialized vision model for substantial
   additional gains; query generation is only 12.19% of current measured cost.
4. Investigate bounded concurrent capture/model allocation separately. A working
   camera alone plus working VLM alone is not proof of a working combined pipeline.
5. Evaluate a fast detector/tracker for narrow conditions and selective slower
   VLM explanations. Any runtime migration requires a separate compatibility
   plan; no dependencies were changed during this benchmark.

## Evidence

- [Completed analysis JSON](../../results/camera_analysis_20261022T163529601772802/analysis_fast.json)
- [Completed analysis log](../../results/camera_analysis_20261022T163529601772802/terminal.log)
- [Successful exit status](../../results/camera_analysis_20261022T163529601772802/exit_status.txt)
- [Analysis telemetry](../../results/camera_analysis_20261022T163529601772802/analysis_fast_tegrastats.txt)
- [Capture provenance and copied input paths](../../results/camera_analysis_20261022T163529601772802/capture.json)
- [Original capture manifest](../../results/camera_fast_20261022T161952378124366/capture.json)
- [October 5 results](../../results/camera_v2-2~2026-10-05/analysis.json)
- [October 21 three-image control](../../results/postrestart_control_20261021T205726708981754/analysis.json)

No model/camera execution, dependency changes, commits or pushes were performed
by the assistant while extracting these results.
