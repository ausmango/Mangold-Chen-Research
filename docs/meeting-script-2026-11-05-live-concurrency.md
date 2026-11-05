# Research discussion: achieving concurrent live descriptions

**Priority:** keep CSI capture and preview running while one AI worker describes
the newest available image. Model replacement is one possible way to meet that
goal; it is not the entire concurrency solution.

## 1. Established results and the unresolved gap

| Evidence | Result | Interpretation |
|---|---:|---|
| October 5 inference | 30/30 images; mean **8.3241 s** | Completed recording-then-analysis baseline |
| October 22 inference | 30/30 images; mean **4.4612 s** | **46.41% lower latency**, **1.866× throughput** |
| Image encoding + prefix preparation | **3.9174 s; 87.81%** of inference | Main measured computational cost; includes visual processing and language-model image prefill |
| Query + generation | **0.5437 s; 12.19%** | Further shortening has limited remaining runtime benefit |
| Peak whole-system RAM | **5401 MB** vs 5400 MB previously | Essentially unchanged, not a RAM-efficiency improvement |
| Model footprint / peak CUDA allocation | **1.8969 / 2.4977 GiB** | Separate measurements, not additive to system RAM |
| Camera alone | **29.76 FPS** in a completed recording check | Capture throughput works independently |
| Concurrent camera + Moondream | Argus buffer allocation failed; **zero live answers** in latest attempt | Still an unresolved hardware/software integration problem |

The 46.41% latency change is descriptive: fresh camera images and a much shorter
prompt/output task changed between sessions. It does not establish equal answer
quality or isolate one optimization. **No new benchmark was run on November 5.**

**Discussion point:** the next milestone is a working live pipeline, rather than
another reduction in offline answer length.

## 2. How large is the step from 2 billion to 450 million parameters?

A parameter is a learned numerical weight. It is a measure of model scale, not
a direct measure of accuracy, memory use or response time.

| Model | Nominal parameter class | Fraction of nominal 2B baseline | Reduction in parameter count | Role |
|---|---:|---:|---:|---|
| Current Moondream | **2,000M** | 100% | Reference | Preserve as measured baseline |
| LFM2-VL-450M | **450M** | **22.5%** | **77.5% fewer** | First smaller live-image candidate |
| SmolVLM2-500M | **500M** | **25.0%** | **75.0% fewer** | Candidate for sampled video/multi-frame context |
| SmolVLM2-256M | **256M** | **12.8%** | **87.2% fewer** | Lower-capacity control, not initial preference |

**2B → 450M is a substantial capacity change:** the nominal reference has about
**4.44 times as many parameters**. These calculations use advertised size classes,
not exact tensor counts, and are **not predicted speed or accuracy percentages**.

A crucial distinction: our current model is already quantized to a 4-bit release,
whereas candidate examples use BF16. BF16 stores 16 bits per value. Fewer weights
can therefore use more bits each. Working memory also includes visual features,
caches, temporary tensors and camera buffers. **77.5% fewer parameters does not
mean 77.5% less RAM or 4.44× faster inference.** The current vision path is not
uniformly INT4 simply because the release name says “4-bit.”

## 3. Similar purpose, different architecture and deployment

| Aspect | Current Moondream | LFM2-VL-450M | SmolVLM2-500M |
|---|---|---|---|
| Exact release | `moondream/moondream-2b-2025-04-14-4bit` | `LiquidAI/LFM2-VL-450M` | `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` |
| Similarity | Image descriptions and questions | Same broad image-and-text purpose | Same purpose plus multi-image/video inputs |
| Distinguishing feature | Existing custom `.query()` and EncodedImage interface | Published **86M vision encoder**; adjustable visual-token budget | Idefics3-based video-capable family |
| What changes in code | Existing adapter | Different processor/chat template and generation adapter | Different processor/chat template and generation adapter |
| Software gate | Working Transformers 4.57.6 stack | Card supports **4.57+**; actual checkpoint execution untested here | Card example uses FlashAttention 2; verify a compatible installed-backend path |
| License | Apache-2.0 | LFM Open License | Apache-2.0 |
| Measured on this Jetson | Saved-image inference succeeds; concurrent attempt fails | **Not tested** | **Not tested** |

Sources: [Moondream release](https://huggingface.co/moondream/moondream-2b-2025-04-14-4bit),
[Liquid AI specification](https://huggingface.co/LiquidAI/LFM2-VL-450M),
[SmolVLM2 specification](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct).

These candidates are not smaller copies of Moondream. Training and architecture
differ. Their labels/descriptions could be similar in form but different in
completeness or correctness. No local checkpoint size, footprint, latency or
hazard-quality measurement exists for either candidate.

The newer [LFM2.5-VL-450M](https://huggingface.co/LiquidAI/LFM2.5-VL-450M)
requires Transformers 5.1+ per its card. It is deferred rather than upgrading our
working installation. A supported version family is not proof of runtime success.

## 4. The smaller-model caveat: what quality could we lose?

| Risk | Why it matters here | How to evaluate |
|---|---|---|
| Missing small/partly hidden details | Cables, equipment edges, workers in clutter | Review the actual input image against each claim |
| Naming an object instead of a concern | “Ladder” does not explain unsafe placement | Score whether the sentence identifies a supported condition/location |
| Unsupported inference | Model may invent movement or missing protection | Separate supported, unsupported and missed details |
| Insufficient temporal evidence | One image cannot establish how a person moved | Separate current-image descriptions from event-detection claims |
| Overconfident answers | A fluent response may be wrong | Require uncertainty where evidence is unclear |

Parameter count alone does not determine task accuracy. As a concrete within-
family example, published Video-MME scores are **42.2 for SmolVLM2-500M** versus
**33.7 for 256M**, an **8.5-point difference**. These are external video-benchmark
scores—not construction-hazard accuracy, and not a comparison with our Moondream
results. [Published evaluation](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct).

**Decision rule:** concurrency is the first engineering gate. A working live
prototype remains experimental until description quality is checked. Choose the
smallest adequate model, not simply the smallest available. Liquid AI explicitly
states its original model is not intended for safety-critical decisions.

## 5. Concurrency needs changes beyond the model

**Target flow:**

```text
CSI capture + preview ──► replaceable latest-frame slot ──► one inference worker
          ▲                                                    │
          └──── latest completed answer + source-frame age ◄────┘
                 independent supervisor watches both
```

| Current issue | Proposed improvement | What it does NOT establish |
|---|---|---|
| Argus camera buffers fail while model is resident/loading | Measure camera-open, model-load, prefix initialization and first-query memory separately; compare smaller working sets | Total available RAM alone does not diagnose the exact failed allocation |
| Native `camera.read()` or release can block | Capture/UI process separated from inference; independent heartbeat/watchdog | Process separation does not create physical RAM or guarantee a driver fix |
| Stop flag may not be checked; join can wait indefinitely | Bounded shutdown with explicit termination escalation | A kernel-level uninterruptible call may still delay exit |
| Processing old frames causes stale answers | One replaceable latest-frame slot with sequence/time; no inference backlog | Skipped frames may contain events we miss |
| Two prompts delay the next observation | One descriptive prompt per frame for live acceptance | Paired prompt comparison belongs in a separate study |

The latest failed live run had Codex absent. Closing Codex helped earlier model-
only headroom but was not a complete concurrency solution. The exact root cause
is still unresolved. Review evidence:
[partial metrics](../results/live_compare_20261022T160402845773289/video_01/analysis.json),
[Argus errors](../results/live_compare_20261022T160402845773289/video_1_terminal.log).

The supervisor design is proposed, not an implemented or demonstrated fix.
OpenCV windows remain on the main thread of the process that owns capture.

## 6. Additional improvement topics, ranked by relevance

| Priority | Topic | Why discuss it | Evidence or uncertainty |
|---|---|---|---|
| **1** | Reliable concurrent capture and recovery | The application must keep observing while AI runs and report failures clearly | Existing native-call/stop issues identified by source review |
| **2** | Smaller visual workload | Our measured image/prefix stage dominates total time | Smaller encoder/token budgets are hypotheses to test, not guaranteed gains |
| **3** | Recording overhead | Full AVI recording competes for per-frame CPU time | Completed camera-only test: mean writer call **17.00 ms**, about **51.0%** of a 33.33 ms/30-FPS frame budget |
| **4** | Finer stage profiling | Separate preprocessing, visual encoder, image-prefix prefill and decoding | Current 3.92 s encoding timer combines several components |
| **5** | Image-token / region budgets | Process fewer visual pieces or a relevant region | Can hide small hazards; cropping may remove context |
| **6** | Scene-change or event-triggered queries | Reduce repeated descriptions of nearly unchanged scenes | Triggers can miss subtle changes; use a bounded refresh interval and label stale results |
| **7** | First partial text / streaming | Can show an answer beginning earlier | Does not eliminate encoding or guarantee quicker completed answers |
| **8** | Encoder compilation/backend optimization | Potentially targets the costly stage without changing the task | Existing model has a compile path; startup cost, memory peaks and compatibility are untested |
| **9** | Detector/tracker + selective VLM explanation | Frequent narrow observations, occasional language explanations | Requires defined conditions, new validation and a separate integration plan |

Recording measurements come from a **different camera-only run**, not concurrent
inference. The 17 ms writer call is not 51% of model inference time. Disabling
recording is useful for workload isolation, but the previous allocation failure
occurred before the writer opened; it is not a proven fix. [Recording metrics](../results/camera_setup_20261021T202315853338873_20s/analysis.json).

An illustrative cost calculation: if the measured image/prefix stage became
2× faster while query time stayed fixed, total time would be about
`3.9174/2 + 0.5437 = 2.5024 s` (**43.91% lower** than 4.4612 s).
This is a hypothetical scenario, not a new measurement or model prediction.

## 7. Cache: retain the useful distinction

| Mechanism | Useful for | Limitation |
|---|---|---|
| Existing text KV cache | Avoid repeated attention computation for prior tokens during generation | Already implemented; not a new contribution |
| Encoded-image prefix reuse | Several questions about one fixed image without repeating its encoding | Does not represent a changed camera scene |
| Latest-frame mailbox | Prevent old frames accumulating while inference is busy | Improves freshness, not the cost of an individual query |
| Conditional refresh of an old answer | Potential UI/workload optimization | Must show answer age and refresh periodically; not new visual evidence |

For a first concurrent demonstration, one query per latest frame is preferable
to multiple questions about one image. Caching an old scene must not be reported
as faster analysis of the current scene.

## 8. What “live” means and how we should measure it

At 30 FPS, a 4.46-second processing interval spans roughly **134 camera frames**.
That is an illustration based on our offline average, not measured live timing.
A live design can skip old images while keeping preview responsive. It does not
therefore understand every frame or establish temporal hazard recognition.

| Metric | Question it answers |
|---|---|
| Capture FPS / maximum gap | Is the camera still observing smoothly? |
| Source-frame age when inference starts | Did the worker select current evidence? |
| Camera delivery → completed answer | How long until the observation becomes available? |
| First partial text delay | How soon does visible output begin? |
| Answer age displayed on screen | Could the current caption describe an outdated situation? |
| Useful answers per minute | How frequently does the system provide relevant updates? |
| P50/P95 latency + first-use result | What is typical delay, and how bad are slower responses? |
| Peak system RAM / current and peak CUDA | Is the combined workload sustainable? |
| Supported / unsupported / missed claims | Is the language useful and justified by the image? |

NVIDIA's [Live VLM WebUI](https://github.com/NVIDIA-AI-IOT/live-vlm-webui)
is a reference for separating camera interaction from a backend and controlling
frame interval. It is not evidence that the exact model/camera/hardware in a demo
matches our CSI configuration. NVIDIA's
[DeepStream inference plugin](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvinfer.html)
supports TensorRT detection/classification/segmentation workflows; this is not an
automatic conversion of Moondream or a prevalidated hazard detector.

## 9. Proposed next experiment: live first, quality second, richer context later

| Stage | Proposed work | Pass condition / evidence |
|---|---|---|
| A. Reliability | Capture/preview with a delayed dummy worker and induced failure | No growing frame queue; supervisor reports and bounds worker failure |
| B. Concurrency | One real VLM, same camera path, one descriptive prompt, no AVI initially | Proposed two-minute session with at least five completed answers while capture sequences advance |
| C. Quality | Human review of analyzed images and one-sentence answers | Supported detail and location; uncertainty when unclear; explicit misses/unsupported claims |
| D. Performance | Same camera/image budget and common 32-token cap across candidates | Actual delay, quality, output length and RAM compared; no percentage forecast |
| E. Added functionality | Reintroduce full recording; later bounded multi-frame context | Recheck memory, response age and stability after each change |

These are proposed gates, not completed tests. Start with a fresh spawned
inference worker rather than forking after CUDA initialization. Preserve the
working dependencies and pinned Moondream baseline; change one factor at a time.
If current packages cannot support a candidate, defer it or plan a separate
approved environment instead of upgrading the working stack.

Suggested one-sentence prompt for restoring detail:

> Describe the visible workers or equipment and any clearly visible safety concern in one short sentence. If unclear, say so. Do not infer hidden hazards or movement between frames.

## 10. Decisions to discuss with the professor

- **Primary milestone:** reliable live periodic descriptions before temporal event detection?
- **Required content:** equipment/activity only, or concern + location + visible evidence?
- **Acceptable quality loss:** what omissions would make a smaller model unsuitable?
- **Latency objective:** what answer delay and worst-case freshness are useful for this application?
- **Model order:** LFM2-VL-450M first for the current version family; SmolVLM2-500M next if temporal input is needed?
- **Architecture scope:** direct VLM descriptions first, or a narrow detector/tracker with selective explanations?

**Current recommendation:** preserve Moondream as reference, validate a supervised
latest-frame pipeline, then evaluate one smaller candidate with useful sentences.
Concurrency is the first engineering gate; a successful live demo alone does not
establish safe or accurate hazard detection.

## Evidence and status

- [October 22 completed metrics](../results/camera_analysis_20261022T163529601772802/analysis_fast.json)
- [October 5 completed metrics](../results/camera_v2-2~2026-10-05/analysis.json)
- [Detailed historical comparison](research-log/2026-10-22-camera-benchmark-comparison.md)
- [November 5 research review and source-level findings](research-log/2026-11-05-live-inference-research.md)

**Completed today:** evidence review, parameter calculations, primary-source
comparison and concurrency design review. **Not completed:** new-model inference,
a working concurrency fix, new latency gains, or accuracy validation. No camera
runs, model downloads, dependency changes, commits or pushes.
