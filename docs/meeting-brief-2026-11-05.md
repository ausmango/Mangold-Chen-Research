# November 5 meeting brief — live inference research

For the expanded read-aloud discussion (~10–13 minutes) and direct model
comparison, use [the live-concurrency meeting script](meeting-script-2026-11-05-live-concurrency.md).
This page remains the shorter version.


## Say this first

“Last time we reduced mean image inference from 8.32 to 4.46 seconds across
30 samples: 46.4% lower latency. That was recording followed by analysis, and
shorter hazard labels replaced fuller descriptions. I have not run a new
benchmark this week. I used this session to identify the next bottlenecks and
research a live-first design and smaller models.”

## Three findings worth discussing

1. **Stop optimizing only answer length.** Our measured image/prefix stage is
   3.92 s, about 88% of total; query/generation is 0.54 s. Removing all query time
   could only save another ~12% with encoding unchanged. “Image encoding” means
   turning pixels into visual features and preparing that information for the
   language model; our timer includes both visual processing and image-prefix
   prefill, so we need finer profiling before selecting an optimization.
2. **Concurrency has a reliability problem as well as a memory problem.** Argus
   buffer allocation failed with Codex closed. A blocked native camera call can
   also freeze the current main thread, and its stop event cannot interrupt it.
   Proposed fix: independently supervised capture/UI and one inference process,
   connected by a replaceable latest-frame mailbox. This improves isolation and
   recovery; it does not create more shared RAM or prove the allocation fix.
3. **Smaller visual workloads are a stronger next hypothesis.** The original
   `LiquidAI/LFM2-VL-450M` has an 86M vision encoder and adjustable visual-token
   controls; its card supports Transformers 4.57+. SmolVLM2-500M is a small
   video-capable alternative. These have not been tested on our camera. The newer
   LFM2.5 variant needs Transformers 5.1+, so it should not replace our environment.

Sources: [LFM2-VL-450M card](https://huggingface.co/LiquidAI/LFM2-VL-450M),
[SmolVLM2-500M card](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct),
[LFM2.5-VL-450M card](https://huggingface.co/LiquidAI/LFM2.5-VL-450M).

## Explain the NVIDIA comparison

“Live video on screen and live AI responses are different rates. NVIDIA's
[Live VLM WebUI](https://github.com/NVIDIA-AI-IOT/live-vlm-webui) exposes a VLM
backend and a frame-processing interval. We should copy that separation and
measure response freshness, rather than assume the model understands every
frame because preview runs at 30 FPS. We haven't identified the exact video,
so its hardware, model, and pipeline aren't a controlled comparison.”

## Explain the cache

“Moondream already uses KV caching. It also supports asking several questions
about one encoded image without encoding it again. Neither lets us treat an old
image as evidence for a new scene. This week’s objective is stable concurrent
capture and cheaper visual processing, not claiming newly enabled caching.”

## Propose one concrete next milestone

“Get a supervised two-minute camera session with a single small VLM, current-frame
selection, and at least five useful descriptions while capture keeps advancing.
Log actual camera-to-answer delay, frame age, P50/P95, capture FPS and RAM. Start
with one stream, one descriptive prompt, no full recording, and no two-prompt
comparison. Add recording and a little temporal context only after it is stable.”

Those are proposed acceptance criteria, not achieved performance. We need to
review supported/missed/unsupported details in the images before making accuracy
or safety claims. Live periodic description is not verified temporal hazard
recognition. No new model is guaranteed to fit or meet a two-second goal.

## End with the decision you need

“Do you prefer that I first preserve the existing stack and try LFM2-VL-450M for
live frame descriptions, or prioritize SmolVLM2-500M's multi-frame/video capability?
I recommend the first live-image reliability trial, then temporal context.”

Full evidence, source-level findings and staged plan:
[research note](research-log/2026-11-05-live-inference-research.md).
