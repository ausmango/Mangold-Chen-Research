# Meeting script: model choice and live concurrency — November 5, 2026

Read the spoken sections in order, pausing for discussion. They are intended for
approximately 10–13 minutes at a conversational pace. The comparison table and
source notes are reference material; you do not need to read every cell aloud.
This is a research/design update, not a new benchmark or a claimed working fix.

## 1. Open with the priority and the established progress

The main priority now is concurrency: keeping the camera running while the AI
provides descriptions, rather than recording everything and analyzing it later.

We already have a useful baseline. Across 30 camera-captured images, mean
inference fell from 8.32 seconds to 4.46 seconds, a 46.4% reduction in latency.
However, that successful run was sequential. It also replaced detailed answers
with short labels, and the images were newly captured, so the percentage does
not isolate one optimization or demonstrate equal accuracy.

I am keeping that result as the established baseline. I have not run a new model
benchmark this session. The progress today is reviewing the concurrency failure,
comparing smaller model candidates, and defining a live-first implementation
and evaluation plan.

## 2. How similar are the proposed models to Moondream?

They are similar in purpose: they are vision-language models that take visual
information and a text question and generate language. We would still aim to ask
about workers, equipment, activities, and clearly visible concerns.

They are not smaller versions of our exact checkpoint. They have different
architectures, training, image processors, and interfaces. We would need a model
adapter rather than simply changing the name in the current `model.query` call.
We also cannot assume their strengths on construction scenes match Moondream's.

Our current model is the official 4-bit Moondream 2B-class release. It has already
completed our saved-image tasks, giving us a known reference for both runtime
and output limitations. I do not recommend deleting or replacing that baseline.

The first alternative I would investigate is the original Liquid AI
LFM2-VL-450M. Its published specification includes an 86-million-parameter vision
encoder and adjustable image-token controls, and its documented Transformers
requirement is 4.57 or newer. That makes it a candidate for our installed version
family, not a confirmed compatible deployment. The newer LFM2.5 version has a
higher software requirement, so it is not the first choice for preserving our
current setup. [Model source](https://huggingface.co/LiquidAI/LFM2-VL-450M).

The other main candidate is SmolVLM2-500M. It supports images, multiple images,
and video input. That makes it worth considering if we later want a short window
of visual context. Video support does not mean it can continuously understand
every camera frame or detect a fall reliably. [Model source](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct).

## 3. A smaller model may sacrifice capability—but size is not the whole answer

The concern about smaller models is valid. They can miss subtle details, confuse
objects, or give less complete explanations. But I would not assume that accuracy
always decreases predictably with parameter count. Training data, visual
processing, precision, and the specific task also matter.

For example, Hugging Face reports a Video-MME score of 42.2 for its 500M variant
and 33.7 for its 256M variant. That illustrates a capability tradeoff within one
family. It does not measure construction-hazard accuracy, and it does not rank
either one against our Moondream run. [Published evaluation](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct).

The choice should therefore be the smallest model that gives useful, sufficiently
supported descriptions while operating live—not simply the smallest download.
A fluent wrong answer is not progress. Equally, a detailed answer that arrives
after a changing scene has become irrelevant has limited practical value.

I would separate two milestones. First, demonstrate reliable concurrency with
responses that are clearly marked as experimental. Then establish whether the
descriptions meet our quality needs. Concurrency is the first engineering gate;
quality remains a separate gate before claiming useful hazard monitoring.

## 4. What concurrency would actually look like

The camera should capture and preview continuously. The inference worker takes
the newest available frame, processes it, and returns a description. While it
works, arriving frames replace the pending frame rather than building a backlog.
Once the worker finishes, it selects the newest frame again.

For illustration, at 30 camera frames per second, a 4.46-second processing period
spans roughly 134 camera frames. Processing every old frame in order would make
answers increasingly stale. Skipping intermediate frames is therefore a normal
part of this latest-frame design, not evidence that we are understanding the
entire video sequence.

The display should show the live image, the last completed answer, and the age
of the frame used for that answer. Otherwise a smooth preview can make a stale
description look current. The number I care about is actual camera-to-answer
delay and how often useful updates arrive, not only camera FPS.

## 5. Why the current attempt failed, and what we can improve

The saved concurrent attempt reported camera-buffer allocation errors while
Moondream was loading. Codex was closed, so we now know closing it was not the
complete solution. It helped model-only memory headroom, but did not establish
stable operation with the camera active.

The Jetson shares physical memory between the CPU, GPU, camera and other
processes. Some available total RAM does not guarantee every required allocation
will succeed. We have not isolated the exact allocation constraint, so I would
not claim that a smaller model is a guaranteed fix.

The software also needs stronger failure handling. A native camera read can
block the main thread. Setting a stop flag does not necessarily interrupt that
call, and the current worker-join loop can wait without an overall deadline.

My proposed design is a supervised capture process and one separate inference
process. The capture process owns its preview on its main thread. They exchange
only the latest frame through bounded shared memory, along with its sequence
number and timestamp. An independent supervisor monitors progress and handles
worker failures with explicit shutdown deadlines.

This provides better isolation and recovery. It does not create extra physical
RAM. It also has not yet been implemented and validated as a fix. We should
change one factor at a time instead of changing the model, camera settings and
runtime together and then guessing what helped.

## 6. Why I am shifting attention away from shorter prompts

Our measurements show that the image-and-prefix stage averages 3.92 seconds,
while the query and text-generation stage averages 0.54 seconds. About 88% of
our inference time is therefore spent preparing the visual information.

Image encoding means converting pixels into internal visual features the
language model can use. Our measured interval also includes preparing the image
prefix for the language model, so it is not a pure vision-encoder measurement.
We need finer profiling before attributing all of that time to one component.

Even eliminating the whole query stage would save only about another 12% if
encoding did not change. That is why increasingly short answers are unlikely
to produce the next major improvement. They can also remove the explanations
we actually want.

The better hypothesis is to reduce visual-processing cost and memory demand
while restoring a useful short sentence. Options to investigate include a smaller
visual model, supported visual-token controls, and later backend optimization.
Reducing resolution blindly is not enough: the current crop layout may remain
the same, and small objects can become less visible.

## 7. What the cache helps with—and what it does not

Moondream already uses KV caching during text generation. It also allows an
encoded image to be reused for several questions about that same image.
We did not newly enable that mechanism.

Same-image reuse is useful for a prompt comparison: encode once, then ask two
questions serially. But it makes us spend longer on that observation before
selecting another frame. For the first live demonstration, I recommend one
prompt per selected image rather than two comparison prompts.

A cached image must not stand in for a changed scene. Reusing it indiscriminately
could make the system answer quickly about something that is no longer visible.
We should improve freshness through current-frame selection, not hide the cost
by repeatedly describing an old frame.

## 8. What I would implement and test next

The first trial should be deliberately simple: one camera, one model worker,
one latest-frame slot, and one descriptive prompt. Leave full video recording
off initially so we can isolate live inference. That is a test simplification,
not proof that recording caused the earlier loading failure.

To separate causes, first validate the supervised capture path with a deliberately
delayed dummy worker. Then keep that camera path fixed and try one real model at
a time. If the same allocation failure persists, preserve the failed attempt;
do not interpret process separation as proof that memory pressure is fixed.

We should ask for one useful sentence, such as the visible worker or equipment
and any clearly supported concern. A common 32-token output cap gives more room
than the short-label benchmark. Exact prompts, model revisions and preprocessing
must be recorded; the same token cap does not make different tokenizers identical.

The first milestone is a supervised two-minute live session that produces at
least five descriptions while camera sequences keep advancing. That is a proposed
acceptance criterion, not a result or proof of real-time safety performance.

Then measure capture FPS, source-frame age, image-processing time, complete-answer
delay, memory peaks and update frequency. Report typical and slower-case latency,
including the first-use result. Separately review the images for supported,
unsupported and missed details. We should also test whether a failed worker can
be reported and stopped without freezing the interface.

Only after that works should we add full recording and, if needed, a small
multi-frame window. Multiple frames may help with changing activities, but they
add processing and memory cost. More temporal context is not free.

## 9. How this relates to NVIDIA-style live demonstrations

NVIDIA's Live VLM WebUI is a useful architectural reference because it separates
camera interaction from the model backend and exposes a frame-processing
interval. A live interface does not tell us that a model examines every frame,
or that its hardware and camera path match ours. We have not identified the
exact video, so I would not quote its speed as our expected performance.
[Source](https://github.com/NVIDIA-AI-IOT/live-vlm-webui).

A later alternative is a fast detector and tracker for narrowly defined visible
conditions, with a VLM providing selected explanations. That would require its
own validation; detecting a person or excavator is not automatically detecting
a dangerous interaction.

## 10. Close with the decision and the limits

My recommendation is to keep Moondream as our documented baseline, prototype
supervised latest-frame concurrency, and first evaluate LFM2-VL-450M for live
single-frame descriptions. SmolVLM2-500M is the next candidate if temporal input
becomes a priority. I would not jump straight to the 256M model just because it
is smaller.

The research progress is a clearer diagnosis, sourced model options, and a
live-first validation plan. The latest measured speedup is still the previous
46.4% latency reduction. We are not claiming that a new model is faster or more
accurate on our device yet.

The decision I would like guidance on is: what level of description quality
must we preserve while making concurrency work, and is live periodic description
the next milestone before we attempt temporal hazard recognition?

## Reference table: current model versus candidates

All prospective models are untested on this CSI setup. Parameter class is not a
checkpoint-size, RAM, speed or accuracy measurement. Larger text context also
does not imply lower visual latency.

| Model | Similarity to current task | Important difference | Local evidence / role |
|---|---|---|---|
| [Official Moondream 2B-class 4-bit](https://huggingface.co/moondream/moondream-2b-2025-04-14-4bit) | Current image caption/query model | Quantized release with model-specific `.query` and encoded-image API | 30/30 saved images; 4.4612 s mean; 1.8969 GiB footprint; 5401 MB system RAM peak. Concurrent attempt failed. |
| [LiquidAI/LFM2-VL-450M](https://huggingface.co/LiquidAI/LFM2-VL-450M) | Image-and-text descriptions/questions | Different backbone/processor; 86M vision encoder, visual-token controls; LFM license | Card supports Transformers 4.57+; first lower-disruption candidate, runtime unverified. |
| [SmolVLM2-500M-Video-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct) | Image questions plus multi-image/video input | Idefics3-based; separate processing/API; Apache-2.0 | Video-context candidate; verify dependencies and current attention backend. |
| [SmolVLM2-256M-Video-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-256M-Video-Instruct) | Same broad task family as 500M sibling | Lower-capacity alternative; published video scores lower within this family | Optional capacity control, not the initial default. |

Source-specific notes: the original Liquid model card states it is not intended
for safety-critical decisions. None of these general-purpose model cards is
validation of construction-hazard accuracy. The newer
[LFM2.5-VL-450M](https://huggingface.co/LiquidAI/LFM2.5-VL-450M) card requires
Transformers 5.1+, so it is deferred rather than upgrading our working stack.

## Evidence to open if asked

- [Completed October 22 metrics](../results/camera_analysis_20261022T163529601772802/analysis_fast.json)
- [Completed October 5 metrics](../results/camera_v2-2~2026-10-05/analysis.json)
- [Concurrent attempt: no completed images](../results/live_compare_20261022T160402845773289/video_01/analysis.json)
- [Concurrent Argus error log](../results/live_compare_20261022T160402845773289/video_1_terminal.log)
- [Full November 5 research review](research-log/2026-11-05-live-inference-research.md)

All proposed improvements are hypotheses/design choices. No new benchmarks,
model installs, dependency changes, camera runs, commits or pushes occurred while
preparing this script.
