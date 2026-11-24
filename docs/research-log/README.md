# Research log

The project tested local construction-scene analysis on an 8 GB Jetson Orin Nano. Read the stages in order to follow the decisions, or start with the [latest recorded results](2026-11-24-five-video-results.md).

## September · Model feasibility and camera capture

- [Sep 21 · Moondream precision benchmarks](2026-09-21-moondream-benchmarks.md) compares FP16, generic INT8, and official 4-bit loading and image tasks.
- [Sep 28 · First CSI camera trial](2026-09-28-moondream-live-camera.md) documents a partial two-frame trial and repetitive generation.

## October · Repeatable measurements and concurrency attempts

- [Oct 5 · Five-video camera baseline](2026-10-05-moondream-camera-benchmark.md) establishes the 30-image sequential benchmark.
- [Oct 21 · Concurrent Moondream attempt](2026-10-21-moondream-concurrent-camera.md) records the latest-frame design, cache inspection, and hardware failures.
- [Oct 22 · Meeting evidence](2026-10-22-meeting-summary.md) records the interim controls and their limits.
- [Oct 22 · Completed sequential comparison](2026-10-22-camera-benchmark-comparison.md) reports the later 30-image run and explains the changed test conditions.

## November · New architecture and recorded demonstration

- [Nov 5 · Live inference design review](2026-11-05-live-inference-research.md) explains the smaller model shortlist and the case for independent capture and inference.
- [Nov 12 · Scout and Inspector implementation](2026-11-12-scout-inspector-refactor.md) records the split into detection and description workers and the CPU checks.
- [Nov 12 · Provisioning and TensorRT repair](2026-11-12-scout-inspector-provisioning.md) traces model setup, engine building, and the successful standalone build.
- [Nov 12 · First successful concurrent run](2026-11-12-first-concurrent-camera-benchmark.md) shows camera delivery during nine real VLM requests.
- [Nov 12 · Initial live analysis summary](2026-11-12-live-analysis-summary.md) compares the first concurrent trials and illustrates caption errors.
- [Nov 24 · Five-video results](2026-11-24-five-video-results.md) links all five MP4s, saved summaries, example frames, and run settings.
- [Nov 24 · Research meeting log](2026-11-24-meeting-log.md) gives a short viewing order, explains the role of YOLO, and proposes the next evaluation.

Read the dated notes before comparing numbers. The sequential and concurrent trials differ in inputs, models, sampling policies, and timing scopes. The latest evidence establishes camera and inference overlap. Hazard and temporal event accuracy still need labelled evaluation.
