# Moondream CSI Camera Benchmark — October 5, 2026

## Purpose

Evaluate local construction-scene descriptions using an official 4-bit
Moondream model on an NVIDIA Jetson Orin Nano.

The target was five construction videos with six camera samples per video,
for 30 independently analyzed frames. Measurements included model footprint,
loading time, per-frame inference latency, total inference time, CUDA
allocation, and whole-system RAM usage.

## Outcome

- Five camera recordings completed.
- Six samples were saved from each recording.
- All 30 samples were analyzed successfully.
- Mean inference latency across 30 frames: 8.3241 seconds.
- Total measured inference time: 249.7241 seconds, or approximately 4.16 minutes.
- Model load time: 25.1305 seconds.
- Reported model footprint: 1.8969 GiB.
- Peak PyTorch CUDA allocation: 2.4977 GiB.
- Peak whole-system RAM: 5,400 MB of 7,620 MB, approximately 70.9%.

These findings demonstrate successful camera acquisition and subsequent
local frame analysis. Accuracy has not been formally evaluated.

## Hardware and Software

- NVIDIA Jetson Orin Nano with approximately 8 GB shared system memory.
- NVIDIA CSI camera using Argus and GStreamer.
- Python with OpenCV, Pillow, PyTorch, Hugging Face Transformers, and TorchAO.
- PyTorch: 2.8.0.
- Transformers: 4.57.6.
- TorchAO: 0.11.0.
- Model: `moondream/moondream-2b-2025-04-14-4bit`.
- Pinned revision: `a89c59223ef8b5bb7826780728eeec172727ca84`.
- Approximate checkpoint size from earlier measurement: 1.89 GiB.
- System memory measured using NVIDIA tegrastats at 500 ms intervals.

## Changes from the September 28 Experiment

The earlier script captured one frame, generated a caption and a construction
answer, and then captured the next frame. Inference delayed subsequent
sampling, making brief events easy to miss. The previous experiment completed
two of six planned frames, with one construction response taking 78.3490 seconds.

The updated script separates acquisition from analysis:

1. Preview and align the CSI camera.
2. Record each entire playback.
3. Save six frames at scheduled times during recording.
4. Release the camera after all five captures.
5. Load Moondream once and analyze the saved frames.
6. Save structured results after every completed inference.

Additional changes:

- One construction query per frame instead of captioning plus querying.
- Explicit generation-token limit.
- Deterministic decoding with temperature zero.
- Camera output reduced from 1280 × 720 to 640 × 360.
- Model revision pinned for reproducibility.
- Separate folders and metadata for each source video.
- Full camera recordings and recording-frame timestamps retained locally.
- Per-video average and total inference times calculated automatically.
- Partial inference results preserved during interruption or failure.

The new design avoids simultaneous model inference and camera acquisition.
It therefore evaluates sequential capture and analysis rather than concurrent
live inference.

## Camera Recording and Sampling

The camera recorded videos displayed on another screen. The source files
were not passed directly to Moondream.

The acquisition pipeline requested a 1280 × 720 sensor stream and converted
it to 640 × 360 camera output. Recordings used MJPEG AVI with a nominal
30 FPS playback rate. CSV timestamps preserve actual acquisition timing.

Six sampling targets were evenly spaced at the centers of six equal
intervals within each entered duration:

`target_seconds = (sample_index + 0.5) × duration / 6`

Recorded sample times are relative to starting camera capture. They are
approximate source-video positions because playback was started manually
after capture began.

| Video | Entered label | Duration | Recorded frames | Saved samples |
|---|---|---:|---:|---:|
| 1 | Red Machine | 60 s | 1,801 | 6 |
| 2 | Video 2 | 35 s | 1,050 | 6 |
| 3 | sVideo 3 | 19 s | 570 | 6 |
| 4 | Video 4 | 25 s | 750 | 6 |
| 5 | Video 5 | 11 s | 330 | 6 |

Total requested playback duration: 150 seconds.
Total recorded camera frames: 4,501.
Total samples analyzed: 30.

### Source Videos and Saved Samples

| Video | Source | Actual sample times in seconds |
|---|---|---|
| 1 | [Source video 1](https://drive.google.com/file/d/1Kh8cFl-OckC_hMw1OA0Ld7OBrvnlOBMn/view) | 5.027, 15.024, 25.022, 35.021, 45.018, 55.017 |
| 2 | [Source video 2](https://drive.google.com/file/d/1V_I9_77qyGNHI0bDMz9p4ZmCIl6WVMUd/view) | 2.931, 8.763, 14.596, 20.427, 26.259, 32.091 |
| 3 | [Source video 3](https://drive.google.com/file/d/1eAHZRBoRS9tW2u75KlQIVbTEe4AUtwJH/view) | 1.599, 4.765, 7.931, 11.097, 14.263, 17.429 |
| 4 | [Source video 4](https://drive.google.com/file/d/1Z4rXvQwLszuOFTq0TfLRgwRaF1T0j-d3/view) | 2.098, 6.264, 10.429, 14.595, 18.761, 22.928 |
| 5 | [Source video 5](https://drive.google.com/file/d/1ok9RtYCoPQZHGZcAsuAdVDmx9ElNl464/view) | 0.933, 2.765, 4.598, 6.431, 8.264, 10.098 |

Source links may require access permission.

## Prompts and Generation Settings

### Initially Proposed V2 Prompt

> In at most two short sentences, describe visible workers, equipment, and
> obstructions. Mention a potential hazard only if clearly visible. Do not
> guess or repeat.

Settings: `max_tokens=48`, `temperature=0`.

The evidence inspected for this research note does not establish a completed
run with this prompt. It should not be treated as a measured comparison arm.

### Prompt Used in the Successful Run

> Inspect this construction image for visible safety concerns. Describe the
> workers, equipment, and activity. Check for objects or cables obstructing
> walking paths, uneven ground, workers near machinery, and people visibly
> stumbling or falling. For each clearly visible concern, state what it is
> and where it appears. If no concern is clearly visible, say so. Do not assume
> movement, missing protective equipment, or hidden hazards. Answer in at most
> three concise sentences without repetition.

Settings: `max_tokens=96`, `temperature=0`.

The cached model code showed a default generation limit of 768 tokens.
The successful run explicitly reduced this to 96 tokens. This limits output
length but does not guarantee accurate descriptions or eliminate repetition.

## Successful Inference Results

| Video | Completed samples | Mean inference latency | Total inference time |
|---|---:|---:|---:|
| 1 | 6/6 | 9.1645 s | 54.9869 s |
| 2 | 6/6 | 8.6596 s | 51.9576 s |
| 3 | 6/6 | 8.3916 s | 50.3495 s |
| 4 | 6/6 | 7.8535 s | 47.1209 s |
| 5 | 6/6 | 7.5515 s | 45.3091 s |
| All videos | 30/30 | 8.3241 s | 249.7241 s |

Inference latency includes the model query on each image, including image
encoding and text generation. It excludes camera acquisition and model loading.

Model loading plus summed inference time was approximately 274.85 seconds.
This is not the complete experiment wall time: it excludes acquisition,
manual setup, and other processing overhead.

### All Returned Descriptions

The complete timestamped descriptions are preserved in the
[generated report](../../results/camera_v2-2~2026-10-05/report.md).

Exact per-frame results, generation settings, and memory measurements are
available in the [analysis JSON](../../results/camera_v2-2~2026-10-05/analysis.json).

The [terminal log](../../results/camera_v2-2~2026-10-05/analysis_log.txt)
preserves execution output and warnings.

## Qualitative Findings

The successful report includes descriptions of uneven ground, debris,
cables, workers near equipment, tilted machinery, and possible stumbling
or falling. These are model-generated claims and require comparison with
the saved images.

Examples:

- Video 1 at approximately 5.03 seconds described tilted equipment and a
  person who appeared to be falling.
- Video 2 at approximately 20.43 seconds mentioned cables and debris
  on the floor.
- Video 3 at approximately 14.26 seconds described a worker as visibly
  stumbling.
- Video 5 at approximately 6.43 seconds described overturned equipment
  and a person who appeared to be falling or stumbling.

The report also exposes limitations:

- Video 2's first description repeated the ladder statement.
- Video 3 at approximately 7.93 seconds repeated the same kneeling-worker
  statement three times.
- Video 3 produced inconsistent activity descriptions across frames,
  including concrete pumping, pressure washing, and pumping water into a pool.
- Some responses exceeded the requested three-sentence format.
- Some frames were described as having no visible workers or machinery.
  The images must be inspected to determine whether this reflects the
  scene, camera framing, or an incorrect response.

The presence of a hazard-related phrase is not proof of correct hazard
detection. Ground-truth labels and manual review remain necessary.

## Memory Measurements and Failed Attempt

| Measurement | Failed attempt | Successful attempt |
|---|---:|---:|
| Initial CUDA-visible free memory | 4.3890 GiB | 5.1728 GiB |
| Model loading time | 21.5109 s | 25.1305 s |
| Model footprint | 1.8969 GiB | 1.8969 GiB |
| CUDA-visible free memory after load | 3.0882 GiB | 3.3684 GiB |
| Difference in free memory after load | 1.3008 GiB | 1.8044 GiB |
| Peak PyTorch allocation | 2.2128 GiB | 2.4977 GiB |
| Peak whole-system RAM | 4,692 MB | 5,400 MB |
| Completed descriptions | 0 | 30 |
| Status | Failed | Finished |

The failed attempt ended with a CUDA allocator internal assertion mentioning
`NVML_SUCCESS`. It failed before completing the first description.

Although named `analysis_prompt1.json`, that file records the same safety
prompt and 96-token setting as the successful run. Its filename does not
establish that it used the earlier 48-token prompt.

Jetson memory is shared between CPU, GPU, and other system components.
Whole-system RAM includes other processes. CUDA free-memory differences
are not isolated measurements of model storage, and the memory figures
must not be added together.

## Comparison with September 28

The earlier partial camera experiment recorded:

- Two completed frames out of six planned.
- Peak whole-system RAM of 6,004 MB.
- Construction-query latencies of 14.8674 and 78.3490 seconds.
- Severe repetitive generation in the second query.

The October 5 run completed 30 frames with an average inference latency
of 8.3241 seconds and peak system RAM of 5,400 MB.

Peak whole-system RAM was 604 MB lower, approximately 10.1% below the
earlier observed peak. This is a descriptive comparison, not an isolated
effect of one optimization.

Several conditions changed simultaneously: capture/inference scheduling,
output resolution, prompt, token limit, decoding settings, and background
memory availability. The recordings and sampled images also differed.
A controlled comparison is needed to attribute improvements.

## Limitations

- Five videos and six samples per video form a small feasibility dataset.
- Six evenly spaced samples may miss brief events between samples.
- Full recordings preserve those events but are not analyzed by this script.
- Each image is analyzed independently; no temporal reasoning is performed.
- The generated report is a timeline of descriptions, not an additional
  AI-generated whole-video summary.
- Camera capture and inference are sequential rather than concurrent.
- The current inference rate is approximately one frame every 8.32 seconds.
  Real-time hazard detection has not been demonstrated.
- The camera films a display, introducing framing, lighting, and screen artifacts.
- Reducing output resolution may obscure small hazards or protective equipment.
- Accuracy, false positives, false negatives, and event recall are unmeasured.
- The generation cap limits response length, but repetition still occurs.

## Next Steps

1. Review all 30 images against their descriptions and label visible hazards.
2. Record which claims are supported, unsupported, or missed.
3. Retrieve before/during/after frames around brief events from the recordings.
4. Compare prompts on identical frames with a fixed token budget.
5. Measure performance at different image resolutions.
6. Implement and measure concurrent capture with latest-frame inference.
7. Add temporal tracking or a lightweight event detector if rapid event
   detection is required.
8. Define how a whole-video summary will combine timestamped observations
   without treating uncertain model claims as verified events.

## Evidence Links

- [Benchmark implementation](../../experiments/moondream_4bit_camera_v2.py)
- [Capture metadata](../../results/camera_v2-2~2026-10-05/capture.json)
- [Complete generated report](../../results/camera_v2-2~2026-10-05/report.md)
- [Successful analysis JSON](../../results/camera_v2-2~2026-10-05/analysis.json)
- [Successful terminal log](../../results/camera_v2-2~2026-10-05/analysis_log.txt)
- [Successful RAM telemetry](../../results/camera_v2-2~2026-10-05/analysis_tegrastats.txt)
- [Failed analysis JSON](../../results/camera_v2-2~2026-10-05/analysis_prompt1.json)
- [Failed-attempt RAM telemetry](../../results/camera_v2-2~2026-10-05/analysis_prompt1_tegrastats.txt)
- [Video 1 samples and timestamps](../../results/camera_v2-2~2026-10-05/video_01/)
- [Video 2 samples and timestamps](../../results/camera_v2-2~2026-10-05/video_02/)
- [Video 3 samples and timestamps](../../results/camera_v2-2~2026-10-05/video_03/)
- [Video 4 samples and timestamps](../../results/camera_v2-2~2026-10-05/video_04/)
- [Video 5 samples and timestamps](../../results/camera_v2-2~2026-10-05/video_05/)

Full `camera.avi` recordings are retained locally in each video folder.
