# Moondream 4-Bit Live CSI Camera Test — September 28, 2026

## Objective

Evaluate whether the official 4-bit Moondream model can process frames captured from a live NVIDIA Jetson CSI camera while measuring model size, system memory usage, CUDA memory usage, and end-to-end latency.

This was a partial feasibility test using a construction video displayed in front of the CSI camera. The planned experiment included six frames, but the run was stopped after two frames because the second construction-specific response entered a repetition loop and took significantly longer to finish.

## Configuration

- Hardware: NVIDIA Jetson Orin Nano
- System memory: 7,620 MB
- Model: `moondream/moondream-2b-2025-04-14-4bit`
- Quantization: Official TorchAO 4-bit checkpoint
- Frameworks: PyTorch, Hugging Face Transformers, and TorchAO
- Camera: NVIDIA CSI camera
- Camera resolution: 1280 × 720
- Reported camera rate: 30 FPS
- Model load time: 26.0223 seconds
- Model memory footprint: 1.8969 GiB
- CUDA memory consumed after loading: 1.7900 GiB
- Peak PyTorch allocation: Not recorded because the run was interrupted
- Peak system RAM: 6,004 MB of 7,620 MB
- Peak system RAM utilization: 78.79%

Peak system RAM utilization was calculated as:

```text
6,004 / 7,620 × 100 = 78.79%
```

## Source Video

Construction Video 1 from the mentor-provided research dataset:

https://drive.google.com/file/d/1Kh8cFl-OckC_hMw1OA0Ld7OBrvnlOBMn/view

The construction video was displayed on another screen and recorded by the Jetson CSI camera. The model did not process the source video as a continuous video sequence. Instead, the camera captured individual frames, and Moondream analyzed each captured frame independently.

Therefore, this experiment tested live visual sampling rather than temporal video understanding.

## Experiment Flow

For every planned sample, the benchmark was designed to:

1. Capture the latest frame from the CSI camera.
2. Save the frame as a JPEG image.
3. Generate a generic short caption.
4. submit the same frame with a construction-specific prompt.
5. Record camera capture latency.
6. Record generic-caption latency.
7. Record construction-query latency.
8. Record combined inference latency.
9. Record total end-to-end latency.
10. Wait until the next sampling interval.

Six samples were planned. Two samples completed before the experiment was manually stopped.

## Inference Tasks

### Task 1: Generic Short Caption

The first task used Moondream's short-caption method:

```python
model.caption(image, length="short")
```

This task did not use a custom written prompt. It requested a short general description of the captured image.

### Task 2: Construction-Specific Query

The second task used the following exact prompt:

> Describe what is happening in this construction scene. Identify visible workers, equipment, activities, and any potential safety concerns.

The purpose of this prompt was to determine whether a construction-specific request provided more useful jobsite information than generic image captioning.

## Benchmark Results

| Metric | Frame 1 | Frame 2 |
|---|---:|---:|
| Capture timestamp | 2026-09-28 23:34:53 UTC | 2026-09-28 23:35:22 UTC |
| Approximate video time | 0.0010 s | 29.5286 s |
| Camera capture latency | 0.0010 s | 0.0006 s |
| Generic-caption latency | 14.2960 s | 6.9566 s |
| Construction-query latency | 14.8674 s | 78.3490 s |
| Combined model latency | 29.1634 s | 85.3056 s |
| End-to-end latency | 29.5277 s | 85.3212 s |

### Two-Frame Summary

- Average camera capture latency: 0.0008 seconds
- Average generic-caption latency: 10.6263 seconds
- Average construction-query latency: 46.6082 seconds
- Average combined model latency: 57.2345 seconds
- Average end-to-end latency: 57.4245 seconds

These averages describe only the two completed frames. They should not be treated as final benchmark averages for the planned six-frame experiment.

## Frame 1

### Capture Information

- Saved frame: `results/camera_frames/frame_001.jpg`
- Capture timestamp: `2026-09-28T23:34:53.047792+00:00`
- Approximate source-video time: 0.0010 seconds
- Camera capture latency: 0.0010 seconds

### Generic Caption

Latency: 14.2960 seconds

Returned output:

> A red excavator, equipped with a bucket and shovel, is parked on a dirt lot, with a building and trees in the background.

### Construction-Specific Query

Latency: 14.8674 seconds

Returned output:

> A red excavator is parked on a dirt lot, partially obscured by a fisheye lens. The excavator is positioned in a construction site, with some buildings and trees visible in the background. A worker is visible in the cab of the excavator, operating the machinery. The excavator appears to be parked in a muddy or wet area, and there are some safety concerns, such as the possibility of slipping on the wet ground. The excavator's operator is wearing safety gear, including a helmet and protective clothing.

### Frame 1 Observations

The generic caption correctly focused on the primary piece of visible equipment and produced a concise answer.

The construction-specific response included additional contextual claims about:

- A worker operating the excavator
- Wet or muddy ground
- A possible slipping hazard
- A helmet and protective clothing

These claims require manual verification against `frame_001.jpg`. The prompt encouraged the model to identify workers and safety concerns, so it may have inferred details that were unclear or not actually visible.

The two inference tasks required 29.1634 seconds in total. Including capture and processing overhead, the end-to-end latency was 29.5277 seconds.

## Frame 2

### Capture Information

- Saved frame: `results/camera_frames/frame_002.jpg`
- Capture timestamp: `2026-09-28T23:35:22.575307+00:00`
- Approximate source-video time: 29.5286 seconds
- Camera capture latency: 0.0006 seconds

### Generic Caption

Latency: 6.9566 seconds

Returned output:

> A red and black Kobalt XK1654 mini excavator digs into a pile of dirt in a residential backyard.

### Construction-Specific Query

Latency: 78.3490 seconds

The response began with:

> A small orange excavator is parked on a residential property, digging into the ground. A worker is visible in the driver's seat, operating the excavator. The excavator is positioned in a residential area, with a house and some landscaping visible in the background. The excavator is equipped with a bucket attachment, and the worker appears to be using a shovel to move dirt. The excavator is parked on a paved area, and there is some dirt and debris visible in the excavator's bucket.

After this opening, the response repeatedly generated variations of the same claims about:

- The excavator
- The paved area
- The house
- The operator
- Dirt and debris in the bucket

The output continued repeating until it was truncated in the benchmark log. The complete unedited generated response is preserved in:

`results/moondream_4bit_camera_benchmark_retry3.txt`

### Frame 2 Observations

The generic caption remained concise and completed in 6.9566 seconds.

The construction-specific response entered a severe repetition loop and required 78.3490 seconds. This was approximately:

- 11.26 times slower than the generic caption for the same frame
- 5.27 times slower than the Frame 1 construction query

The repeated generation caused the total end-to-end latency for Frame 2 to increase to 85.3212 seconds.

This result demonstrates that generated output length and repetition can create highly unpredictable latency, even when the model successfully fits in memory.

## Memory Behavior

The model's reported memory footprint was 1.8969 GiB, confirming that the official 4-bit checkpoint was loaded.

The benchmark consumed 1.7900 GiB of additional CUDA-visible memory immediately after loading the model.

During the partial experiment, `tegrastats` measured a peak system RAM usage of:

```text
6,004 MB of 7,620 MB
```

This represents approximately 78.79% of the Jetson's available system memory.

The Jetson uses unified memory, meaning the CPU, GPU, camera pipeline, and other processes share the same physical system memory. The reported model footprint, CUDA allocation, and system RAM usage should therefore not be added together as if they represented completely separate memory pools.

The run also produced repeated messages such as:

```text
NvMapMemAllocInternalTagged: error 12
NvMapMemHandleAlloc: error 0
```

The NVIDIA Argus camera pipeline also reported an `InsufficientMemory` error while initializing or copying camera buffers. Despite these warnings, two camera frames were captured and processed successfully.

These messages indicate that the model and camera pipeline were operating under memory pressure, even though peak system RAM did not reach the full 7,620 MB capacity.

## Primary Findings

1. The official 4-bit Moondream checkpoint successfully loaded on the Jetson Orin Nano.
2. The complete CSI camera-to-model inference path worked for two live frames.
3. Camera capture latency was negligible compared with model inference latency.
4. Generic short captions completed in approximately 7–14 seconds.
5. The construction-specific query completed normally for Frame 1 but entered a repetition loop for Frame 2.
6. The repetition loop increased query latency from approximately 15 seconds to more than 78 seconds.
7. Peak system RAM reached 6,004 MB, or approximately 78.79% of available memory.
8. NVIDIA camera and memory-allocation warnings show that the combined model and CSI pipeline remained under memory pressure.
9. The test processed independent frames and did not perform continuous-video or temporal reasoning.
10. The experiment was stopped after two of six planned frames and must be labeled as a partial feasibility test.

## Limitations

- Only two of the six planned frames completed.
- The camera was held manually instead of using a fixed mount.
- The source video was recorded from another screen, which introduced screen borders, viewing angles, lighting differences, and possible image distortion.
- The model analyzed independent frames without information from previous frames.
- The construction prompt did not enforce an output-length limit.
- The long repetitive output distorted the average query and end-to-end latency.
- Some generated details may have been inferred or hallucinated and require comparison with the saved frames.
- Final PyTorch peak-allocation metrics were not written because the experiment was interrupted before normal program completion.

## Recommended Next Steps

1. Add a supported generation-token limit to prevent runaway responses.
2. Change the prompt to explicitly request no more than three concise sentences.
3. Test one task per captured frame rather than performing both captioning and querying on every frame.
4. Mount the CSI camera so the video remains aligned without being held manually.
5. Add an on-screen preview before beginning the timed benchmark.
6. Separate camera initialization from timed frame capture.
7. Record metrics after every frame so partial runs remain machine-readable.
8. Add graceful interrupt handling so metrics are saved when `Ctrl+C` is used.
9. Manually label visible equipment, workers, activities, and safety concerns for accuracy evaluation.
10. Repeat the test after limiting response length and reducing camera-buffer memory pressure.

## Evidence Files

- Benchmark script: `experiments/moondream_4bit_camera.py`
- Raw benchmark output: `results/moondream_4bit_camera_benchmark_retry3.txt`
- Raw memory telemetry: `results/moondream_4bit_camera_tegrastats_retry3.txt`
- First captured frame: `results/camera_frames/frame_001.jpg`
- Second captured frame: `results/camera_frames/frame_002.jpg`

## Conclusion

This partial experiment established that an official 4-bit vision-language model can load and process live CSI camera frames on an 8 GB Jetson Orin Nano.

The primary performance limitation was not frame capture. It was model inference—particularly unconstrained text generation. A normal short caption required approximately 7–14 seconds, while one repetitive construction response required more than 78 seconds.

The next experiment should focus on controlling response length, reducing memory pressure, and saving results after every frame. These changes should make the live-camera benchmark more reliable and produce latency measurements that better represent a practical construction-jobsite deployment.
