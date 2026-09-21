# Moondream2 Jetson Benchmarks — 2026-09-21

## Goal

Compare model size, memory use, and image-inference latency on the 8 GB Jetson Orin Nano. The image tasks are short, normal, and long bird captions; a pothole question; a cable-trip question; face detection; and person pointing.

## Results

| Measurement | FP16 baseline | BitsAndBytes INT8 | Official 4-bit |
|---|---:|---:|---:|
| Checkpoint size | ~3.6 GiB | Not saved as a checkpoint | 1.89 GiB |
| Loaded model footprint | ~3.59 GiB | 1.91 GiB | 1.8969 GiB |
| Peak system RAM | Not captured | See INT8 logs; no completed image run | 5,504 / 7,620 MB |
| Peak PyTorch CUDA allocation | Not captured | Not captured for completed inference | 2.5718 GiB |
| Cached load time | Not captured | 51.03 s | 25.9449 s |
| Completed image tasks | 0 | 0 | 7 of 7 |

FP16 loaded once but repeatedly failed with NVIDIA memory-allocation errors. Generic whole-model BitsAndBytes INT8 loaded and reduced the model footprint, but its first caption failed because Moondream's vision code used an INT8 weight with an FP16 input (`Half` versus `Char`). Neither configuration produced image-inference latency.

The official 4-bit checkpoint loaded and completed every image task. Transformers' CUDA allocator warmup was skipped because it failed on this Jetson. A run with less free memory also failed while allocating the generation cache; the completed benchmark began with approximately 5.3 GiB CUDA memory free. NVIDIA allocation warnings still appeared during the successful run.

## Official 4-bit task latencies

Each time includes image processing and generation for that task. The first caption also includes first-use setup, so these are individual task times, not a steady-state throughput measurement.

| Task | Latency |
|---|---:|
| Bird short caption | 14.0626 s |
| Bird normal caption | 13.0010 s |
| Bird long caption | 19.4479 s |
| Pothole question | 5.9810 s |
| Cable-trip question | 6.0659 s |
| Face detection | 5.6811 s |
| Person pointing | 5.0362 s |

## Outputs

- Short bird caption: “A hummingbird hovers near a tall flower with orange and red blossoms, displaying its green and gray plumage against a blurred green background.”
- Pothole answer: “There is one pothole in the image.”
- Cable-trip answer: “The person is falling because they have stepped on a loose electrical cord.”
- Face detection: one face.
- Person pointing: one person.

The normal and long captions, coordinates, and all task metrics are in [the metrics JSON](../../results/moondream_4bit_metrics.json). See the [full benchmark output](../../results/moondream_4bit_benchmark_retry.txt), [system RAM samples](../../results/moondream_4bit_tegrastats_retry.txt), and [rendered images](../../results/4bit_rendered/). The reproducible code is [moondream_4bit.py](../../experiments/moondream_4bit.py).

The 4-bit checkpoint size is the Hugging Face repository's `model.safetensors` file size. Peak system RAM comes from `tegrastats`; PyTorch CUDA allocation is a different measurement and should not be added to system RAM on this unified-memory device.
