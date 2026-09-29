import csv
import importlib.metadata
import json
import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import torch
import transformers.modeling_utils as modeling_utils
from PIL import Image
from transformers import AutoModelForCausalLM


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FRAME_DIR = RESULTS_DIR / "camera_frames"

CSV_PATH = RESULTS_DIR / "moondream_4bit_camera_metrics.csv"
SUMMARY_PATH = RESULTS_DIR / "moondream_4bit_camera_summary.json"

MODEL_ID = "moondream/moondream-2b-2025-04-14-4bit"
TOTAL_FRAMES = 6

VIDEO_NUMBER = 1
VIDEO_URL = (
    "https://drive.google.com/file/d/"
    "1Kh8cFl-OckC_hMw1OA0Ld7OBrvnlOBMn/view"
)
VIDEO_LABEL = "Construction video 1 from mentor-provided dataset"

CONSTRUCTION_PROMPT = (
    "Describe what is happening in this construction scene. "
    "Identify visible workers, equipment, activities, "
    "and any potential safety concerns."
)

RESULTS_DIR.mkdir(exist_ok=True)
FRAME_DIR.mkdir(exist_ok=True)

# Required workaround for the Jetson allocator-warmup failure.
modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None


def calculate_stats(values):
    sorted_values = sorted(values)
    p95_index = math.ceil(0.95 * len(sorted_values)) - 1

    return {
        "minimum": round(min(values), 4),
        "maximum": round(max(values), 4),
        "mean": round(statistics.mean(values), 4),
        "median": round(statistics.median(values), 4),
        "p95": round(sorted_values[p95_index], 4),
    }


print("Loading official Moondream 4-bit model...", flush=True)

free_before, total_cuda = torch.cuda.mem_get_info()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

load_start = time.perf_counter()

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    device_map={"": "cuda"},
)

model.eval()
torch.cuda.synchronize()

load_time = time.perf_counter() - load_start
free_after_load, _ = torch.cuda.mem_get_info()

model_footprint = model.get_memory_footprint() / 1024**3
cuda_consumed = (free_before - free_after_load) / 1024**3

print("\nMODEL LOADED")
print(f"Load time: {load_time:.4f} seconds")
print(f"Model footprint: {model_footprint:.4f} GiB")
print(f"CUDA consumed after load: {cuda_consumed:.4f} GiB")

input(
    "\nModel loaded successfully. Start construction Video 1 "
    "from the beginning, then press Enter to begin camera capture..."
)


camera_pipeline = (
    "nvarguscamerasrc sensor-id=0 ! "
    "video/x-raw(memory:NVMM),width=1280,height=720,"
    "framerate=30/1,format=NV12 ! "
    "nvvidconv ! "
    "video/x-raw,width=1280,height=720,format=BGRx ! "
    "videoconvert ! "
    "video/x-raw,format=BGR ! "
    "appsink drop=true max-buffers=1 sync=false"
)

print("\nOpening NVIDIA CSI camera...", flush=True)

camera = cv2.VideoCapture(
    camera_pipeline,
    cv2.CAP_GSTREAMER,
)

if not camera.isOpened():
    raise RuntimeError("Could not open the NVIDIA CSI camera")

actual_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
actual_fps = camera.get(cv2.CAP_PROP_FPS)

print(
    f"Camera opened: {actual_width}x{actual_height}, "
    f"reported FPS: {actual_fps:.2f}",
    flush=True,
)
print("Camera pipeline initialized.", flush=True)

print("\nStarting paired live-frame tests...")
print("Task 1: Generic short caption")
print(f"Task 2 prompt: {CONSTRUCTION_PROMPT}\n")

rows = []
test_start = time.perf_counter()

try:
    for frame_number in range(1, TOTAL_FRAMES + 1):
        capture_start = time.perf_counter()
        ok, frame = camera.read()

        if not ok:
            raise RuntimeError(
                f"Camera failed to capture frame {frame_number}"
            )

        captured_perf = time.perf_counter()
        captured_utc = datetime.now(timezone.utc).isoformat()

        capture_latency = captured_perf - capture_start
        video_elapsed = captured_perf - test_start

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )
        image = Image.fromarray(rgb_frame)

        frame_path = FRAME_DIR / f"frame_{frame_number:03d}.jpg"
        cv2.imwrite(str(frame_path), frame)

        # Task 1: generic short caption with no written prompt.
        torch.cuda.synchronize()
        caption_start = time.perf_counter()

        with torch.inference_mode():
            generic_caption = model.caption(
                image,
                length="short",
            )["caption"].strip()

        torch.cuda.synchronize()
        caption_latency = time.perf_counter() - caption_start

        # Task 2: construction-specific visual question.
        torch.cuda.synchronize()
        query_start = time.perf_counter()

        with torch.inference_mode():
            construction_description = model.query(
                image,
                CONSTRUCTION_PROMPT,
            )["answer"].strip()

        torch.cuda.synchronize()
        completed_perf = time.perf_counter()

        query_latency = completed_perf - query_start
        combined_model_latency = caption_latency + query_latency
        end_to_end_latency = completed_perf - capture_start

        row = {
            "frame_number": frame_number,
            "video_number": VIDEO_NUMBER,
            "video_label": VIDEO_LABEL,
            "video_url": VIDEO_URL,
            "captured_utc": captured_utc,
            "approximate_video_elapsed_seconds": round(
                video_elapsed,
                4,
            ),
            "camera_capture_latency_seconds": round(
                capture_latency,
                4,
            ),
            "frame_path": str(frame_path.relative_to(ROOT)),
            "generic_caption_method": (
                'model.caption(image, length="short")'
            ),
            "generic_caption_prompt": "None",
            "generic_caption": generic_caption,
            "generic_caption_latency_seconds": round(
                caption_latency,
                4,
            ),
            "construction_query_prompt": CONSTRUCTION_PROMPT,
            "construction_query_answer": construction_description,
            "construction_query_latency_seconds": round(
                query_latency,
                4,
            ),
            "combined_model_latency_seconds": round(
                combined_model_latency,
                4,
            ),
            "end_to_end_latency_seconds": round(
                end_to_end_latency,
                4,
            ),
        }

        rows.append(row)

        print(f"=== FRAME {frame_number}/{TOTAL_FRAMES} ===")
        print(f"Captured: {captured_utc}")
        print(
            f"Approximate video time: "
            f"{video_elapsed:.4f} seconds"
        )
        print(
            f"Camera capture latency: "
            f"{capture_latency:.4f} seconds"
        )
        print(f"Saved: {frame_path}")
        print(
            f"Generic caption latency: "
            f"{caption_latency:.4f} seconds"
        )
        print(f"Generic caption: {generic_caption}")
        print(
            f"Construction query latency: "
            f"{query_latency:.4f} seconds"
        )
        print(
            "Construction description: "
            f"{construction_description}"
        )
        print(
            f"Combined model latency: "
            f"{combined_model_latency:.4f} seconds"
        )
        print(
            f"End-to-end latency: "
            f"{end_to_end_latency:.4f} seconds\n",
            flush=True,
        )

finally:
    camera.release()


with CSV_PATH.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=rows[0].keys(),
    )
    writer.writeheader()
    writer.writerows(rows)


capture_times = [
    row["camera_capture_latency_seconds"]
    for row in rows
]

caption_times = [
    row["generic_caption_latency_seconds"]
    for row in rows
]

query_times = [
    row["construction_query_latency_seconds"]
    for row in rows
]

combined_times = [
    row["combined_model_latency_seconds"]
    for row in rows
]

end_to_end_times = [
    row["end_to_end_latency_seconds"]
    for row in rows
]

summary = {
    "configuration": "official_4bit_live_csi_camera",
    "model": MODEL_ID,
    "checkpoint_size_gib": 1.89,
    "torchao_version": importlib.metadata.version("torchao"),
    "video_source": {
        "number": VIDEO_NUMBER,
        "label": VIDEO_LABEL,
        "url": VIDEO_URL,
    },
    "camera": {
        "type": "NVIDIA CSI",
        "sensor_id": 0,
        "resolution": f"{actual_width}x{actual_height}",
        "reported_fps": round(actual_fps, 2),
    },
    "experiment_design": {
        "sampled_live_frames": len(rows),
        "outputs_generated": len(rows) * 2,
        "generic_caption_method": (
            'model.caption(image, length="short")'
        ),
        "generic_caption_prompt": None,
        "construction_query_method": "model.query(image, prompt)",
        "construction_query_prompt": CONSTRUCTION_PROMPT,
        "temporal_reasoning": False,
        "frame_handling": (
            "Each camera frame was analyzed independently."
        ),
    },
    "load_time_seconds": round(load_time, 4),
    "model_footprint_gib": round(model_footprint, 4),
    "cuda_free_before_load_gib": round(
        free_before / 1024**3,
        4,
    ),
    "cuda_consumed_after_load_gib": round(
        cuda_consumed,
        4,
    ),
    "pytorch_final_allocated_gib": round(
        torch.cuda.memory_allocated() / 1024**3,
        4,
    ),
    "pytorch_peak_allocated_gib": round(
        torch.cuda.max_memory_allocated() / 1024**3,
        4,
    ),
    "camera_capture_latency_seconds": calculate_stats(
        capture_times
    ),
    "generic_caption_latency_seconds": calculate_stats(
        caption_times
    ),
    "construction_query_latency_seconds": calculate_stats(
        query_times
    ),
    "combined_model_latency_seconds": calculate_stats(
        combined_times
    ),
    "end_to_end_latency_seconds": calculate_stats(
        end_to_end_times
    ),
    "results_csv": str(CSV_PATH.relative_to(ROOT)),
    "frame_directory": str(FRAME_DIR.relative_to(ROOT)),
}

with SUMMARY_PATH.open("w", encoding="utf-8") as file:
    json.dump(summary, file, indent=2)


print("\n=== LIVE CAMERA BENCHMARK COMPLETE ===")
print(f"Frames analyzed: {len(rows)}")
print(f"Outputs generated: {len(rows) * 2}")
print(
    "Mean camera-capture latency: "
    f"{summary['camera_capture_latency_seconds']['mean']:.4f} "
    "seconds"
)
print(
    "Mean generic-caption latency: "
    f"{summary['generic_caption_latency_seconds']['mean']:.4f} "
    "seconds"
)
print(
    "Mean construction-query latency: "
    f"{summary['construction_query_latency_seconds']['mean']:.4f} "
    "seconds"
)
print(
    "Mean combined model latency: "
    f"{summary['combined_model_latency_seconds']['mean']:.4f} "
    "seconds"
)
print(
    "Mean end-to-end latency: "
    f"{summary['end_to_end_latency_seconds']['mean']:.4f} "
    "seconds"
)
print(
    "PyTorch peak allocation: "
    f"{summary['pytorch_peak_allocated_gib']:.4f} GiB"
)
print(f"CSV saved: {CSV_PATH}")
print(f"Summary saved: {SUMMARY_PATH}")
