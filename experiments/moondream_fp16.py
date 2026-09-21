import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from transformers import AutoModelForCausalLM
import transformers.modeling_utils as modeling_utils

ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "data" / "sample_images"
RESULTS_DIR = ROOT / "results"

MODEL_ID = "vikhyatk/moondream2"
REVISION = "2025-04-14"

RESULTS_DIR.mkdir(exist_ok=True)

print("Loading Moondream2 FP16 baseline...", flush=True)

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
start = time.perf_counter()


# Skip Transformers' optional allocator warm-up on memory-constrained Jetson.
# This affects loading strategy only, not model inference.
modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    revision=REVISION,
    trust_remote_code=True,
    dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map={"": "cuda"},
)
model.eval()
torch.cuda.synchronize()

load_time = time.perf_counter() - start

print("\nModel loaded")
print("Model:", MODEL_ID)
print("Revision:", REVISION)
print("Precision:", next(model.parameters()).dtype)
print("Device:", next(model.parameters()).device)
print(f"Load time: {load_time:.2f} seconds")
print(
    "Model memory allocated:",
    f"{torch.cuda.memory_allocated() / 1024**3:.2f} GiB",
)


def timed_inference(name, operation):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()

    with torch.inference_mode():
        output = operation()

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    print(f"{name} time: {elapsed:.2f} seconds")
    print(
        f"{name} peak memory:",
        f"{torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB",
    )

    return output


# 1. Normal caption — guide section 4.3
bird = Image.open(IMAGE_DIR / "bird.png").convert("RGB")

caption_result = timed_inference(
    "Normal caption",
    lambda: model.caption(bird, length="normal"),
)

print("\nNormal caption:")
print(caption_result["caption"])


# 2. Pothole VQA — guide section 4.4
pothole = Image.open(IMAGE_DIR / "potholes.png").convert("RGB")
pothole_question = "How many potholes are there in the image?"

pothole_result = timed_inference(
    "Pothole VQA",
    lambda: model.query(pothole, pothole_question),
)

print(f"\nVisual query: {pothole_question}")
print(pothole_result["answer"])


# 3. Cable-trip VQA — guide section 4.5
cable = Image.open(IMAGE_DIR / "cable-trip.jpg").convert("RGB")
cable_question = "Why is the person falling?"

cable_result = timed_inference(
    "Cable-trip VQA",
    lambda: model.query(cable, cable_question),
)

print(f"\nVisual query: {cable_question}")
print(cable_result["answer"])


# 4. Face detection — guide section 4.6
driving = Image.open(IMAGE_DIR / "driving-gaze.png").convert("RGB")

detection_result = timed_inference(
    "Face detection",
    lambda: model.detect(driving, "face"),
)

objects = detection_result["objects"]
print(f"\nFound {len(objects)} face(s)")

# Draw detected bounding boxes and save the result.
annotated = driving.copy()
draw = ImageDraw.Draw(annotated)
width, height = annotated.size

for box in objects:
    x_min = int(box["x_min"] * width)
    y_min = int(box["y_min"] * height)
    x_max = int(box["x_max"] * width)
    y_max = int(box["y_max"] * height)

    draw.rectangle(
        [x_min, y_min, x_max, y_max],
        outline="green",
        width=3,
    )
    draw.text((x_min, max(0, y_min - 15)), "Face", fill="green")

output_path = RESULTS_DIR / "driving-gaze-detected.jpg"
annotated.save(output_path)

print("Annotated image saved to:", output_path)
