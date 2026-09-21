import json
import time
from pathlib import Path

import transformers.modeling_utils as modeling_utils
import torch
from PIL import Image, ImageDraw
from transformers import AutoModelForCausalLM


ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "data" / "sample_images"
RESULTS_DIR = ROOT / "results"
RENDER_DIR = RESULTS_DIR / "4bit_rendered"

MODEL_ID = "moondream/moondream-2b-2025-04-14-4bit"

RESULTS_DIR.mkdir(exist_ok=True)
RENDER_DIR.mkdir(exist_ok=True)

metrics = {
    "configuration": "official_4bit",
    "model": MODEL_ID,
    "torchao_version": __import__("torchao").__version__,
    "tasks": [],
}


def timed_call(name, operation):
    torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.inference_mode():
        result = operation()

    torch.cuda.synchronize()
    latency = time.perf_counter() - start

    metrics["tasks"].append({
        "task": name,
        "latency_seconds": round(latency, 4),
    })

    print(f"Latency: {latency:.4f} seconds")
    return result


print("Loading Moondream2 OFFICIAL 4-BIT...", flush=True)

free_before, total_memory = torch.cuda.mem_get_info()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None

torch.cuda.synchronize()
load_start = time.perf_counter()

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    device_map={"": "cuda"},
)

model.eval()
torch.cuda.synchronize()
load_time = time.perf_counter() - load_start

free_after_load, _ = torch.cuda.mem_get_info()

metrics["load_time_seconds"] = round(load_time, 4)
metrics["cuda_free_before_gib"] = round(free_before / 1024**3, 4)
metrics["cuda_free_after_load_gib"] = round(free_after_load / 1024**3, 4)
metrics["cuda_consumed_after_load_gib"] = round(
    (free_before - free_after_load) / 1024**3, 4
)
metrics["model_memory_footprint_gib"] = round(
    model.get_memory_footprint() / 1024**3, 4
)

print("\nOFFICIAL 4-BIT MODEL LOADED")
print(f"Load time: {load_time:.4f} seconds")
print(f"Model footprint: {metrics['model_memory_footprint_gib']:.4f} GiB")
print(f"CUDA consumed after load: {metrics['cuda_consumed_after_load_gib']:.4f} GiB")


# Guide tasks 1–3: bird captions
bird = Image.open(IMAGE_DIR / "bird.png").convert("RGB")

print("\n=== SHORT CAPTION: bird.png ===")
short_caption = timed_call(
    "bird_short_caption",
    lambda: model.caption(bird, length="short")["caption"],
)
print(short_caption)

print("\n=== NORMAL CAPTION: bird.png ===")
normal_caption = timed_call(
    "bird_normal_caption",
    lambda: model.caption(bird, length="normal")["caption"],
)
print(normal_caption)

print("\n=== LONG CAPTION: bird.png ===")
long_caption = timed_call(
    "bird_long_caption",
    lambda: model.caption(bird, length="long")["caption"],
)
print(long_caption)


# Guide task 4: pothole question
potholes = Image.open(IMAGE_DIR / "potholes.png").convert("RGB")
pothole_question = "How many potholes are there in the image?"

print(f"\n=== QUERY: {pothole_question} ===")
pothole_answer = timed_call(
    "potholes_query",
    lambda: model.query(potholes, pothole_question)["answer"],
)
print(pothole_answer)


# Guide task 5: cable-trip question
cable_trip = Image.open(IMAGE_DIR / "cable-trip.jpg").convert("RGB")
cable_question = "Why is the person falling?"

print(f"\n=== QUERY: {cable_question} ===")
cable_answer = timed_call(
    "cable_trip_query",
    lambda: model.query(cable_trip, cable_question)["answer"],
)
print(cable_answer)


# Guide task 6: face detection
driving = Image.open(IMAGE_DIR / "driving-gaze.png").convert("RGB")

print("\n=== OBJECT DETECTION: face ===")
objects = timed_call(
    "driving_gaze_face_detection",
    lambda: model.detect(driving, "face")["objects"],
)
print(f"Found {len(objects)} face(s)")
print(objects)

face_render = driving.copy()
face_draw = ImageDraw.Draw(face_render)
width, height = face_render.size

for box in objects:
    x_min = int(box["x_min"] * width)
    y_min = int(box["y_min"] * height)
    x_max = int(box["x_max"] * width)
    y_max = int(box["y_max"] * height)

    face_draw.rectangle(
        [x_min, y_min, x_max, y_max],
        outline="green",
        width=3,
    )
    face_draw.text(
        (x_min, max(0, y_min - 15)),
        "Face",
        fill="green",
    )

face_output = RENDER_DIR / "driving-gaze_faces.png"
face_render.save(face_output)
print("Saved:", face_output)


# Guide task 7: point to people
print("\n=== POINTING: person ===")
points = timed_call(
    "driving_gaze_person_pointing",
    lambda: model.point(driving, "person")["points"],
)
print(f"Found {len(points)} person(s)")
print(points)

point_render = driving.copy()
point_draw = ImageDraw.Draw(point_render)
radius = 8

for point in points:
    x = int(point["x"] * width)
    y = int(point["y"] * height)

    point_draw.ellipse(
        [x - radius, y - radius, x + radius, y + radius],
        fill="red",
        outline="white",
        width=2,
    )
    point_draw.text(
        (x + 10, y - 10),
        "Person",
        fill="red",
    )

point_output = RENDER_DIR / "driving-gaze_people.png"
point_render.save(point_output)
print("Saved:", point_output)


torch.cuda.synchronize()

metrics["pytorch_allocated_gib"] = round(
    torch.cuda.memory_allocated() / 1024**3, 4
)
metrics["pytorch_peak_allocated_gib"] = round(
    torch.cuda.max_memory_allocated() / 1024**3, 4
)

metrics["outputs"] = {
    "short_caption": short_caption,
    "normal_caption": normal_caption,
    "long_caption": long_caption,
    "potholes_answer": pothole_answer,
    "cable_trip_answer": cable_answer,
    "face_detections": objects,
    "person_points": points,
}

metrics_path = RESULTS_DIR / "moondream_4bit_metrics.json"

with metrics_path.open("w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=2)

print("\n=== FINAL MEMORY ===")
print(f"PyTorch allocated: {metrics['pytorch_allocated_gib']:.4f} GiB")
print(f"PyTorch peak allocated: {metrics['pytorch_peak_allocated_gib']:.4f} GiB")
print("Metrics saved:", metrics_path)
print("\nOFFICIAL 4-BIT IMAGE BENCHMARK COMPLETED")
