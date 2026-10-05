import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "moondream/moondream-2b-2025-04-14-4bit"
REVISION = "a89c59223ef8b5bb7826780728eeec172727ca84"
PROMPT = (
    "Inspect this construction image for visible safety concerns. "
    "Describe the workers, equipment, and activity. "
    "Check for objects or cables obstructing walking paths, uneven ground, "
    "workers near machinery, and people visibly stumbling or falling. "
    "For each clearly visible concern, state what it is and where it appears. "
    "If no concern is clearly visible, say so. "
    "Do not assume movement, missing protective equipment, or hidden hazards. "
    "Answer in at most three concise sentences without repetition."
)
SETTINGS = {"max_tokens": 96, "temperature": 0}
SAMPLES = 6


def save_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def capture(run):
    pipeline = (
        "nvarguscamerasrc sensor-id=0 ! "
        "video/x-raw(memory:NVMM),width=1280,height=720,"
        "framerate=30/1,format=NV12 ! "
        "nvvidconv ! video/x-raw,width=640,height=360,format=BGRx ! "
        "videoconvert ! video/x-raw,format=BGR ! "
        "appsink drop=true max-buffers=1 sync=false"
    )
    manifest_path = run / "capture.json"
    if manifest_path.exists():
        raise RuntimeError("This run already has captures. Use a new run name.")

    manifest = {
        "design": "CSI recording followed by independent frame analysis",
        "requested_videos": 5,
        "requested_frames_per_video": SAMPLES,
        "camera_output": "640x360",
        "videos": [],
    }
    save_json(manifest_path, manifest)
    camera = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not camera.isOpened():
        raise RuntimeError("CSI camera could not open.")

    writer = None
    try:
        for number in range(1, 6):
            label = input(f"\nVideo {number}: enter title or filename: ").strip()
            source = input("Source URL or identifier (optional): ").strip()
            duration = float(input("Playback duration in seconds: "))
            if duration <= 0:
                raise ValueError("Duration must be positive.")

            folder = run / f"video_{number:02d}"
            folder.mkdir()
            video = {
                "number": number,
                "label": label,
                "source": source,
                "duration_requested_seconds": duration,
                "status": "preview",
                "frames": [],
            }
            manifest["videos"].append(video)
            save_json(manifest_path, manifest)

            print("Align camera. Pause video at its beginning.")
            print("Press S in preview, then immediately start playback. Q quits.")
            while True:
                ok, frame = camera.read()
                if not ok:
                    raise RuntimeError("Camera stopped during preview.")
                preview = frame.copy()
                cv2.putText(
                    preview, "S: start capture | Q: quit",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 2,
                )
                cv2.imshow("CSI benchmark", preview)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    raise KeyboardInterrupt
                if key == ord("s"):
                    break

            height, width = frame.shape[:2]
            recording = folder / "camera.avi"
            writer = cv2.VideoWriter(
                str(recording),
                cv2.VideoWriter_fourcc(*"MJPG"),
                30.0, (width, height),
            )
            if not writer.isOpened():
                raise RuntimeError("Recording writer could not open.")

            targets = [(i + 0.5) * duration / SAMPLES for i in range(SAMPLES)]
            video["recording"] = str(recording.relative_to(run))
            video["recording_nominal_fps"] = 30
            video["status"] = "capturing"
            start = time.perf_counter()
            count = 0
            with (folder / "recording_timestamps.csv").open("w") as timestamps:
                timestamps.write("recorded_frame,elapsed_seconds,read_seconds\n")
                while True:
                    read_start = time.perf_counter()
                    ok, frame = camera.read()
                    now = time.perf_counter()
                    if not ok:
                        raise RuntimeError("Camera stopped during recording.")
                    elapsed = now - start
                    read_latency = now - read_start
                    if elapsed >= duration:
                        break

                    writer.write(frame)
                    count += 1
                    timestamps.write(f"{count},{elapsed:.6f},{read_latency:.6f}\n")

                    index = len(video["frames"])
                    if index < SAMPLES and elapsed >= targets[index]:
                        path = folder / f"frame_{index + 1:03d}.jpg"
                        if not cv2.imwrite(str(path), frame):
                            raise RuntimeError(f"Could not save {path}")
                        video["frames"].append({
                            "number": index + 1,
                            "path": str(path.relative_to(run)),
                            "target_seconds": targets[index],
                            "elapsed_seconds": elapsed,
                            "sampling_lateness_seconds": elapsed - targets[index],
                            "camera_read_seconds": read_latency,
                            "captured_utc": datetime.now(timezone.utc).isoformat(),
                        })
                        save_json(manifest_path, manifest)
                        print(f"Saved sample {index + 1}/6 at {elapsed:.2f}s",
                              flush=True)

                    preview = frame.copy()
                    cv2.putText(
                        preview, f"Video {number}: {elapsed:.1f}/{duration:.1f}s",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0), 2,
                    )
                    cv2.imshow("CSI benchmark", preview)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        raise KeyboardInterrupt

            writer.release()
            writer = None
            video["recorded_frames"] = count
            video["capture_wall_seconds"] = time.perf_counter() - start
            video["status"] = (
                "complete" if len(video["frames"]) == SAMPLES else "incomplete"
            )
            save_json(manifest_path, manifest)
            print(f"Video {number} saved. Stop playback.", flush=True)
    finally:
        if writer is not None:
            writer.release()
        camera.release()
        cv2.destroyAllWindows()
        save_json(manifest_path, manifest)

    print(f"\nCapture finished: {run}")
    print("The AVI uses nominal 30 FPS; CSV timestamps preserve actual timing.")


def analyze(run):
    import statistics
    import importlib.metadata
    import torch
    import transformers.modeling_utils as modeling_utils
    from PIL import Image
    from transformers import AutoModelForCausalLM

    manifest = json.loads((run / "capture.json").read_text())
    path = run / "analysis.json"
    if path.exists():
        raise RuntimeError("Analysis already exists; preserve it before rerunning.")

    modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    free_before, _ = torch.cuda.mem_get_info()
    print("Loading official 4-bit model...", flush=True)
    start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION,
        trust_remote_code=True, device_map={"": "cuda"},
    ).eval()
    torch.cuda.synchronize()
    free_after, _ = torch.cuda.mem_get_info()
    results = {
        "model": MODEL_ID,
        "revision": REVISION,
        "torch": torch.__version__,
        "torchao": importlib.metadata.version("torchao"),
        "transformers": importlib.metadata.version("transformers"),
        "prompt": PROMPT,
        "settings": SETTINGS,
        "load_seconds": time.perf_counter() - start,
        "model_footprint_gib": model.get_memory_footprint() / 1024**3,
        "cuda_free_before_load_gib": free_before / 1024**3,
        "cuda_free_after_load_gib": free_after / 1024**3,
        "cuda_free_difference_gib": (free_before - free_after) / 1024**3,
        "status": "running",
        "frames": [],
    }
    save_json(path, results)
    try:
        for video in manifest["videos"]:
            for sample in video["frames"]:
                with Image.open(run / sample["path"]) as opened:
                    image = opened.convert("RGB")
                torch.cuda.synchronize()
                started = time.perf_counter()
                with torch.inference_mode():
                    answer = model.query(
                        image, PROMPT, settings=SETTINGS
                    )["answer"].strip()
                torch.cuda.synchronize()
                latency = time.perf_counter() - started
                results["frames"].append({
                    "video_number": video["number"],
                    "video_label": video["label"],
                    **sample,
                    "description": answer,
                    "inference_seconds": latency,
                    "cuda_allocated_gib":
                        torch.cuda.memory_allocated() / 1024**3,
                    "cuda_peak_allocated_gib":
                        torch.cuda.max_memory_allocated() / 1024**3,
                })
                save_json(path, results)
                print(
                    f"\nVideo {video['number']} frame {sample['number']}: "
                    f"{latency:.2f}s\n{answer}", flush=True,
                )
        results["status"] = "finished"
    except KeyboardInterrupt:
        results["status"] = "interrupted"
        print("\nInterrupted; completed results preserved.", flush=True)
    except Exception as error:
        results["status"] = "failed"
        results["error"] = repr(error)
        raise
    finally:
        results["cuda_peak_allocated_gib"] = (
            torch.cuda.max_memory_allocated() / 1024**3
        )
        results["video_statistics"] = []
        report = [
            "# CSI camera benchmark",
            "",
            f"Prompt: {PROMPT}",
            f"Settings: {SETTINGS}",
            "",
            "Descriptions are independent observations, not verified events.",
        ]
        for video in manifest["videos"]:
            rows = [
                row for row in results["frames"]
                if row["video_number"] == video["number"]
            ]
            times = [row["inference_seconds"] for row in rows]
            stats = {
                "video_number": video["number"],
                "completed_frames": len(rows),
                "mean_inference_seconds":
                    statistics.mean(times) if times else None,
                "total_inference_seconds": sum(times),
            }
            results["video_statistics"].append(stats)
            report.extend([
                "", f"## Video {video['number']}: {video['label']}",
                f"Completed frames: {len(rows)}/6",
                f"Mean inference seconds: {stats['mean_inference_seconds']}",
                f"Total inference seconds: {sum(times):.4f}", "",
            ])
            for row in rows:
                report.append(
                    f"- {row['elapsed_seconds']:.2f}s: "
                    f"{row['description']} "
                    f"(inference: {row['inference_seconds']:.2f}s)"
                )
        save_json(path, results)
        (run / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"\nResults: {path}\nReport: {run / 'report.md'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["capture", "analyze"])
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    run = Path(args.run).resolve()
    run.mkdir(parents=True, exist_ok=True)
    if args.mode == "capture":
        capture(run)
    else:
        analyze(run)
