"""Concurrent periodic image analysis; not verified temporal event detection.

Only main owns OpenCV capture/windows. One worker owns all model/CUDA calls.
The shared slot holds one latest frame; an in-flight frame is held separately.
"""
import argparse
import csv
import importlib.metadata
import json
import math
import re
import signal
import statistics
import subprocess
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

MODEL_ID = "moondream/moondream-2b-2025-04-14-4bit"
REVISION = "a89c59223ef8b5bb7826780728eeec172727ca84"


def save_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)

PROMPT = (
    "Identify clearly visible construction hazards in one short sentence. "
    "If none are clear, say so. Do not infer movement or hidden hazards."
)
SETTINGS = {"max_tokens": 32, "temperature": 0}
PIPELINE = (
    "nvarguscamerasrc sensor-id=0 ! "
    "video/x-raw(memory:NVMM),width=1280,height=720,"
    "framerate=30/1,format=NV12 ! "
    "nvvidconv ! video/x-raw,width=640,height=360,format=BGRx ! "
    "videoconvert ! video/x-raw,format=BGR ! "
    "appsink drop=true max-buffers=1 sync=false"
)


class StopRequested(Exception):
    """Main-thread control flow for a stop during model initialization."""


def utc():
    return datetime.now(timezone.utc).isoformat()


class LatestFrame:
    def __init__(self):
        self.condition = threading.Condition()
        self.frame = None
        self.sequence = 0

    def publish(self, frame, captured, timestamp):
        with self.condition:
            self.sequence += 1
            self.frame = (self.sequence, captured, timestamp, frame)
            self.condition.notify()

    def take(self, previous, stop):
        with self.condition:
            self.condition.wait_for(
                lambda: stop.is_set() or self.sequence > previous, timeout=0.2
            )
            if stop.is_set() or self.sequence <= previous:
                return None
            return self.frame

    def count(self):
        with self.condition:
            return self.sequence


def worker(run, slot, stop, ready, results):
    torch = None
    try:
        import cv2
        import torch
        import transformers.modeling_utils as modeling_utils
        from PIL import Image
        from transformers import AutoModelForCausalLM

        modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        free_before, _ = torch.cuda.mem_get_info()
        started = time.perf_counter()
        print("Loading pinned official 4-bit model...", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, revision=REVISION, trust_remote_code=True,
            device_map={"": "cuda"},
        ).eval()
        torch.cuda.synchronize()
        results.update({
            "load_seconds": time.perf_counter() - started,
            "model_footprint_gib": model.get_memory_footprint() / 1024**3,
            "cuda_free_before_load_gib": free_before / 1024**3,
            "cuda_free_after_load_gib": torch.cuda.mem_get_info()[0] / 1024**3,
            "cuda_allocated_after_load_gib": torch.cuda.memory_allocated() / 1024**3,
            "versions": {name: importlib.metadata.version(name)
                         for name in ("torch", "transformers", "torchao", "Pillow")},
            "status": "model_ready",
        })
        save_json(run / "analysis.json", results)
        ready.set()
        previous = 0
        while not stop.is_set():
            sample = slot.take(previous, stop)
            if sample is None:
                continue
            sequence, captured, timestamp, frame = sample
            previous_sequence = previous
            previous = sequence
            row = {
                "sequence": sequence, "captured_utc": timestamp,
                "model": MODEL_ID, "revision": REVISION,
                "prompt": PROMPT, "settings": dict(SETTINGS),
                "capture_monotonic_seconds": captured,
                "skipped_sequences_since_previous": sequence - previous_sequence - 1,
                "path": f"frames/frame_{sequence:06d}.png",
                "status": "started",
            }
            results["frames"].append(row)
            if not cv2.imwrite(str(run / row["path"]), frame):
                raise RuntimeError(f"Could not save {row['path']}")
            # Persist selected frame and metadata even if query fails.
            save_json(run / "analysis.json", results)
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            torch.cuda.synchronize()
            started = time.perf_counter()
            row.update({
                "inference_started_utc": utc(),
                "inference_start_monotonic_seconds": started,
                "frame_age_at_inference_start_seconds": started - captured,
                "capture_sequence_at_inference_start": slot.count(),
            })
            # Keep disk I/O outside the synchronized inference interval.
            with torch.inference_mode():
                answer = model.query(image, PROMPT, settings=SETTINGS)["answer"].strip()
            torch.cuda.synchronize()
            completed = time.perf_counter()
            row.update({
                "status": "complete", "answer": answer,
                "completed_utc": utc(),
                "inference_seconds": completed - started,
                "capture_to_completed_answer_seconds": completed - captured,
                "capture_sequence_at_completion": slot.count(),
                "cuda_allocated_gib": torch.cuda.memory_allocated() / 1024**3,
                "cuda_peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
            })
            save_json(run / "analysis.json", results)
            save_json(run / f"frames/frame_{sequence:06d}.json", row)
            print(json.dumps(row), flush=True)
    except BaseException:
        results["worker_error"] = traceback.format_exc()
        results["status"] = "failed"
        if results["frames"] and results["frames"][-1]["status"] == "started":
            results["frames"][-1]["status"] = "failed"
        print(results["worker_error"], flush=True)
        stop.set()
    finally:
        if torch is not None:
            try:
                results["cuda_allocated_gib"] = torch.cuda.memory_allocated() / 1024**3
                results["cuda_peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 1024**3
            except Exception:
                results["cuda_metrics_error"] = traceback.format_exc()
        try:
            save_json(run / "analysis.json", results)
        except Exception:
            results["worker_persistence_error"] = traceback.format_exc()
            print(results["worker_persistence_error"], flush=True)
            stop.set()
        finally:
            ready.set()


def summarize(run, results):
    rows = [row for row in results["frames"] if row["status"] == "complete"]
    results["completed_frames"] = len(rows)
    results["summary"] = {}
    for key in ("inference_seconds", "frame_age_at_inference_start_seconds",
                "capture_to_completed_answer_seconds"):
        values = [row[key] for row in rows]
        results["summary"][key] = {
            "mean": statistics.mean(values) if values else None,
            "min": min(values) if values else None,
            "max": max(values) if values else None,
        }
    count, first, last, max_gap, total_read = 0, None, None, 0.0, 0.0
    capture_csv = run / "capture_timestamps.csv"
    if capture_csv.exists():
        with capture_csv.open() as stream:
            for row in csv.DictReader(stream):
                now = float(row["monotonic_seconds"])
                if last is not None:
                    max_gap = max(max_gap, now - last)
                if first is None:
                    first = now
                last = now
                count += 1
                total_read += float(row["read_seconds"])
    span = last - first if count > 1 else 0.0
    results["camera_capture"] = {
        "delivered_frames": count,
        "first_to_last_seconds": span,
        "measured_fps": (count - 1) / span if span > 0 else None,
        "fps_definition": "(delivered frames - 1) / (last - first delivery timestamp); excludes alignment preview",
        "max_delivery_gap_seconds": max_gap if count > 1 else None,
        "mean_read_seconds": total_read / count if count else None,
        "requested_fps": 30,
    }
    try:
        text = (run / "tegrastats.txt").read_text(errors="replace")
    except OSError as error:
        text = ""
        results["telemetry_read_error"] = repr(error)
    samples = [(int(a), int(b)) for a, b in re.findall(r"RAM (\d+)/(\d+)MB", text)]
    results["system_ram"] = {
        "source": "tegrastats, 500 ms sampling; whole system, including other processes",
        "samples": len(samples),
        "peak_used_mb": max((a for a, _ in samples), default=None),
        "total_mb": samples[-1][1] if samples else None,
    }
    if not samples:
        results["telemetry_error"] = "No tegrastats RAM samples; inspect tegrastats.txt"
    save_json(run / "analysis.json", results)


def align_preview(camera, stop):
    """Preview on main without publishing frames to inference or the timed CSV."""
    import cv2
    print("Align camera; pause source video at its beginning. Press S, then start playback. Q quits.", flush=True)
    while not stop.is_set():
        ok, frame = camera.read()
        if not ok:
            raise RuntimeError("CSI camera stopped during alignment")
        cv2.putText(frame, "S: start test, then start video | Q: quit",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow("Moondream concurrent CSI", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            return
        if key == ord("q"):
            raise StopRequested
    raise StopRequested


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=60,
                        help="Capture seconds from first delivered frame, excluding load; drain current query")
    parser.add_argument("--wait-for-start", action="store_true",
                        help="Align preview, press S, then start source-video playback")
    parser.add_argument("--camera-only", action="store_true",
                        help="Camera/preview/MJPEG recording check; does not load or import the model")
    parser.add_argument("--record", action="store_true",
                        help="Record MJPEG AVI during live inference as well as camera-only mode")
    parser.add_argument("--source-label", default="unspecified",
                        help="Source video label/identifier preserved in metadata")
    args = parser.parse_args()
    if not math.isfinite(args.duration) or args.duration <= 0:
        parser.error("--duration must be finite and positive")
    run = args.run.resolve()
    run.mkdir(parents=True, exist_ok=False)
    (run / "frames").mkdir()
    results = {
        "created_utc": utc(), "model": MODEL_ID, "revision": REVISION,
        "mode": "camera_only" if args.camera_only else "live_inference",
        "initialization_order": "camera_verified_before_model",
        "wait_for_start": args.wait_for_start, "source_label": args.source_label,
        "prompt": PROMPT, "settings": SETTINGS, "pipeline": PIPELINE,
        "requested_duration_seconds": args.duration, "status": "initializing",
        "design": "Periodic live image analysis, not verified temporal event detection",
        "timestamp_definition": "Host monotonic and UTC immediately after camera.read; not sensor exposure time",
        "timing_definition": "Inference includes query image encoding and decoding; frame age includes frame saving/preparation; completion excludes result disk write",
        "frames": [],
    }
    save_json(run / "analysis.json", results)
    main_results = {}
    stop, ready = threading.Event(), threading.Event()
    slot = LatestFrame()
    def request_stop(signum, unused):
        main_results["stop_reason"] = signal.Signals(signum).name
        stop.set()
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    camera = None
    thread = None
    telemetry = None
    capture_start = None
    writer = None
    recording_count, write_total, write_max = 0, 0.0, 0.0
    try:
        import cv2
        with (run / "tegrastats.txt").open("w") as telemetry_log:
            telemetry = subprocess.Popen(
                ["tegrastats", "--interval", "500"],
                stdout=telemetry_log, stderr=subprocess.STDOUT,
            )
        # Reserve Argus/CSI buffers before allocating model/CUDA storage.
        # isOpened alone is insufficient: verify actual frame delivery first.
        print("Opening CSI camera before model loading...", flush=True)
        camera = cv2.VideoCapture(PIPELINE, cv2.CAP_GSTREAMER)
        if not camera.isOpened():
            raise RuntimeError("CSI camera could not open")
        ok, initial_frame = camera.read()
        if not ok:
            raise RuntimeError("CSI camera produced no frame before model loading")
        main_results["camera_verified_utc"] = utc()
        del initial_frame
        if args.camera_only:
            ready.set()
        else:
            thread = threading.Thread(target=worker,
                                      args=(run, slot, stop, ready, results),
                                      name="single-inference-worker")
            thread.start()
            # Drain/preview camera while loading. Do not publish alignment frames
            # to inference or start the benchmark clock before model readiness.
            while not ready.wait(0.001):
                if stop.is_set():
                    break
                ok, loading_frame = camera.read()
                if not ok:
                    raise RuntimeError("CSI camera stopped during model loading")
                main_results["frames_delivered_during_model_load"] = (
                    main_results.get("frames_delivered_during_model_load", 0) + 1)
                cv2.putText(loading_frame, "Loading model; keep source video paused | Q: stop",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.imshow("Moondream concurrent CSI", loading_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    main_results["stop_reason"] = "loading_preview_q"
                    stop.set()
                    break
        if stop.is_set():
            raise StopRequested
        if args.wait_for_start:
            try:
                align_preview(camera, stop)
            except StopRequested:
                main_results.setdefault("stop_reason", "alignment_stopped")
                raise
        print("Starting timed capture on next delivered frame.", flush=True)
        last_capture = None
        max_gap = 0.0
        main_results["status"] = "capturing"
        with (run / "capture_timestamps.csv").open("w", buffering=1) as captures:
            captures.write("sequence,captured_utc,monotonic_seconds,read_seconds\n")
            while not stop.is_set():
                read_start = time.perf_counter()
                if capture_start is not None and read_start - capture_start >= args.duration:
                    main_results["stop_reason"] = "duration"
                    break
                ok, frame = camera.read()
                now, timestamp = time.perf_counter(), utc()
                if not ok:
                    raise RuntimeError("CSI camera read failed")
                if capture_start is None:
                    capture_start = now
                    main_results["capture_started_utc"] = timestamp
                if last_capture is not None:
                    max_gap = max(max_gap, now - last_capture)
                last_capture = now
                if (args.camera_only or args.record) and writer is None:
                    height, width = frame.shape[:2]
                    writer = cv2.VideoWriter(str(run / "camera.avi"),
                                             cv2.VideoWriter_fourcc(*"MJPG"),
                                             30.0, (width, height))
                    if not writer.isOpened():
                        raise RuntimeError("MJPEG recording writer could not open")
                    main_results["recording"] = "camera.avi"
                    if not cv2.imwrite(str(run / "setup_first_frame.png"), frame):
                        raise RuntimeError("Could not save camera setup still")
                if writer is not None:
                    write_started = time.perf_counter()
                    writer.write(frame)
                    write_seconds = time.perf_counter() - write_started
                    write_total += write_seconds
                    write_max = max(write_max, write_seconds)
                    recording_count += 1
                slot.publish(frame, now, timestamp)
                captures.write(f"{slot.count()},{timestamp},{now:.9f},{now-read_start:.9f}\n")
                preview = frame.copy()
                cv2.putText(preview, f"Live {now-capture_start:.1f}s | Q: stop",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                latest_answer = results.get("latest_answer")
                if latest_answer:
                    cv2.putText(preview, latest_answer[:85], (10, preview.shape[0]-15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                cv2.imshow("Moondream concurrent CSI", preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    main_results["stop_reason"] = "preview_q"
                    break
        main_results["capture_max_gap_seconds"] = max_gap
    except StopRequested:
        pass
    except BaseException:
        main_results["main_error"] = traceback.format_exc()
        print(main_results["main_error"], flush=True)
    finally:
        stop.set()
        with slot.condition:
            slot.condition.notify_all()
        main_results["capture_wall_seconds"] = (
            time.perf_counter() - capture_start if capture_start is not None else 0
        )
        main_results["recording_statistics"] = {
            "submitted_frames": recording_count, "nominal_fps": 30 if args.camera_only or args.record else None,
            "mean_write_seconds": write_total / recording_count if recording_count else None,
            "max_write_seconds": write_max if recording_count else None,
        }
        # Windows and capture are always released by main, before draining worker.
        for cleanup in ([writer.release] if writer is not None else []) + ([camera.release] if camera is not None else []) + (
                [cv2.destroyAllWindows] if 'cv2' in locals() else []):
            try:
                cleanup()
            except Exception:
                main_results.setdefault("cleanup_errors", []).append(traceback.format_exc())
                print(main_results["cleanup_errors"][-1], flush=True)
        main_results["captured_frames"] = slot.count()
        if thread is not None:
            while thread.is_alive():
                thread.join(timeout=0.2)
        results.update(main_results)
        if telemetry is not None:
            telemetry.terminate()
            try:
                telemetry.wait(timeout=5)
            except subprocess.TimeoutExpired:
                telemetry.kill()
                telemetry.wait()
        results["status"] = ("failed" if "worker_error" in results or "main_error" in results
                             or "worker_persistence_error" in results or "cleanup_errors" in results
                             else "finished" if results.get("stop_reason") == "duration" else "stopped")
        results["finished_utc"] = utc()
        summarize(run, results)
        print(json.dumps({key: results[key] for key in
                          ("status", "camera_capture", "recording_statistics", "system_ram")}), flush=True)
        print(f"Results preserved: {run / 'analysis.json'}", flush=True)
    return 1 if results["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
