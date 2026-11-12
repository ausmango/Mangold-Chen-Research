# Running the provisioned two-model camera test

The new pipeline is separate from the existing Moondream scripts. No camera or
GPU inference was launched during provisioning. See the dated provisioning note
for exact installed/downloaded evidence and build status.

Read-only enumeration found the connected camera is the **IMX219 CSI camera**.
To use that existing camera, when positioned, run in a desktop terminal:

```bash
cd /home/charlotte/research
bash experiments/run_scout_inspector.sh --camera csi --duration 60 --preview
```

The runner uses the repository-local `.venv-scout` environment automatically.
On first launch it exports the downloaded YOLO11n weights to a local TensorRT FP16
engine. **This is a GPU operation and may take several minutes.** It happens before
opening the camera and before starting Inspector. Further runs reuse the engine.
The script then starts its own local SmolVLM2 Inspector server, waits for readiness,
and launches a 60-second preview/capture/inference run. It shuts down the server
and monitoring processes it owns on exit.

The local config initially triggers on **person presence** so you can stand in
view to exercise both workers. The example config still uses person/vehicle
proximity. This smoke configuration does not establish a construction hazard.
Later edit `configs/scout_inspector.local.json` to enable proximity or other rules.
Stock COCO cannot establish PPE absence or recognize every construction machine.

The timer starts on the **first delivered camera frame**, after model initialization.
There is no S-key wait. For a video test, prepare/align the familiar source before
launch and start playback when preview appears. Q or Ctrl+C stops early. Full
wall time also includes export, startup and bounded shutdown. No new description
per video frame is promised.

For a USB webcam instead, connect it and use `--camera usb`; device selection
is automatic and explicitly excludes the CSI node. No USB camera was connected
during provisioning. To specify a shorter CSI trial:

```bash
bash experiments/run_scout_inspector.sh --camera csi --duration 20 --preview
```

Save folders are unique: `results/scout_live_*/`. They contain the full-session
tegrastats log (including export/server startup), Inspector server log, export log
on first use, client terminal log, provenance and exit status. `client/` contains
`run.json`, `config.json`, `scout.jsonl`, `inspections.jsonl`, sampled client RAM and
analyzed images/answers under `frames/`. It does not save a continuous video.

Confirm capture keeps updating while Inspector is generating, and that completed
answers appear. Inspect `client/run.json` for errors/capture rate and the completed
rows of `client/inspections.jsonl` for inference time, frame age and answer delay.
These are unverified generated observations, not established accuracy.

No-hardware preflight:

```bash
bash experiments/run_scout_inspector.sh --check
```

Before first engine export this is expected to report the missing `.engine`;
launching the real test performs that export. Missing projector/model/runtime
files are genuine setup failures. A port-8080 conflict stops the runner before it
can accidentally query a different model; do not indiscriminately kill processes.
If startup/export fails, keep the result folder and inspect its logs.

Existing dependencies are reused. `pip check` reports an `opencv-python` metadata
requirement because this machine uses the working OS OpenCV package (4.5.4), not
the pip distribution. Do not install a pip OpenCV wheel over it. All optional new
Python packages are isolated; the underlying Jetson stack was not upgraded.

The downloaded projector metadata reports a native 512×512 image size and 16-pixel
patches (`gguf_metadata.json` in the provisioning evidence). Submitting a
384×384 letterbox bounds the transmitted image but does not prove the encoder
runs at 384 or produces fewer native visual tokens: the backend can resize it.
Server CPU threads and OpenMP are limited to two in this runner to leave CPU
capacity for Scout/preview; this scheduling choice is not a measured speedup.

Provisioned quantization is requested Q4_K_M with automatic higher-precision
fallbacks for 193/291 tensors (reported 5.89 bits/weight), not uniformly INT4.
Text checkpoint: 303,253,504 bytes; F16 projector: 199,470,624 bytes. Loaded RAM
and actual inference performance remain unmeasured. See the provisioning note.
