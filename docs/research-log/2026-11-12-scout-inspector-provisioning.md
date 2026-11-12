# November 12, 2026 — provisioning the USB pipeline

**Completed: downloads, isolated Python setup, CUDA runtime build and CPU quantization. Hardware test deferred to user.**
The user authorized model downloads and setup, and explicitly withheld camera
readiness. No camera capture, GPU inference, Scout engine export or new benchmark
has run. Do not treat CPU import/test times as model performance.

Evidence directory: `results/scout_provision_20261112T165840067947Z/`.

| Asset | Exact source | Downloaded bytes |
|---|---|---:|
| YOLO11n `.pt` | `ultralytics/assets` release `v8.3.0` | 5,613,764 |
| SmolVLM2-500M text F16 GGUF | `ggml-org/SmolVLM2-500M-Video-Instruct-GGUF` commit `ccd7aae53bcb1997355c2f094959e72b3642ce17` | 820,424,704 |
| Matching vision projector F16 GGUF | Same repository/commit | 199,470,624 |
| llama.cpp source archive | `2a4daf0e08d6a9b6993ee0d4a9a3cfc35a6f405c` | 38,939,831 |

`download-manifest.json` records exact URLs and SHA256 for each file. The official
GGUF source provides F16/Q8_0; Q4_K_M text quantization will be derived locally,
keeping the projector at F16. These are checkpoint bytes, not loaded footprint.

Created `.venv-scout` using `venv --without-pip --system-site-packages`, because
normal venv creation failed for missing `ensurepip`. Reused the existing pip module
to install new packages **inside** that environment with `--no-deps`:
Ultralytics 8.3.162, ultralytics-thop 2.0.9, seaborn 0.13.2, ONNX 1.17.0,
protobuf 4.25.8 and py-cpuinfo 9.0.0. An initial isolated 8.3.0 installation was
replaced after source inspection found its TensorRT <=10.1 warning; 8.3.162 source
excludes 10.1 specifically and supports the installed 10.3 API. No TensorRT upgrade.

`stack_before.json` and `stack_after.json` match for Torch 2.8.0, Torchvision 0.23.0,
Transformers 4.57.6, TorchAO 0.11.0, NumPy 1.24.4 and huggingface-hub 0.36.2.
Host protobuf remains 3.12.4; the newer protobuf is isolated. `optional_versions.json`
records paths. CPU import checks succeeded for Ultralytics, ONNX, OpenCV 4.5.4 and
TensorRT 10.3.0. Torchvision reports its native operators available. No inference
was used to perform these checks. OpenCV emitted an EGL-display diagnostic during
import; it did not prevent imports, and preview remains untested.

`pip_check_final.txt` reports `opencv-python` missing because the functioning OpenCV
comes from OS packages, not pip metadata. Deliberately did not install a pip OpenCV
wheel over the Jetson package. This is not a claim that pip check passed cleanly.

CMake configured CUDA 12.6 with explicit architecture 87, Release, OpenSSL disabled
for the local-only server, and two build jobs. Logs: `cmake_configure.log`,
`build.log`. No sudo or stack changes. Source was downloaded as a pinned archive;
parent repository Git corruption can make automatic build-banner Git info wrong;
use the archive SHA/`PROVISIONED_REVISION` for actual source provenance.

Added a managed launcher that will export Scout on first hardware launch before
starting Inspector, preserve startup/export/tegrastats logs, then run the bounded
client. It owns/cleans up only its own server and monitoring processes and rejects
an occupied server port. Quickstart: `docs/scout-inspector-quickstart.md`.
The default local smoke trigger is person presence; proximity remains available
in the example. This makes exercising concurrency possible without asserting a
hazard. No full continuous video recording is added.

Added `experiments/summarize_scout_inspector.py` to summarize saved rows, including
actual camera deliveries during inference intervals. This does not establish
simultaneous GPU kernels, verified quality, or real-time guarantees. New CPU tests
(10) and existing tests (13) passed in the isolated environment; see
`cpu_tests_final.txt` and `legacy_tests.txt`. Failure traceback in the contract suite
is intentional. Shell/Python syntax checks passed.

CPU-only GGUF inspection (`gguf_metadata.json`) found native vision image size 512,
patch size 16 and idefics3 projector scale factor 4. A transmitted 384-square image
can be upscaled; reduced encoder work/token counts cannot be inferred solely from
our resize setting. No runtime token count or performance result is invented.

Next: finish runtime build, perform CPU Q4_K_M quantization, record hashes/bytes,
then leave the fully local command for the user. Scout TensorRT export and actual
camera/GPU acceptance test remain for the user's explicit hardware launch.

## Build adjustment

The initial two-job build was intentionally stopped (exit 143) to shorten setup.
`build_restart.json` records the scoped process restart; `cmake_configure_resume.log`
and `build_resume.log` preserve the resumed attempt. Resumed with four compiler
jobs and `GGML_CUDA_FA_QUANTS=f16-f16`, matching explicit F16 K/V cache settings
in the launcher. Text weights will still be Q4_K_M; KV precision is separate.
The ggml source commit is explicitly set in CMake. The common server build banner
can still inherit parent Git information for an archive build, so the separately
recorded archive/source manifest remains authoritative. Compiler-flag changes
can rebuild some previously completed objects. No GPU inference is involved.

## Additional offline validation

CPU dispatcher inspection confirms Torch and Torchvision are built for CUDA 12.6
and the Torchvision NMS CUDA kernel is registered; no CUDA tensor was allocated
and no operator inference was run (`cuda_operator_registration.json`). This is
compatibility evidence, not a successful GPU benchmark.

The expanded CPU suite now has 12 passing tests (10 core contracts and two managed
launcher lifecycle tests), plus 13 passing legacy tests. Lifecycle tests use fake
hardware executables, exercise success and failure exit codes, preserve partial
summaries and check that owned mock server/monitor processes are stopped.
Latest log: `cpu_tests_provision_final.txt`. All these are CPU-only checks.

## Completed provisioning

`build_resume.log` finishes successfully for both `llama-server` and
`llama-quantize`; CMake exit 0. `server_ldd.txt` reports no missing shared libraries.
CPU quantization completed successfully in 13,824.70 ms; this is a file conversion
time, **not model inference latency**. Output: `models/inspector/model-Q4_K_M.gguf`,
303,253,504 bytes; hashes and exact source recorded in the updated download manifest.

**Quantization caveat:** Q4_K_M was requested, but 193/291 tensors needed fallback
because 960-wide dimensions are not divisible by 256. Quantizer reports 5.89 bits
per weight; do not label the resulting checkpoint uniformly INT4. The separate
F16 projector remains 199,470,624 bytes. Combined checkpoint bytes are 502,724,128,
not loaded footprint and already greater than a 400 MB checkpoint budget. Runtime
footprint and latency are unmeasured. Future Q4_0 or projector quantization would
be separately labelled configurations requiring quality/performance validation.

Read-only sysfs inspection found `/dev/video0` is the IMX219 CSI node, no USB
webcam connected (`camera_enumeration.json`). Added selectable CSI support using
the **exact** historical pipeline, checked with AST comparison; USB auto-selection
excludes CSI. New suite now passes 15 CPU tests, including those adapter checks;
13 legacy tests also passed. `cpu_tests_final.txt` records the new suite. Syntax
and shell checks passed. Local HTTP bypasses environment proxies.

`preflight_final.txt` reports only the missing Scout `.engine`, expected because
GPU export was explicitly deferred. The managed launcher automatically performs
that one-time export before starting Inspector, while preserving export logs.
No camera opened, no GPU model inference/export, no actual live benchmark, no
commit or push. `provision-status.json` records completed setup and the next command:

```bash
cd /home/charlotte/research
bash experiments/run_scout_inspector.sh --camera csi --duration 60 --preview
```

For USB instead, connect a webcam and select `--camera usb`. Person presence is
the initial smoke trigger. Results go to unique `results/scout_live_*` folders.
The 60-second timer begins on first delivered frame, after export/server startup;
no S-key alignment wait. The launcher owns and cleans up its server/tegrastats.

## Actual GPU Scout build repair (after user launched the first attempt)

Failed attempt: `results/scout_live_20261112T180401394542702/`.
Successful ONNX export was followed by NvMap error 12 and a Cask GEMM tactic error
inside the PyTorch-resident TensorRT builder. No camera frames or Inspector answers
were collected. Sampled system RAM peak was 5,787 MB. Interrupted exporter PID
20376 remained allocated after Ctrl+C and was stopped specifically; unrelated
processes were not killed. This does not prove memory was the only underlying cause.

Added `experiments/build_scout_from_onnx.py`: standalone trtexec, no Torch import,
128 MiB workspace, builder optimization level 0, avgTiming 1, and skipInference
(no performance phase, although building still executes GPU profiling kernels).
The first sandboxed attempt could not access the device; its failed log remains
`results/scout_standalone_build_20261112T181022810284826`.
Reran with authorized GPU device access. Evidence directory: `results/scout_standalone_device_20261112T181215607080372/`.
Build exit 0, 46.1992 seconds, sampled peak RAM 4,678 MB. Background/workflow differs;
this is not a controlled RAM reduction or inference speedup benchmark.

The new engine is 9,463,447 bytes. Embedded Ultralytics metadata preserves names,
stride, input size and task; manifest records source ONNX hash, engine hash and
build flags. Synthetic Scout GPU inference passed with 80 labels and person present
(`scout_validation.log`); camera and Inspector were not launched. Local preflight
now passes. The 15 CPU checks passed after updating the managed-launcher mock for
the new builder. Existing scripts and installed stack remain unchanged.

The managed launcher now skips building when the engine is present; a future rebuild
uses the standalone helper. This low-effort engine prioritizes a first working test,
not maximum throughput. The user can run the CSI command in the quickstart now.
No commit/push. Live concurrency/answer latency remain unmeasured until that run.
