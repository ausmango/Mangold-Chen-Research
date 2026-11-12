# November 12, 2026 — USB Scout & Inspector refactor

## Completed

Implemented a separate USB entry point (`main.py`) and modules under
`src/scout_inspector/`. Legacy Moondream/CSI scripts, dependencies and benchmark
results were preserved. No camera capture, GPU inference/export, model download,
dependency installation, commit or push was performed.

Scout owns USB reads, YOLO TensorRT detection and configurable rules. Inspector
runs independently with one serial local SmolVLM2 request. Their bridge is exactly
`queue.Queue(maxsize=1)`: a newer triggered frame replaces pending work. Main owns
OpenCV windows. Nominal frames are discarded, images letterboxed to 384×384 by
default, inputs older than one second rejected and delayed answers marked stale.
Two Scout hints at most are appended to limit prompt expansion. Full trigger
metadata remains saved. The configured generation limit is 32 tokens at temperature
zero, not a claim that all replies contain 32 measured tokens.

Local llama.cpp INT4 text-model/server adapter and optional local Transformers FP16
adapter are implemented. Exact revision, artifact SHA256/bytes, load scope, prompt,
settings, analyzed PNG, exact HTTP JPEG, answer, timestamps, frame age, inference
latency, host-delivery-to-answer delay and available memory/token metrics are
preserved. Tegrastats runs alongside the client. External-server model load time,
loaded footprint and CUDA allocation are explicitly unknown to the client; obtain
server startup evidence separately. The FP16 adapter records model load, footprint
and PyTorch allocation/peak, excluding untracked TensorRT/external allocations.

Added strict example configuration, explicit local engine export utility, root
README and `docs/scout-inspector-architecture.md` with repository audit, setup,
model provenance, launch/stop commands and constraints. Updated `docs/context.md`
and appended new scope guidance to AGENTS.md, preserving existing instructions.

## Actual CPU verification

Final saved evidence directory:
`results/scout_inspector_final_checks_20261112T163946445455257/`.

| Check | Result | Saved evidence |
|---|---|---|
| New CPU contracts | 10 passed | `new_tests.txt` |
| Existing experiment regression tests | 13 passed | `legacy_tests.txt` |
| Python syntax compilation | Exit 0 | `syntax.txt`, `status.txt` |
| Unconfigured example preflight | Expected exit 2; exact revision required | `example_preflight.txt`, `status.txt` |

New tests cover one-slot replacement/task accounting; capture progressing while a
fake Inspector is blocked; at most one active Inspector; stale input rejection;
exact HTTP payload/settings/JPEG preservation; completed answer preservation during
stop; nominal scene gating; unsupported PPE labels; bounded join with a fake blocked
camera; partial evidence on intentional worker failure; configuration provenance;
and tegrastats peak parsing. Test printed timings come from fake backends and are
**not VLM or camera performance measurements**. Intentional failure traceback in
the passing suite is expected.

Earlier CPU attempts were also retained in uniquely named results folders.
No successfully tested hardware concurrency, accuracy, FPS, smaller runtime RAM,
or speedup is claimed for this implementation.

## Limitations and next acceptance test

- 30 FPS is a requested rate. Scout serial detection, Python scheduling, disk
  metadata writes and shared GPU contention can reduce it. Driver buffering and
  sensor-exposure age remain unmeasured. Capture timing starts after `read()`.
- A bounded queue prevents backlog, not model OOM or old answers. One active
  inference can still finish late. Native CUDA/V4L2 calls can hang; bounded joins
  save partial evidence and identify blocked workers, with emergency client exit.
- The default detector uses COCO vehicle proxies, not trained construction/PPE
  classes. Missing-PPE heuristics are disabled. Generated descriptions are
  unverified observations, not validated hazards, events or OSHA violations.
- Sub-400 MB Inspector runtime is unproven. INT4 text checkpoint size excludes
  higher-precision vision/projector, KV cache, activations and runtime buffers.
  Replacing 2B with nominal 500M means fewer parameters, not established quality.
- Existing Moondream KV caching was already implemented. New scheduling/gating
  does not newly enable it; cross-image cache reuse is not claimed.
- Optional dependencies, engine, pinned model assets and local server are not
  provisioned. Adapters/export recipes remain hardware untested. Setup must
  preserve the installed Jetson stack; no forced dependency upgrades.
- Git status failed because loose object
  `.git/objects/3f/1353488eaae7982c4f428ee7903b6ad5cc3ef2` is empty/corrupt.
  No Git repair or metadata modification attempted.

After supplying validated assets and confirming camera readiness, run the command
below with an already ready Inspector server (server launch/monitoring in setup
guide). This records analyzed images, not a full continuous video:

```bash
cd /home/charlotte/research
python3 main.py --config configs/scout_inspector.local.json --check
python3 -u main.py --config configs/scout_inspector.local.json --duration 60 --preview
```

Timer starts on first delivered USB frame after initialization. Align source
beforehand and start playback when preview appears. No S-key wait exists. Q,
Ctrl+C, SIGTERM or expiry initiates bounded shutdown. Inspect actual evidence for
continued capture during inference, useful outputs, frame/answer age, sampled RAM
peak and allocation errors before claiming progress over legacy performance.

Historical comparison only: October 5 mean 8.3241 s for 30 frames; October 22 mean
4.4612 s for 30 frames (46.41% lower, changed prompt/input conditions). These were
sequential workflows. They are not a new dual-model benchmark. See prior dated
notes and their saved JSON evidence.
