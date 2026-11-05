# October 22 short-output benchmark: run before 12:25

Implementation prepared; no new hardware run or measured improvement yet.
The user will restart and run manually. Close Codex normally and leave it closed
until completion. Use a standalone system terminal and the same five source
videos as the October 5 benchmark. No dependency changes or sudo.

```bash
cd /home/charlotte/research
bash experiments/run_camera_fast_benchmark.sh 12:23
```

The argument is today's local cutoff; default 12:23 reserves two minutes before
the user's 12:25 stop. If that time has passed the launcher refuses to start.
Capture has a maximum 10-minute budget; analysis a maximum 15-minute budget;
each is further limited by time remaining before the cutoff. External timeout
sends TERM and KILL five seconds later if needed. Native uninterruptible driver
calls can still delay exit; this is not a hard kernel-level guarantee.

## Workflow

1. Launcher saves a unique run name and startup RAM/process snapshot.
2. Capture uses **unchanged v2 capture code and CSI pipeline**, asks for title,
   source identifier (Enter skips), and duration for each of five videos.
3. Use Video 1 Red Machine=60 s, Video 2=35 s, Video 3=19 s, Video 4=25 s,
   Video 5=11 s, if these are the same source clips used previously.
4. Before each clip, pause at the start and align preview. Press S in preview,
   then immediately start playback. Six samples occur at interval centers;
   recording is retained as MJPEG AVI plus actual timestamp CSV.
5. On completion, the camera subprocess and preview exit. Model loading and
   analysis then start automatically with no camera allocation retained by
   that Python process. Q during capture preserves partial files and stops the
   launcher; it does not automatically analyze an incomplete capture.
6. One model processes all 30 images, each under three profiles, 90 serial
   queries. Profile order rotates/reverses to balance order within six frames.
   Every profile independently re-encodes each image: no hidden cross-query
   encoding reuse or simultaneous model queries.
7. At completion, terminal prints profile means and paired comparisons.
   Share the run path/output and reopen Codex after the model process exits.

## What is being tested

| Profile | Task | Token cap |
|---|---|---:|
| short32 | Exact October 21 short construction-hazard prompt | 32 |
| compact16 | Clearest hazard in at most six words, or “none clear” | 16 |
| flag4 | Only YES, NO or UNCLEAR for a clearly visible hazard | 4 |

All use temperature=0. The exact prompts/settings are in `PROFILES` in
`experiments/moondream_4bit_camera_fast.py`, and saved top-level plus per result.
Model/revision, CUDA mapping, allocator warmup workaround, camera resolution,
and dependencies are unchanged. Existing model KV caching remains in use;
this experiment does not newly enable caching.

The 4-token arm changes task complexity: a flag is not a hazard explanation and
can be less useful or wrong. The cap can truncate responses; saved approximate
retokenized answer lengths and possible-cap flags help review, but do not prove
EOS/truncation or exact generated-token counts. No accuracy claims without review.

Measured encoding includes image preparation inside `encode_image`, image-prefix
prefill and first-use cache setup. Query measures prompt prefill/text generation
from that encoded image. Their sum is standalone image inference latency, with
synchronization; PIL file loading and JSON writes are outside it. First-use costs
are retained, with separate means excluding the first image. Incomplete pairs
are excluded from paired speed comparisons. No warmup query is silently removed.

The pinned crop code already picks one local tile plus global tile for 640x360
inputs: usable local crop is 266 px and height minus margins is 248 px. Reducing
input resolution therefore does not automatically reduce encoder crop count.
No speculative resizing/model-compile/runtime changes are included in this
short session. A shorter output may improve latency; no magnitude is promised.

## Saved evidence and limitations

Run directory: `results/camera_fast_<UTC timestamp>/`.

- `capture.json`, five `video_XX/` folders: recordings, six JPEGs each, timestamps.
- `capture_tegrastats.txt`: capture phase whole-system RAM/processor telemetry.
- `analysis_fast.json`: all answers, exact prompt/settings/model revision,
  load/footprint, encode/query/total times, CUDA current/peak, sampled system RAM,
  per-video/per-profile means, matched-pair comparisons, capture FPS/gaps.
- Per-image per-profile answer JSON sidecars.
- `analysis_fast_tegrastats.txt`: model-load/inference phase telemetry.
- Sibling `_startup.txt`, `_capture_terminal.log`, `_analysis_terminal.log`, and
  phase exit statuses. A killed run can lack a final summary; prior JSON updates
  preserve completed queries.

Capture-to-answer and frame-age values include offline recording/waiting and
use UTC clock differences; they are not live response times. Clock adjustments
can affect them. Camera timestamps are host-based, not sensor exposure times.
The three profiles may differ in quality/detail; speed alone does not establish
safety utility. This sequential benchmark **does not demonstrate real-time or
concurrent hazard detection**. New clips introduce differences from old images;
paired arms within today's run provide a stronger comparison.

Do not log claimed efficiency gains yet. After the user confirms completion,
inspect raw evidence, compare timing/outputs/possible truncation, and update the
research log with observed results and limitations.
