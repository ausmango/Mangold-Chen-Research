# Five-video live camera demonstration · November 24, 2026

All five recorded trials completed. The application view shows the live camera,
YOLO boxes and confidence, the active SmolVLM request, and finished captions.
Camera delivery continued during **111 of 111 completed requests**. The demo
establishes concurrent capture and image description. Fall and rollover
detection remain unvalidated, and several captions visibly misread the footage.

## Run results

| Clip | Camera FPS | Answers | Mean inference (s) | Mean answer delay (s) | Peak system RAM (MB) | Source crop | Application recording |
|---|---:|---:|---:|---:|---:|---|---|
| [1](../../results/scout_five_20261124T160831624025736/video_01/scout_live_20261124T160844327971045/summary_20261124T161025205187Z.json) | 16.72 | 44 | 1.354 | 1.692 | 4,413 | Yes | [Recording](../../results/scout_five_20261124T160831624025736/playback/video_01.mp4) |
| [2](../../results/scout_five_20261124T160831624025736/video_02/scout_live_20261124T161044493091622/summary_20261124T161146135508Z.json) | 15.53 | 25 | 1.397 | 1.623 | 4,343 | Yes | [Recording](../../results/scout_five_20261124T160831624025736/playback/video_02.mp4) |
| [3](../../results/scout_five_20261124T160831624025736/video_03/scout_live_20261124T161153766252165/summary_20261124T161235934726Z.json) | 15.67 | 14 | 1.343 | 1.609 | 4,257 | Yes | [Recording](../../results/scout_five_20261124T160831624025736/playback/video_03.mp4) |
| [4](../../results/scout_five_20261124T160831624025736/video_04/scout_live_20261124T161304754487387/summary_20261124T161344890299Z.json) | 16.19 | 19 | 1.323 | 1.553 | 4,344 | Full frame | [Recording](../../results/scout_five_20261124T160831624025736/playback/video_04.mp4) |
| [5](../../results/scout_five_20261124T160831624025736/video_05/scout_live_20261124T161353429053326/summary_20261124T161417738284Z.json) | 15.68 | 9 | 1.262 | 1.479 | 4,286 | Full frame | [Recording](../../results/scout_five_20261124T160831624025736/playback/video_05.mp4) |

The five recordings play for approximately **2 min 30 s** in total. The runs
delivered **2,427 camera frames**. Across all answers, mean inference was **1.349 s**
(range 0.831–1.707 s), mean frame age at inference start was **0.276 s**,
and mean capture-to-answer delay was **1.625 s** (range 1.093–2.103 s).
The longest gap between successive answers within a clip was **1.65 s**.
Zero answers exceeded the configured 3 s threshold at completion, and
**39/111** hit the output token limit. A caption can still become stale afterward
as its source frame ages. Peak whole-system RAM was **4,413 MB**, sampled
by tegrastats. Recording was active and contributes to system load.
Video playback is encoded at 10 FPS. Measured camera delivery is in the table.

Frame age and answer delay use host delivery timestamps. They exclude unknown
camera buffering. The runs record the camera observing a screen, so playback
alignment, glare, and framing can affect what each model sees. Clips 4 and 5
used the full frame. This set has no labelled event boundaries or scored hazard
accuracy.

The recording links point to H.264 MP4 playback copies. The original MJPEG
AVI recordings and their frame timestamps are preserved in each run.

## What to show

Times below are playback times in the saved application recording. They come
from the recorded answer sequence in `demo_frames.jsonl`; the source image
was captured earlier.

| Recording | Playback time | What to look for |
|---|---:|---|
| [Clip 1](../../results/scout_five_20261124T160831624025736/playback/video_01.mp4) | 00:17.2–00:18.3 | The response identifies an orange and black excavator. Watch the camera count advance while a subsequent request generates. |
| [Clip 1](../../results/scout_five_20261124T160831624025736/playback/video_01.mp4) | 00:50.9–00:52.4 | The response identifies an excavator but misses its pronounced tilt and adds a guessed brand. [Submitted frame 1662](../../results/scout_five_20261124T160831624025736/video_01/scout_live_20261124T160844327971045/client/frames/frame_00001662.jpg) |
| [Clip 3](../../results/scout_five_20261124T160831624025736/playback/video_03.mp4) | 00:13.7–00:14.8 | The caption calls the object a cloud. The submitted image shows people outdoors and prominent screen glare. [Submitted frame 558](../../results/scout_five_20261124T160831624025736/video_03/scout_live_20261124T161153766252165/client/frames/frame_00000558.jpg) |
| [Clip 5](../../results/scout_five_20261124T160831624025736/playback/video_05.mp4) | 00:07.1–00:08.2 | The model calls a compact tracked machine a damaged SUV. [Submitted frame 137](../../results/scout_five_20261124T160831624025736/video_05/scout_live_20261124T161353429053326/client/frames/frame_00000137.jpg) |

The last three examples expose different problems: omitting the posture that
matters, confusing the scene, and giving the wrong equipment class. These
examples support qualitative review. A taller box, a higher confidence, or a
fluent caption does not establish that a hazard occurred.

## How to improve event detection

1. **Improve the source first.** Keep the camera fixed and crop out the bezel
   and room. More importantly, play the source files through an offline input
   for a controlled accuracy test. That removes the screen reflection and
   manual start delay. Keep the CSI recording for the live demonstration.
2. **Label short event windows.** Mark the first visible cue, the onset and
   outcome of each fall, rollover or vehicle incident. Include ordinary
   bending, tilted equipment that stays stable, camera motion and reflections.
   Inspect whether a useful image was submitted during each window. That
   separates missing coverage from a wrong caption on visible evidence.
3. **Add temporal cues before asking for a description.** Keep consecutive
   person positions and pose cues, or equipment orientation and track motion.
   Use a small time-bounded frame buffer for before/during/after evidence while
   retaining one VLM worker and latest-frame dispatch. Stock YOLO boxes provide
   locations but cannot classify all construction machinery or measure orientation
   reliably. Equipment rollovers may need a domain-trained detector or pose cues.
4. **Test the event signal separately from wording.** Compare the current
   single-frame baseline with the temporal approach on the same labelled clips.
   Measure event recall, false alarms per minute, alert delay, camera FPS and RAM.
   A stronger VLM can be tested on the same clear inputs if it still misreads
   visible posture. A prompt change alone cannot supply missing motion evidence.

## Configuration and evidence

- Camera: preserved IMX219 CSI pipeline. Scout: YOLO11n TensorRT engine,
  requested FP16, **9,463,447-byte engine file**, confidence 0.40, NMS IoU 0.70.
  Scout adapter initialization including warmup ranged **5.093–5.951 s**.
- Inspector: `ggml-org/SmolVLM2-500M-Video-Instruct-GGUF`, revision
  `ccd7aae53bcb1997355c2f094959e72b3642ce17`. Text checkpoint
  **303,253,504 bytes**, mixed tensor precision from the requested Q4_K_M
  conversion. F16 projector **199,470,624 bytes**. Loaded footprint and peak
  CUDA allocation are unavailable. The Inspector server's logged load interval
  ranged **0.745–1.070 s**, measured from “loading model” to “model loaded”.
  Client Inspector setup, including connection and artifact checks, ranged
  **1.290–1.379 s**.
- The focused profile offered the latest frame after 500 ms without a YOLO
  offer. One pending slot and one serial VLM request bounded the workload.
  Generation was capped at **32 tokens**, temperature **0**.
- Exact base prompt:

  > If construction equipment is not clearly visible, answer: Equipment unclear. Otherwise name the equipment and describe its visible position in one short sentence: upright, tilted, on its side, or unclear. Use only the equipment in the footage. Ignore reflections. Do not infer movement, a fall, or cause.

  Effective prompts, including Scout hints where present, are saved per request.

Batch evidence is under [the saved batch](../../results/scout_five_20261124T160831624025736). Each clip contains
`client/demo.avi`, `demo_frames.jsonl`, `run.json`, `scout.jsonl`, and
`inspections.jsonl` under `client/`. The per-run summary, tegrastats,
Inspector startup log and exit status are in its enclosing run folder.
