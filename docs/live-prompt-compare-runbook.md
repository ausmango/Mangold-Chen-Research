# Live CSI recording and prompt comparison — October 22

This supersedes today's offline fast-benchmark instructions. User clarified
that capture, recording, preview, and latest-frame inference must be concurrent.
No new hardware success or efficiency gain has been measured yet.

After restarting, close Codex fully and use a standalone terminal:

```bash
cd /home/charlotte/research
bash experiments/run_live_prompt_compare.sh 12:23
```

The launcher uses the five familiar construction videos, durations 60, 35, 19,
25, and 11 seconds. For each, prepare/pause the source and press Enter in the
terminal. CSI opens, verifies delivery, and keeps previewing while the pinned
model loads. **Wait for alignment prompt, press S in the preview, then start
playback.** Timed duration starts at the next delivered frame. MJPEG recording,
preview, and inference run together. At duration expiry, camera/recording close
and the current prompt pair drains. Then prepare the next video. Model reloads
once per clip, not per query, so each clip gets its own cold first-use result.

No queue: one selected frame is in flight, one latest-frame slot is replaceable.
After a prompt pair completes the worker selects the newest delivered frame,
not the next old image. Consequently short clips may have few answers; six
completed live analyses are not guaranteed. Six is no longer the sampling quota
for this explicitly latest-frame test. Keep original six-frame sequential
benchmarks as a separate comparison.

## Comparison and cache

- Reference `short32`: exact October 21 short hazard prompt, max_tokens=32.
- Candidate `compact16`: clearest visible construction hazard in at most six
  words, or 'none clear'; max_tokens=16. Exact prompts saved in JSON.
- Temperature=0, same pinned official 4-bit model/revision/loading workaround,
  camera pipeline and dependencies. One model worker, queries always serial.
- Encode selected image **once**, run both prompts against its EncodedImage,
  alternate which prompt runs first, then discard encoding before next frame.
  This avoids duplicate encoding for paired comparisons. Existing model KV and
  per-image prefix caching is being reused, not newly enabled. Do not reuse
  stale image encodings across changing live frames.
- Two prompts deliberately slow the rate of new observations. For the freshest
  candidate-only updates on another explicitly chosen run:

```bash
bash experiments/run_live_prompt_compare.sh 12:23 candidate-only
```

Do not automatically launch both sessions. The paired session establishes output
and query-time differences; candidate-only measures deployment-style update rate.
Encoding once is not an extra speed optimization for single-query mode; the
normal model query already encodes once. No magnitude of improvement promised.

## Evidence and timing

Unique `results/live_compare_<UTC>/video_XX/` folders contain `analysis.json`,
selected PNGs and answer JSON sidecars, `camera.avi`, delivery timestamp CSV,
and `tegrastats.txt`. Parent folder contains startup snapshot, per-clip logs and
exit statuses. Terminal prints answers as they finish; preview overlays latest
completed answer (truncated only for display; full answer is saved).

- `pairs[].image_encode_seconds`: one shared image-encoding duration.
- Per answer `query_seconds`: own synchronized query from encoded image.
- `standalone_equivalent_inference_seconds`: shared encode plus own query,
  **estimate of standalone cost**, excludes other prompt and result writes.
- `inference_seconds`: actual elapsed from shared encode start to this answer;
  second answer includes first query/intervening work. Do not use it as pure
  per-prompt latency in paired comparisons.
- Frame age at encode start and query start are separate fields.
- `capture_to_completed_answer_seconds`: measured actual delivery-to-answer,
  including previous prompt if second. This is the live responsiveness metric.
- `profile_statistics`: separate profile query/equivalent/actual-delay summaries.
- Capture sequence at encode/query start and completion lets reviewers verify
  new frames arrived while inference executed. Capture FPS/gaps, recording write
  times, load time, model footprint, CUDA allocation/peak and tegrastats peak RAM
  remain saved. `completed_frames` counts distinct analyzed camera images;
  `completed_answers` and `completed_pairs` are separate.

Host delivery timestamps are not sensor exposure times. Recording uses nominal
30 FPS; timestamp CSV records actual timings. Prompt shortening changes answer
content/coverage; generated hazard claims are not verified accuracy. This is
periodic live image analysis, not verified temporal event detection.

## Stopping and validation

Q/Ctrl+C requests a stop; an in-flight pair drains. Native driver calls can
block, so the launcher independently sends TERM at min(time remaining before
12:23, clip duration+150 seconds), then KILL after five seconds. If a clip fails,
launcher stops without retries. An uninterruptible kernel call can still delay
exit. No further restarts or repeat attempts are automated.

Syntax, shell syntax, CLI help and all 13 CPU-only tests pass. New tests verify
one encoding per image/two serial prompts, newest-frame replacement after the
pair, actual timing identities, prompt-order reversal and partial error output.
No camera/GPU test was launched by the assistant. The user requested efficiency
findings be added to research logs only after confirming actual results.
