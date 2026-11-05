#!/usr/bin/env bash
# Analyze preserved CSI captures, without opening a camera or overwriting attempts.
set -u -o pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
source_run="${1:-results/camera_fast_20261022T161952378124366}"
if pgrep -u "$(id -u)" -af '[p]ython.*moondream'; then
 echo 'Another Moondream process remains; stop it before starting this analysis.' >&2
 exit 1
fi
run="results/camera_analysis_$(date -u +%Y%m%dT%H%M%S%N)"
python3 - "$source_run" "$run" <<'PY'
import json, shutil, sys
from pathlib import Path
source, target = map(lambda p: Path(p).resolve(), sys.argv[1:])
manifest = json.loads((source/'capture.json').read_text())
assert len(manifest['videos']) == 5, 'Expected five captured videos'
assert all(len(v['frames']) == 6 for v in manifest['videos']), 'Expected six samples per video'
for video in manifest['videos']:
 for frame in video['frames']:
  if not (source/frame['path']).is_file():
   raise FileNotFoundError(source/frame['path'])
target.mkdir(parents=True, exist_ok=False)
manifest['source_capture_run'] = str(source)
for video in manifest['videos']:
 if 'recording' in video:
  video['original_recording'] = str(source/video.pop('recording'))
 for frame in video['frames']:
  destination = target/frame['path']
  destination.parent.mkdir(parents=True, exist_ok=True)
  shutil.copy2(source/frame['path'], destination)
  frame['source_path'] = str(source/frame['path'])
 folder = Path(video['frames'][0]['path']).parent
 timestamps = source/folder/'recording_timestamps.csv'
 if timestamps.exists():
  shutil.copy2(timestamps, target/folder/'recording_timestamps.csv')
(target/'capture.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print(f'Prepared all 30 saved images in {target}. Original recordings/results untouched.')
PY
prepare_status=$?
if [ "$prepare_status" -ne 0 ]; then exit "$prepare_status"; fi
{
 date -Is
 free -m
 ps -eo pid,ppid,comm,rss --sort=-rss | head -25
} > "$run/startup.txt"
echo 'Loading model automatically; analyzing compact16 on all 30 saved images. No camera or playback needed.'
timeout --signal=TERM --kill-after=5s 900s \
 python3 -u experiments/moondream_4bit_camera_fast.py analyze --run "$run" --profiles compact16 \
 2>&1 | tee "$run/terminal.log"
status=${PIPESTATUS[0]}
printf '%s\n' "$status" > "$run/exit_status.txt"
printf 'Analysis exit: %s\nResults: %s\n' "$status" "$run"
exit "$status"
