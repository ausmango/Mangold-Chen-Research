#!/usr/bin/env bash
# Five live gated trials. Does not promise six Inspector samples per video.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
resume=''
start_video=1
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --resume) resume="$2"; shift 2;;
    --from-video) start_video="$2"; shift 2;;
    *) echo "Unknown argument: $1" >&2; exit 2;;
  esac
done
if [[ ! "$start_video" =~ ^[1-5]$ ]]; then
  echo '--from-video must be 1 through 5' >&2; exit 2
fi
if [[ -n "$resume" ]]; then
  batch="$resume"
  [[ -f "$batch/plan.json" ]] || { echo 'Resume folder has no plan.json' >&2; exit 2; }
else
  batch="results/scout_five_$(date -u +%Y%m%dT%H%M%S%N)"
  mkdir "$batch"
fi
if [[ ! -f "$batch/plan.json" ]]; then
python3 - "$batch" <<'PY'
import json,sys
from pathlib import Path
source=Path('results/camera_v2-2~2026-10-05/capture.json')
old=json.loads(source.read_text())
plan={'baseline_manifest':str(source),'design':'Live person-gated CSI trials; variable completed samples, not fixed six per video',
      'videos':[{'number':v['number'],'label':v['label'],'source':v['source'],
                 'duration_seconds':v['duration_requested_seconds']} for v in old['videos']]}
Path(sys.argv[1],'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
PY
fi
printf 'Five-video evidence folder: %s\n' "$batch"
echo 'Use the same sources/order as October 5. Keep playback paused during alignment.'
echo 'Press S in each preview, then start that video. Q ends a clip early.'
for ((index=start_video; index<=5; index++)); do
  case "$index" in
    1) duration=60; label='Video 1 - Red Machine';;
    2) duration=35; label='Video 2';;
    3) duration=19; label='Video 3';;
    4) duration=25; label='Video 4';;
    5) duration=11; label='Video 5';;
  esac
  printf '\nPrepare %s (%s seconds). Press Enter to load models/open alignment preview.\n' "$label" "$duration"
  read -r
  clip="$batch/video_$(printf '%02d' "$index")"
  mkdir -p "$clip"
  attempt_log="$clip/terminal_$(date -u +%Y%m%dT%H%M%S%N).log"
  if SCOUT_SESSION_BASE="$clip" bash experiments/run_scout_inspector.sh \
      --camera csi --duration "$duration" --preview --wait-for-start --source-label "$label" \
      2>&1 | tee "$attempt_log"; then
    printf 'Clip %s launcher exit 0.\n' "$index"
  else
    printf 'Clip %s failed; partial results preserved in %s. Stopping batch.\n' "$index" "$batch" >&2
    exit 1
  fi
  python3 - "$batch" <<'PY'
import datetime,json,sys
from pathlib import Path
root=Path(sys.argv[1]);completed=[]
for path in sorted(root.glob('video_*/scout_live_*/client/run.json')):
 data=json.loads(path.read_text())
 completed.append({'path':str(path),'source_label':data['metadata'].get('source_label'),
                   'status':data['metadata'].get('status'),'answers':data['counters'].get('completed_answers',0)})
name='progress_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json'
(root/name).write_text(json.dumps(completed,indent=2)+'\n')
if completed and completed[-1]['status']!='finished':
 raise SystemExit('Last clip ended early; batch stopped. Partial evidence preserved.')
PY
done
printf '\nAll five timed live clips completed. Evidence: %s\n' "$batch"
echo 'Inspector answer counts may vary by scene; do not describe this as a fixed 30-image test.'
