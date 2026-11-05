#!/usr/bin/env bash
# Run manually with Codex closed. Five familiar clips; serial prompt pairs on newest frame.
set -u -o pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
deadline_epoch=$(date -d "today ${1:-12:23}" +%s) || exit 2
if [ "$deadline_epoch" -le "$(date +%s)" ]; then echo 'Deadline passed; nothing launched.'; exit 2; fi
if pgrep -u "$(id -u)" -af '[p]ython.*moondream'; then echo 'Existing model process; stop before running.'; exit 1; fi
mode="${2:-paired}"
profile_args=()
if [ "$mode" = candidate-only ]; then
 profile_args=(--candidate-only)
elif [ "$mode" != paired ]; then
 echo 'Mode must be paired or candidate-only.'; exit 2
fi
session="results/live_compare_$(date -u +%Y%m%dT%H%M%S%N)"
mkdir "$session" || exit 1
{
 date -Is
 uptime
 free -m
 ps -eo pid,ppid,comm,rss --sort=-rss | head -25
} > "$session/startup.txt"
echo "Session: $session"
echo 'Keep source paused until model is ready. S starts each clip; Q stops early.'
durations=(60 35 19 25 11)
for i in 0 1 2 3 4; do
 number=$((i+1))
 echo "Prepare Video $number (${durations[$i]} seconds). Press Enter to open camera/load model; leave video paused."
 read -r || exit 1
 budget=$((deadline_epoch - $(date +%s)))
 if [ "$budget" -le 0 ]; then echo 'Deadline reached.'; exit 124; fi
 # Bound blocked model/camera calls, even before timed capture starts.
 max_budget=$((durations[$i]+150))
 if [ "$budget" -gt "$max_budget" ]; then budget=$max_budget; fi
 run="$session/video_$(printf '%02d' "$number")"
 timeout --signal=TERM --kill-after=5s "$budget" \
  python3 -u experiments/moondream_4bit_live_compare.py --run "$run" \
  --duration "${durations[$i]}" --wait-for-start --record "${profile_args[@]}" \
  --source-label "Construction Video $number (live prompt comparison)" \
  2>&1 | tee "$session/video_${number}_terminal.log"
 status=${PIPESTATUS[0]}
 printf '%s\n' "$status" > "$session/video_${number}_exit_status.txt"
 if [ "$status" -ne 0 ]; then echo "Run stopped/failed (status $status). Evidence: $session. No automatic retry."; exit "$status"; fi
 echo 'Clip finished. Stop its playback. Results preserved.'
done
echo "All clips attempted: $session"
