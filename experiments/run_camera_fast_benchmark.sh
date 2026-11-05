#!/usr/bin/env bash
# Manual launch only: capture 5 videos x 6 frames, release camera, then analyze.
# Deadline is local time; default reserves 2 minutes before the user's 12:25 stop.
set -u -o pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
deadline="${1:-12:23}"
deadline_epoch=$(date -d "today $deadline" +%s) || exit 2
if [ "$deadline_epoch" -le "$(date +%s)" ]; then
  echo "Deadline $deadline has already passed. No benchmark launched." >&2
  exit 2
fi
if pgrep -u "$(id -u)" -af '[p]ython.*moondream'; then
  echo 'Existing Moondream Python process found; stop it before this run.' >&2
  exit 1
fi
run="results/camera_fast_$(date -u +%Y%m%dT%H%M%S%N)"
printf 'Run: %s\nLocal deadline: %s\n' "$run" "$deadline"
{
  date -Is
  uptime
  free -m
  ps -eo pid,ppid,comm,rss --sort=-rss | head -25
} > "${run}_startup.txt"
cat "${run}_startup.txt"

capture_budget=$((deadline_epoch - $(date +%s)))
if [ "$capture_budget" -gt 600 ]; then capture_budget=600; fi
echo 'Five video names, sources and durations are automatic. NO terminal metadata input needed.'
echo 'Reference durations: Video 1=60, Video 2=35, Video 3=19, Video 4=25, Video 5=11 seconds.'
echo 'For each: pause source, align preview, press S, then immediately start playback.'
timeout --foreground --signal=TERM --kill-after=5s "$capture_budget" \
  python3 -u experiments/moondream_4bit_camera_fast.py capture --run "$run" \
  2>&1 | tee "${run}_capture_terminal.log"
capture_status=${PIPESTATUS[0]}
printf '%s\n' "$capture_status" > "${run}_capture_exit_status.txt"
if [ "$capture_status" -ne 0 ]; then
  echo 'Capture did not complete. Partial artifacts preserved; analysis not started.'
  exit "$capture_status"
fi

analysis_budget=$((deadline_epoch - $(date +%s)))
if [ "$analysis_budget" -le 0 ]; then
  echo 'Deadline reached. Captures preserved; no model loaded.'
  exit 124
fi
if [ "$analysis_budget" -gt 900 ]; then analysis_budget=900; fi
echo 'Camera process exited. Deadline mode: compact16 only, 30 serial queries. Reference can be run on these images later.'
timeout --signal=TERM --kill-after=5s "$analysis_budget" \
  python3 -u experiments/moondream_4bit_camera_fast.py analyze --run "$run" --profiles compact16 \
  2>&1 | tee "${run}_analysis_terminal.log"
analysis_status=${PIPESTATUS[0]}
printf '%s\n' "$analysis_status" > "${run}_analysis_exit_status.txt"
printf 'Analysis exit: %s\nResults: %s\n' "$analysis_status" "$run"
exit "$analysis_status"
