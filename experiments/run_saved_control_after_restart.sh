#!/usr/bin/env bash
# Run manually after restart. No camera access; no dependency changes.
set -u -o pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

if pgrep -u "$(id -u)" -af '[p]ython.*moondream'; then
  echo 'An existing Moondream Python process was found. Stop it before this control.' >&2
  exit 1
fi

run="results/postrestart_control_$(date -u +%Y%m%dT%H%M%S%N)"
printf 'Control outputs: %s\n' "$run"
{
  date -Is
  uptime
  free -m
  head -25 /proc/meminfo
  ps -eo pid,ppid,comm,rss --sort=-rss | head -25
  printf 'PYTORCH_CUDA_ALLOC_CONF=%s\n' "${PYTORCH_CUDA_ALLOC_CONF-<unset>}"
  printf 'PYTORCH_ALLOC_CONF=%s\n' "${PYTORCH_ALLOC_CONF-<unset>}"
  printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES-<unset>}"
} > "${run}_startup.txt"
cat "${run}_startup.txt"

timeout --signal=TERM --kill-after=5s 180s \
  python3 -u experiments/moondream_4bit_saved_control.py \
  --run "$run" 2>&1 | tee "${run}_terminal.log"
benchmark_status=${PIPESTATUS[0]}
printf '%s\n' "$benchmark_status" > "${run}_exit_status.txt"
printf '\nBenchmark exit status: %s\nOutputs: %s\n' "$benchmark_status" "$run"
if [ "$benchmark_status" -eq 124 ] || [ "$benchmark_status" -eq 137 ]; then
  echo 'Deadline reached; preserve partial files. Final summaries may be absent.'
fi
exit "$benchmark_status"
