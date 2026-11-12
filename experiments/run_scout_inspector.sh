#!/usr/bin/env bash
# User-invoked hardware launch. Setup/--check never open the camera or run inference.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python_bin="$PWD/.venv-scout/bin/python"
server_bin="$PWD/dependencies/llama.cpp/build/bin/llama-server"
config="$PWD/configs/scout_inspector.local.json"
export YOLO_AUTOINSTALL=False
export YOLO_OFFLINE=True
export YOLO_CONFIG_DIR="$PWD/dependencies/ultralytics-config"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1
if [[ ! -x "$python_bin" || ! -x "$server_bin" ]]; then
  echo 'Isolated Python environment or llama.cpp build is missing.' >&2
  exit 2
fi
# This managed launcher owns one known server configuration; reject mismatches.
"$python_bin" - "$config" <<'PYCONFIG'
import json,sys
from pathlib import Path
cfg=json.loads(Path(sys.argv[1]).read_text())
expected={'backend':'llama_cpp','base_url':'http://127.0.0.1:8080/v1',
          'model_path':'models/inspector/model-Q4_K_M.gguf',
          'mmproj_path':'models/inspector/mmproj.gguf'}
for key,value in expected.items():
    if cfg['inspector'][key] != value:
        raise SystemExit(f'Managed launcher requires inspector.{key}={value}; use the manual setup guide for a custom server.')
for name in ['models/inspector/model-Q4_K_M.gguf','models/inspector/mmproj.gguf']:
    if not Path(name).is_file():
        raise SystemExit('Missing prepared Inspector artifact: '+name)
PYCONFIG
if [[ "${1:-}" == '--check' ]]; then
  exec "$python_bin" main.py --config "$config" --check
fi
session="${SCOUT_SESSION_BASE:-results}/scout_live_$(date -u +%Y%m%dT%H%M%S%N)"
mkdir -p "$session"
command -v tegrastats >/dev/null
free -m > "$session/memory_before.txt"
cp models/provision-manifest.json "$session/provision-manifest.json"
cp "$config" "$session/config.json"
printf 'Session: %s\n' "$session"
echo 'Initial local configuration uses person-presence triggering: stand in view for a smoke test. This is not a hazard detector validation.'
server_pid=''
tegra_pid=''
stop_owned() {
  local pid="$1" label="$2"
  kill -TERM "$pid" 2>/dev/null || true
  for _ in {1..50}; do
    kill -0 "$pid" 2>/dev/null || break
    sleep .1
  done
  if kill -0 "$pid" 2>/dev/null; then
    printf '%s required SIGKILL (PID %s)\n' "$label" "$pid" | tee -a "$session/cleanup.log"
    kill -KILL "$pid" 2>/dev/null || true
    for _ in {1..10}; do
      kill -0 "$pid" 2>/dev/null || break
      sleep .1
    done
  fi
  if kill -0 "$pid" 2>/dev/null; then
    printf '%s remains blocked (PID %s); no unbounded wait attempted\n' "$label" "$pid" | tee -a "$session/cleanup.log"
    return 1
  fi
  wait "$pid" 2>/dev/null || true
}
cleanup() {
  code=$?
  trap - EXIT
  if [[ -n "$server_pid" ]]; then
    stop_owned "$server_pid" Inspector || code=2
  fi
  if [[ -n "$tegra_pid" ]]; then
    stop_owned "$tegra_pid" tegrastats || code=2
  fi
  if [[ -f "$session/client/run.json" ]]; then
    "$python_bin" experiments/summarize_scout_inspector.py "$session" \
      2>&1 | tee "$session/summary_terminal.log" || true
  fi
  printf '%s\n' "$code" > "$session/exit_status.txt"
  printf 'Session preserved: %s (exit %s)\n' "$session" "$code"
  exit "$code"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
# Reuse closed-server sockets in TIME_WAIT; still reject an active listener.
"$python_bin" - <<'PY'
import socket
with socket.socket() as s:
    s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
    try:
        s.bind(('127.0.0.1',8080))
    except OSError as exc:
        raise SystemExit(f'Port 8080 is still occupied by an active listener: {exc}. No new server started.')
PY
tegrastats --interval 500 > "$session/whole_session_tegrastats.txt" 2>&1 &
tegra_pid=$!
printf '%s\n' "$tegra_pid" > "$session/tegrastats.pid"
if [[ ! -f models/scout/yolo11n-fp16.engine ]]; then
  echo 'One-time YOLO TensorRT export (GPU); camera remains closed. This may take several minutes.'
  if [[ -f models/scout/source/yolo11n.onnx ]]; then
    timeout --signal=TERM --kill-after=10s 660s \
      "$python_bin" -u experiments/build_scout_from_onnx.py \
      2>&1 | tee "$session/scout_export.log"
  else
    echo 'Missing local ONNX. Run the documented initial exporter first.' >&2
    exit 2
  fi
fi
"$python_bin" main.py --config "$config" --check
printf 'Starting pinned Inspector. Keep camera/source prepared; capture timer starts after initialization.\n'
"$server_bin" -m models/inspector/model-Q4_K_M.gguf \
  --mmproj models/inspector/mmproj.gguf -ngl 99 -c 1024 --parallel 1 --threads 2 --threads-batch 2 --cache-type-k f16 --cache-type-v f16 \
  --host 127.0.0.1 --port 8080 > "$session/inspector_server.log" 2>&1 &
server_pid=$!
printf '%s\n' "$server_pid" > "$session/inspector.pid"
"$python_bin" - "$server_pid" <<'PY'
import os,sys,time,urllib.request
pid=int(sys.argv[1])
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
for _ in range(120):
    try:
        os.kill(pid,0)
    except ProcessLookupError:
        raise SystemExit('Inspector exited; inspect inspector_server.log in the session folder')
    try:
        with opener.open('http://127.0.0.1:8080/health',timeout=1) as response:
            if response.status==200:
                print('Inspector ready.',flush=True)
                break
    except Exception:
        pass
    time.sleep(1)
else:
    raise SystemExit('Inspector startup deadline exceeded; inspect the saved server log')
PY
if [[ "$#" == 0 ]]; then
  set -- --duration 60 --preview
fi
# Capture/inference application has its own finite duration, startup and shutdown deadlines.
"$python_bin" -u main.py --config "$config" --run "$session/client" "$@" \
  2>&1 | tee "$session/client_terminal.log"
