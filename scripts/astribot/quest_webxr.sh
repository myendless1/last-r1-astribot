#!/usr/bin/env bash
set -euo pipefail

: "${QUEST_SERVER_ROOT:?Set QUEST_SERVER_ROOT to the directory containing quest_server/server.py}"
adb_bin=${ADB:-adb}
host_port=${QUEST_HOST_PORT:-8443}
device_port=${QUEST_DEVICE_PORT:-8443}
python_bin=${ASTRIBOT_ROBOT_PYTHON:-python}
adb_args=()
if [[ -n "${ADB_SERIAL:-}" ]]; then adb_args=(-s "${ADB_SERIAL}"); fi

if ! "${adb_bin}" "${adb_args[@]}" get-state 2>/dev/null | grep -qx device; then
  echo "Quest ADB is unavailable. Enable developer/USB debugging and check: adb devices -l" >&2
  exit 1
fi
if [[ ! -f "${QUEST_SERVER_ROOT}/quest_server/server.py" ]]; then
  echo "Missing ${QUEST_SERVER_ROOT}/quest_server/server.py" >&2
  exit 1
fi
"${adb_bin}" "${adb_args[@]}" reverse "tcp:${device_port}" "tcp:${host_port}"
export PYTHONPATH="${QUEST_SERVER_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
args=(--host 0.0.0.0 --port "${host_port}" --adb "${adb_bin}" --adb-device-port "${device_port}" --test --body-realign-button-index 99)
if [[ -n "${ADB_SERIAL:-}" ]]; then args+=(--adb-serial "${ADB_SERIAL}"); fi
exec "${python_bin}" -m quest_server.server "${args[@]}"
