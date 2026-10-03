#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -x "$script_dir/.venv/bin/python" ]]; then
  # --without-pip also works on minimal Python installations lacking ensurepip.
  python3 -m venv --without-pip "$script_dir/.venv"
fi
if "$script_dir/.venv/bin/python" -m pip --version >/dev/null 2>&1; then
  "$script_dir/.venv/bin/python" -m pip install --upgrade pip
  "$script_dir/.venv/bin/python" -m pip install -r "$script_dir/requirements.txt"
else
  python3 -m pip --python "$script_dir/.venv" install --upgrade pip
  python3 -m pip --python "$script_dir/.venv" install -r "$script_dir/requirements.txt"
fi
