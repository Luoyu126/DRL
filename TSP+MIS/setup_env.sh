#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -x "$project_dir/.venv/bin/python" ]]; then
  python3 -m venv --without-pip "$project_dir/.venv"
fi
if "$project_dir/.venv/bin/python" -m pip --version >/dev/null 2>&1; then
  "$project_dir/.venv/bin/python" -m pip install -r "$project_dir/requirements.txt"
else
  python3 -m pip --python "$project_dir/.venv" install -r "$project_dir/requirements.txt"
fi

