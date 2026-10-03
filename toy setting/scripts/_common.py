from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config, output_dirs  # noqa: E402


def parser(description):
    result = argparse.ArgumentParser(description=description)
    result.add_argument("--config", default=str(ROOT / "configs" / "setting_a.yaml"))
    result.add_argument("--output", default=str(ROOT / "outputs"))
    result.add_argument("--smoke", action="store_true", help="Run a small end-to-end validation budget")
    return result


def setup(args):
    return load_config(args.config, args.smoke), output_dirs(args.output)

