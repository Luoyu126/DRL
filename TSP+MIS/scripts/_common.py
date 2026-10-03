from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import load_config, output_dirs  # noqa: E402


def make_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=str(ROOT / "config.json"))
    parser.add_argument("--output", default=str(ROOT / "outputs"))
    parser.add_argument("--smoke", action="store_true")
    return parser


def setup(args):
    return load_config(args.config, args.smoke), output_dirs(args.output)

