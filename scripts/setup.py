from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bnn_sim.setup_workflow import SetupOptions, build_setup_plan, run_setup_plan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Set up bnn_sim on a new machine.")
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "data"))
    parser.add_argument("--install-kaggle", action="store_true")
    parser.add_argument("--download-data", action="store_true", help="download large public NeuronIO Kaggle datasets")
    parser.add_argument("--convert-raw", action="store_true", help="convert raw NeuronIO files after setup")
    parser.add_argument("--raw-dir")
    parser.add_argument("--shard-dir")
    parser.add_argument("--shard-size", type=int, default=128)
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    options = SetupOptions(
        project_root=PROJECT_ROOT,
        data_dir=Path(args.data_dir).resolve(),
        install_kaggle=args.install_kaggle,
        download_data=args.download_data,
        convert_raw=args.convert_raw,
        raw_dir=Path(args.raw_dir).resolve() if args.raw_dir else None,
        shard_dir=Path(args.shard_dir).resolve() if args.shard_dir else None,
        shard_size=args.shard_size,
        skip_install=args.skip_install,
    )
    plan = build_setup_plan(options)
    run_setup_plan(plan, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
