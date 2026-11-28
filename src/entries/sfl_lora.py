"""Run a fixed-cut LoRA-SFL job from its experiment YAML."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from entries.app.sfl_lora_entry import SflLoRAAppEntry


def main(argv=None, *, diagnostics_factory=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--smoke", action="store_true", help="Use four samples per client split and up to two rounds")
    parser.add_argument("--final", action="store_true", help="Evaluate the locked job on its final holdout")
    args = parser.parse_args(argv)
    entry = SflLoRAAppEntry.from_config(args.config, args.output_dir, smoke=args.smoke)
    entry.run(final=args.final, diagnostics_factory=diagnostics_factory)
    print(f"Completed {entry.runner.completed_round} rounds in {entry.output_dir}")


if __name__ == "__main__":
    main()
