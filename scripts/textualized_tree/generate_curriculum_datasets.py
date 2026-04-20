from __future__ import annotations

import argparse
import json
from pathlib import Path

from generate_data import CURRICULUM_STAGE_CONFIGS, DEFAULT_NUM_TREES, generate_dataset


DEFAULT_STAGES = ["small", "xsmall", "medium", "xmedium", "large", "x3large"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the full curriculum stack of textualized-tree datasets."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data"),
        help="Root directory that will contain tree_data_<stage> folders.",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        default=DEFAULT_STAGES,
        choices=sorted(CURRICULUM_STAGE_CONFIGS.keys()),
        help="Subset of curriculum stages to generate.",
    )
    parser.add_argument(
        "--num-trees",
        type=int,
        default=DEFAULT_NUM_TREES,
        help="Number of tree files to generate per stage.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.98)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--write-summary",
        action="store_true",
        help="Write a summary JSON describing the generated curriculum configuration.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {}

    for index, stage in enumerate(args.stages):
        config = CURRICULUM_STAGE_CONFIGS[stage]
        save_dir = args.output_root / f"tree_data_{stage}"
        stage_seed = args.seed + index

        generate_dataset(
            max_depth=config.max_depth,
            max_nodes=config.max_nodes,
            min_children=config.min_children,
            max_children=config.max_children,
            num_trees=args.num_trees,
            save_dir=save_dir,
            seed=stage_seed,
            stage_name=stage,
            write_splits=True,
            train_ratio=args.train_ratio,
        )

        summary[stage] = {
            "max_depth": config.max_depth,
            "max_nodes": config.max_nodes,
            "min_children": config.min_children,
            "max_children": config.max_children,
            "ref_levels": list(config.ref_levels),
            "num_trees": args.num_trees,
            "seed": stage_seed,
            "output_dir": str(save_dir),
        }
        print(f"Generated {stage} -> {save_dir}")

    if args.write_summary:
        summary_path = args.output_root / "curriculum_summary.json"
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
        print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
