from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class StageConfig:
    max_depth: int
    max_nodes: int
    min_children: int
    max_children: int
    ref_levels: tuple[int, ...]


CURRICULUM_STAGE_CONFIGS: dict[str, StageConfig] = {
    "small": StageConfig(max_depth=2, max_nodes=3, min_children=0, max_children=2, ref_levels=(0, 1)),
    "xsmall": StageConfig(max_depth=3, max_nodes=5, min_children=0, max_children=2, ref_levels=(0, 1, 2)),
    "medium": StageConfig(max_depth=3, max_nodes=10, min_children=0, max_children=3, ref_levels=(0, 1, 2)),
    "xmedium": StageConfig(max_depth=4, max_nodes=15, min_children=1, max_children=3, ref_levels=(1, 2, 3, 4)),
    "large": StageConfig(max_depth=4, max_nodes=30, min_children=1, max_children=3, ref_levels=(2, 3, 4)),
    "x3large": StageConfig(max_depth=4, max_nodes=150, min_children=3, max_children=5, ref_levels=(2, 3, 4)),
}


DEFAULT_NUM_TREES = 500000


@dataclass
class Node:
    value: str
    children: list["Node"] = field(default_factory=list)


class TreeGenerator:
    def __init__(self, max_depth: int, max_nodes: int, min_children: int = 0, max_children: int = 3) -> None:
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self.min_children = min_children
        self.max_children = max_children
        self.node_count = 0
        self.node_values: list[str] = []
        self.reset_node_values()

    def reset_node_values(self) -> None:
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self.node_values = [a + b for a in alphabet for b in alphabet]

    def generate_tree(self) -> Node | None:
        self.node_count = 0
        return self._generate_node(depth=0)

    def _generate_node(self, depth: int) -> Node | None:
        if self.node_count >= self.max_nodes or not self.node_values:
            return None

        node_value = random.choice(self.node_values)
        self.node_values.remove(node_value)
        node = Node(node_value)
        self.node_count += 1

        if depth >= self.max_depth:
            return node

        remaining_nodes = self.max_nodes - self.node_count
        max_possible_children = min(self.max_children, remaining_nodes)
        if max_possible_children < self.min_children:
            return node

        num_children = random.randint(self.min_children, max_possible_children)
        for _ in range(num_children):
            child = self._generate_node(depth + 1)
            if child is not None:
                node.children.append(child)
        return node

    def render(self, node: Node | None, level: int = 0) -> str:
        if node is None:
            return ""
        rows = ["+" * level + node.value]
        for child in node.children:
            rows.append(self.render(child, level + 1))
        return "\n".join(row for row in rows if row)


def gather_parent_child_examples(root: Node | None, rendered_tree: str) -> list[dict]:
    if root is None:
        return []

    examples: list[dict] = []

    def visit(node: Node) -> None:
        for child in node.children:
            examples.append(
                {
                    "indent_tree": rendered_tree,
                    "parent": node.value,
                    "child": child.value,
                    "label": "true",
                    "task_type": "parent_child",
                }
            )
            visit(child)

    def collect_nodes(node: Node) -> list[Node]:
        result = [node]
        for child in node.children:
            result.extend(collect_nodes(child))
        return result

    visit(root)
    all_nodes = collect_nodes(root)
    parent_child_pairs = {(item["parent"], item["child"]) for item in examples}

    for parent in all_nodes:
        for child in all_nodes:
            if parent.value == child.value:
                continue
            if (parent.value, child.value) in parent_child_pairs:
                continue
            examples.append(
                {
                    "indent_tree": rendered_tree,
                    "parent": parent.value,
                    "child": child.value,
                    "label": "false",
                    "task_type": "parent_child",
                }
            )
    return examples


def infer_stage_name(save_dir: Path, stage_name: str | None = None) -> str:
    if stage_name:
        return stage_name
    folder_name = save_dir.name
    if folder_name.startswith("tree_data_"):
        return folder_name.removeprefix("tree_data_")
    return folder_name


def write_split_manifests(
    save_dir: Path,
    *,
    stage_name: str | None = None,
    train_ratio: float = 0.98,
    seed: int = 42,
) -> None:
    json_files = sorted(path.name for path in save_dir.glob("*.json"))
    if not json_files:
        raise ValueError(f"No JSON files found in {save_dir}")

    rng = random.Random(seed)
    rng.shuffle(json_files)

    split_idx = int(len(json_files) * train_ratio)
    split_idx = min(max(split_idx, 1), len(json_files) - 1) if len(json_files) > 1 else len(json_files)
    train_files = json_files[:split_idx]
    test_files = json_files[split_idx:]
    stage = infer_stage_name(save_dir, stage_name)

    train_csv = save_dir / f"train_file_{stage}.csv"
    test_csv = save_dir / f"test_file_{stage}.csv"

    def write_csv(csv_path: Path, filenames: list[str]) -> None:
        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["filename"])
            writer.writerows([[name] for name in filenames])

    write_csv(train_csv, train_files)
    write_csv(test_csv, test_files)


def generate_dataset(
    *,
    max_depth: int,
    max_nodes: int,
    min_children: int,
    max_children: int,
    num_trees: int,
    save_dir: Path,
    seed: int = 42,
    stage_name: str | None = None,
    write_splits: bool = True,
    train_ratio: float = 0.98,
) -> None:
    random.seed(seed)
    save_dir.mkdir(parents=True, exist_ok=True)

    generator = TreeGenerator(
        max_depth=max_depth,
        max_nodes=max_nodes,
        min_children=min_children,
        max_children=max_children,
    )

    for index in range(num_trees):
        generator.reset_node_values()
        tree = generator.generate_tree()
        rendered_tree = generator.render(tree)
        examples = gather_parent_child_examples(tree, rendered_tree)
        with open(save_dir / f"tree_{index:06d}.json", "w", encoding="utf-8") as handle:
            json.dump(examples, handle, indent=2)

    if write_splits:
        write_split_manifests(
            save_dir,
            stage_name=stage_name,
            train_ratio=train_ratio,
            seed=seed,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate one textualized-tree dataset used in the paper.")
    parser.add_argument("--max-depth", type=int, required=True)
    parser.add_argument("--max-nodes", type=int, required=True)
    parser.add_argument("--min-children", type=int, default=0)
    parser.add_argument("--max-children", type=int, default=3)
    parser.add_argument("--num-trees", type=int, default=DEFAULT_NUM_TREES)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--stage-name", type=str)
    parser.add_argument("--train-ratio", type=float, default=0.98)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-splits", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_dataset(
        max_depth=args.max_depth,
        max_nodes=args.max_nodes,
        min_children=args.min_children,
        max_children=args.max_children,
        num_trees=args.num_trees,
        save_dir=args.save_dir,
        seed=args.seed,
        stage_name=args.stage_name,
        write_splits=not args.skip_splits,
        train_ratio=args.train_ratio,
    )
    print(f"Saved {args.num_trees} tree files to {args.save_dir}")


if __name__ == "__main__":
    main()
