from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field
from pathlib import Path


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate textualized tree data used in the paper.")
    parser.add_argument("--max-depth", type=int, required=True)
    parser.add_argument("--max-nodes", type=int, required=True)
    parser.add_argument("--min-children", type=int, default=0)
    parser.add_argument("--max-children", type=int, default=3)
    parser.add_argument("--num-trees", type=int, default=1000)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    args.save_dir.mkdir(parents=True, exist_ok=True)

    generator = TreeGenerator(
        max_depth=args.max_depth,
        max_nodes=args.max_nodes,
        min_children=args.min_children,
        max_children=args.max_children,
    )

    for index in range(args.num_trees):
        generator.node_values = [chr(i) + chr(j) for i in range(65, 91) for j in range(65, 91)]
        tree = generator.generate_tree()
        rendered_tree = generator.render(tree)
        examples = gather_parent_child_examples(tree, rendered_tree)
        with open(args.save_dir / f"tree_{index:06d}.json", "w", encoding="utf-8") as handle:
            json.dump(examples, handle, indent=2)

    print(f"Saved {args.num_trees} tree files to {args.save_dir}")


if __name__ == "__main__":
    main()

