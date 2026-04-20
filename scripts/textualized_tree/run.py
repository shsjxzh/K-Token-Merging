from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from k_token_merging import (
    build_training_batch,
    classification_accuracy,
    load_compressor,
    load_embedding_table,
    load_peft_model,
    load_tokenizer_and_model,
    resolve_pad_token_id,
    save_artifacts,
    save_json,
    set_seed,
)
from k_token_merging.data import PromptExample, TextPairDataset


DEFAULT_STAGES = ["small", "xsmall", "medium", "xmedium", "large", "x3large"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Textualized Tree benchmark from K-Token Merging.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    add_shared_args(train_parser)
    train_parser.add_argument("--tree-data-root", type=Path, required=True)
    train_parser.add_argument("--stages", nargs="+", default=DEFAULT_STAGES)
    train_parser.add_argument("--epochs-per-stage", type=int, default=1)
    train_parser.add_argument("--batch-size", type=int, default=16)
    train_parser.add_argument("--eval-batch-size", type=int, default=32)
    train_parser.add_argument("--learning-rate", type=float, default=1e-4)
    train_parser.add_argument("--save-steps", type=int, default=500)
    train_parser.add_argument("--output-dir", type=Path, default=Path("outputs/textualized_tree"))
    train_parser.add_argument("--resume-peft", type=Path)
    train_parser.add_argument("--resume-encoder", type=Path)
    train_parser.add_argument("--resume-optimizer", type=Path)

    eval_parser = subparsers.add_parser("evaluate")
    add_shared_args(eval_parser)
    eval_parser.add_argument("--tree-data-root", type=Path, required=True)
    eval_parser.add_argument("--stage", required=True)
    eval_parser.add_argument("--batch-size", type=int, default=32)
    eval_parser.add_argument("--checkpoint-dir", type=Path, required=True)
    return parser.parse_args()


def add_shared_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--embedding-file", type=Path, required=True)
    parser.add_argument("--merge-factor", type=int, default=4)
    parser.add_argument("--embedding-dim", type=int, default=896)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)


def process_tree_file(data_dir: Path, filename: str) -> list[dict]:
    file_path = data_dir / filename
    try:
        with open(file_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return []
    return [item for item in data if item.get("task_type") == "parent_child"]


def load_tree_examples(data_dir: Path, csv_path: Path) -> list[PromptExample]:
    file_list = pd.read_csv(csv_path)["filename"].tolist()
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        for result in executor.map(lambda name: process_tree_file(data_dir, name), file_list):
            rows.extend(result)

    return [
        PromptExample(
            prompt=(
                f"Tree: {item['indent_tree']}, Task: Parent/Child Relationship, "
                f"Parent: {item['parent']}, Child: {item['child']}, Answer: "
            ),
            answer=item["label"],
        )
        for item in rows
    ]


def train_stage(
    stage: str,
    args: argparse.Namespace,
    tokenizer,
    peft_model: torch.nn.Module,
    compressor: torch.nn.Module,
    embedding_table: torch.Tensor,
) -> tuple[torch.nn.Module, torch.nn.Module]:
    data_dir = args.tree_data_root / f"tree_data_{stage}"
    train_csv = data_dir / f"train_file_{stage}.csv"
    test_csv = data_dir / f"test_file_{stage}.csv"

    train_examples = load_tree_examples(data_dir, train_csv)
    test_examples = load_tree_examples(data_dir, test_csv)
    train_loader = DataLoader(TextPairDataset(train_examples), batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(TextPairDataset(test_examples), batch_size=args.eval_batch_size, shuffle=False)

    optimizer = AdamW(list(peft_model.parameters()) + list(compressor.parameters()), lr=args.learning_rate)
    if args.resume_optimizer and args.resume_optimizer.exists():
        optimizer.load_state_dict(torch.load(args.resume_optimizer, map_location=args.device))
        args.resume_optimizer = None

    peft_model.train()
    compressor.train()
    embedding_layer = peft_model.get_input_embeddings()
    pad_token_id = resolve_pad_token_id(tokenizer)
    global_step = 0

    for epoch in range(args.epochs_per_stage):
        progress = tqdm(train_loader, desc=f"{stage} epoch {epoch}", leave=False)
        for prompts, answers in progress:
            inputs_embeds, attention_mask, labels = build_training_batch(
                prompts=list(prompts),
                answers=list(answers),
                tokenizer=tokenizer,
                embedding_table=embedding_table,
                compressor=compressor,
                embedding_layer=embedding_layer,
                merge_factor=args.merge_factor,
                embedding_dim=args.embedding_dim,
                pad_token_id=pad_token_id,
                device=args.device,
            )

            outputs = peft_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits.float()
            shift_labels = F.pad(labels, (0, 1), value=tokenizer.eos_token_id)[..., 1:].contiguous()
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), shift_labels.view(-1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
            progress.set_postfix(loss=f"{loss.item():.4f}")

            if args.save_steps and global_step % args.save_steps == 0:
                stage_output_dir = args.output_dir / stage / "step_last"
                save_artifacts(
                    output_dir=stage_output_dir,
                    peft_model=peft_model,
                    compressor=compressor,
                    optimizer=optimizer,
                    metadata={"stage": stage, "epoch": epoch, "step": global_step, "loss": float(loss.item())},
                )

    stage_output_dir = args.output_dir / stage / "step_last"
    accuracy = classification_accuracy(
        dataloader=eval_loader,
        tokenizer=tokenizer,
        peft_model=peft_model,
        compressor=compressor,
        embedding_table=embedding_table,
        embedding_layer=embedding_layer,
        merge_factor=args.merge_factor,
        embedding_dim=args.embedding_dim,
        pad_token_id=pad_token_id,
        device=args.device,
    )
    save_artifacts(
        output_dir=stage_output_dir,
        peft_model=peft_model,
        compressor=compressor,
        optimizer=optimizer,
        metadata={
            "stage": stage,
            "accuracy": accuracy,
            "merge_factor": args.merge_factor,
            "batch_size": args.batch_size,
            "epochs_per_stage": args.epochs_per_stage,
        },
    )
    return peft_model, compressor


def run_train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    tokenizer, base_model = load_tokenizer_and_model(args.model_name, args.device)
    peft_model = load_peft_model(base_model, peft_path=args.resume_peft, trainable=True)
    compressor = load_compressor(
        embedding_dim=args.embedding_dim,
        merge_factor=args.merge_factor,
        device=args.device,
        checkpoint_path=args.resume_encoder,
    )
    embedding_table = load_embedding_table(args.embedding_file, device=args.device)

    for stage in args.stages:
        peft_model, compressor = train_stage(stage, args, tokenizer, peft_model, compressor, embedding_table)
        args.resume_peft = args.output_dir / stage / "step_last"
        args.resume_encoder = args.output_dir / stage / "step_last" / "encoder.pth"


def run_evaluate(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    tokenizer, base_model = load_tokenizer_and_model(args.model_name, args.device)
    peft_model = load_peft_model(base_model, peft_path=args.checkpoint_dir, trainable=False)
    compressor = load_compressor(
        embedding_dim=args.embedding_dim,
        merge_factor=args.merge_factor,
        device=args.device,
        checkpoint_path=args.checkpoint_dir / "encoder.pth",
    )
    embedding_table = load_embedding_table(args.embedding_file, device=args.device)
    pad_token_id = resolve_pad_token_id(tokenizer)
    embedding_layer = peft_model.get_input_embeddings()

    data_dir = args.tree_data_root / f"tree_data_{args.stage}"
    test_csv = data_dir / f"test_file_{args.stage}.csv"
    test_examples = load_tree_examples(data_dir, test_csv)
    test_loader = DataLoader(TextPairDataset(test_examples), batch_size=args.batch_size, shuffle=False)

    accuracy = classification_accuracy(
        dataloader=test_loader,
        tokenizer=tokenizer,
        peft_model=peft_model,
        compressor=compressor,
        embedding_table=embedding_table,
        embedding_layer=embedding_layer,
        merge_factor=args.merge_factor,
        embedding_dim=args.embedding_dim,
        pad_token_id=pad_token_id,
        device=args.device,
    )
    payload = {"stage": args.stage, "accuracy": accuracy}
    save_json(args.checkpoint_dir / "eval_accuracy.json", payload)
    print(json.dumps(payload, indent=2))


def main() -> None:
    args = parse_args()
    if args.command == "train":
        run_train(args)
        return
    run_evaluate(args)


if __name__ == "__main__":
    main()
