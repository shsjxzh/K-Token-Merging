from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from datasets import load_dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from k_token_merging import (
    build_prefill_embeddings,
    build_training_batch,
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
from k_token_merging.modeling import unwrap_module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Amazon Reviews benchmark from K-Token Merging.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    add_shared_args(train_parser)
    train_parser.add_argument("--dataset-name", default="Amazon_Fashion")
    train_parser.add_argument("--test-size", type=int, default=50000)
    train_parser.add_argument("--train-sample-fraction", type=float, default=0.1)
    train_parser.add_argument("--per-gpu-batch-size", type=int, default=2)
    train_parser.add_argument("--per-gpu-eval-batch-size", type=int, default=16)
    train_parser.add_argument("--epochs", type=int, default=1)
    train_parser.add_argument("--learning-rate", type=float, default=1e-4)
    train_parser.add_argument("--grad-accum-steps", type=int, default=1)
    train_parser.add_argument("--save-steps", type=int, default=500)
    train_parser.add_argument("--output-dir", type=Path, default=Path("outputs/amazon_reviews"))
    train_parser.add_argument("--resume-peft", type=Path)
    train_parser.add_argument("--resume-encoder", type=Path)
    train_parser.add_argument("--resume-optimizer", type=Path)

    eval_parser = subparsers.add_parser("evaluate")
    add_shared_args(eval_parser)
    eval_parser.add_argument("--dataset-name", default="Amazon_Fashion")
    eval_parser.add_argument("--test-size", type=int, default=50000)
    eval_parser.add_argument("--per-gpu-eval-batch-size", type=int, default=16)
    eval_parser.add_argument("--checkpoint-dir", type=Path, required=True)
    return parser.parse_args()


def add_shared_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--embedding-file", type=Path, required=True)
    parser.add_argument("--merge-factor", type=int, default=4)
    parser.add_argument("--embedding-dim", type=int, default=896)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--gpu-ids", nargs="+", type=int)
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--master-port", default="52779")
    parser.add_argument("--seed", type=int, default=42)


def resolve_gpu_ids(args: argparse.Namespace) -> list[int]:
    if args.gpu_ids:
        return args.gpu_ids
    if isinstance(args.device, str) and args.device.startswith("cuda:"):
        return [int(args.device.split(":")[1])]
    return []


def gpu_wrap(module: torch.nn.Module, device: torch.device, gpu_rank: int | None) -> torch.nn.Module:
    if gpu_rank is None:
        return DDP(module.to(device))
    return DDP(module.to(device), device_ids=[gpu_rank])


def format_example(example: dict) -> PromptExample:
    label = "positive" if example["rating"] >= 3.5 else "negative"
    return PromptExample(prompt=f"Review: {example['text']}, Label:", answer=label)


def load_examples(dataset_name: str, test_size: int, sample_fraction: float | None, seed: int) -> tuple[list[PromptExample], list[PromptExample]]:
    dataset = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        f"raw_review_{dataset_name}",
        trust_remote_code=True,
    )
    split_dataset = dataset["full"].train_test_split(test_size=test_size, seed=seed)
    train_split = split_dataset["train"]
    if sample_fraction is not None and sample_fraction < 1.0:
        sample_count = max(1, int(len(train_split) * sample_fraction))
        indices = random.sample(range(len(train_split)), sample_count)
        train_split = train_split.select(indices)

    train_examples = [format_example(item) for item in train_split]
    test_examples = [format_example(item) for item in split_dataset["test"]]
    return train_examples, test_examples


def init_process(rank: int, world_size: int, args: argparse.Namespace) -> tuple[torch.device, int | None]:
    use_cuda = bool(args.gpu_ids)
    backend = "nccl" if use_cuda else "gloo"
    dist.init_process_group(
        backend,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=3600),
    )
    if use_cuda:
        gpu_rank = args.gpu_ids[rank]
        torch.cuda.set_device(gpu_rank)
        return torch.device(f"cuda:{gpu_rank}"), gpu_rank
    return torch.device(args.device), None


def train_worker(
    rank: int,
    world_size: int,
    args: argparse.Namespace,
    peft_path: str | None,
    compressor_path: str | None,
    optimizer_path: str | None,
) -> None:
    device, gpu_rank = init_process(rank, world_size, args)
    train_examples, _ = load_examples(args.dataset_name, args.test_size, args.train_sample_fraction, args.seed)

    dataset = TextPairDataset(train_examples)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(dataset, batch_size=args.per_gpu_batch_size, sampler=sampler)

    embedding_table = load_embedding_table(args.embedding_file, device=device)
    tokenizer, base_model = load_tokenizer_and_model(args.model_name, device)
    embedding_layer = base_model.get_input_embeddings()
    compressor = load_compressor(args.embedding_dim, args.merge_factor, device, checkpoint_path=compressor_path)
    peft_model = load_peft_model(base_model, peft_path=peft_path, trainable=True)

    compressor = gpu_wrap(compressor, device, gpu_rank)
    peft_model = gpu_wrap(peft_model, device, gpu_rank)

    optimizer = AdamW(list(peft_model.parameters()) + list(compressor.parameters()), lr=args.learning_rate)
    if optimizer_path and Path(optimizer_path).exists():
        optimizer.load_state_dict(torch.load(optimizer_path, map_location=device, weights_only=True))

    peft_model.train()
    compressor.train()
    pad_token_id = resolve_pad_token_id(tokenizer)
    global_step = 0
    optimizer.zero_grad()

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        progress = tqdm(train_loader, desc=f"amazon epoch {epoch}", disable=(rank != 0), leave=False)
        for batch_index, (prompts, answers) in enumerate(progress, start=1):
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
                device=device,
            )
            outputs = peft_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits.float()
            shift_labels = F.pad(labels, (0, 1), value=tokenizer.eos_token_id)[..., 1:].contiguous()
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), shift_labels.view(-1))
            (loss / args.grad_accum_steps).backward()

            if batch_index % args.grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            global_step += 1
            if rank == 0:
                progress.set_postfix(loss=f"{loss.item():.4f}")

            if args.save_steps and global_step % args.save_steps == 0 and rank == 0:
                save_artifacts(
                    output_dir=args.output_dir / args.dataset_name / "step_last",
                    peft_model=peft_model,
                    compressor=compressor,
                    optimizer=optimizer,
                    metadata={"epoch": epoch, "step": global_step, "loss": float(loss.item())},
                )

        if len(train_loader) % args.grad_accum_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

    if rank == 0:
        save_artifacts(
            output_dir=args.output_dir / args.dataset_name / "step_last",
            peft_model=peft_model,
            compressor=compressor,
            optimizer=optimizer,
            metadata={
                "dataset": args.dataset_name,
                "merge_factor": args.merge_factor,
                "epochs": args.epochs,
                "per_gpu_batch_size": args.per_gpu_batch_size,
                "effective_batch_size": args.per_gpu_batch_size * world_size * args.grad_accum_steps,
                "grad_accum_steps": args.grad_accum_steps,
            },
        )

    dist.destroy_process_group()


def eval_worker(rank: int, world_size: int, args: argparse.Namespace) -> None:
    device, _ = init_process(rank, world_size, args)
    checkpoint_dir = args.checkpoint_dir if getattr(args, "checkpoint_dir", None) else args.output_dir / args.dataset_name / "step_last"

    _, test_examples = load_examples(args.dataset_name, args.test_size, None, args.seed)
    dataset = TextPairDataset(test_examples)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    batch_size = args.per_gpu_eval_batch_size
    eval_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    embedding_table = load_embedding_table(args.embedding_file, device=device)
    tokenizer, base_model = load_tokenizer_and_model(args.model_name, device)
    peft_model = load_peft_model(base_model, peft_path=checkpoint_dir, trainable=False)
    compressor = load_compressor(args.embedding_dim, args.merge_factor, device, checkpoint_path=checkpoint_dir / "encoder.pth")
    embedding_layer = unwrap_module(peft_model).get_input_embeddings()
    pad_token_id = resolve_pad_token_id(tokenizer)

    peft_model.eval()
    compressor.eval()
    correct = 0
    total = 0

    for prompts, answers in tqdm(eval_loader, desc="eval", disable=(rank != 0), leave=False):
        batch_size = len(answers)
        generated_embeds = build_prefill_embeddings(
            prompts=list(prompts),
            tokenizer=tokenizer,
            embedding_table=embedding_table,
            compressor=compressor,
            merge_factor=args.merge_factor,
            embedding_dim=args.embedding_dim,
            pad_token_id=pad_token_id,
            device=device,
        )

        generated_ids = torch.full((batch_size, 1), tokenizer.eos_token_id, dtype=torch.long, device=device)
        with torch.no_grad():
            attention_mask = torch.ones(generated_embeds.shape[:-1], dtype=torch.long, device=device)
            outputs = peft_model(inputs_embeds=generated_embeds, attention_mask=attention_mask)
            next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1)
            generated_ids[:, 0] = next_token_id
            _ = embedding_layer(next_token_id)

        decoded = [item.strip() for item in tokenizer.batch_decode(generated_ids, skip_special_tokens=True)]
        gold = [item.strip() for item in answers]
        correct += sum(pred == ref for pred, ref in zip(decoded, gold))
        total += len(gold)

    correct_tensor = torch.tensor([correct], dtype=torch.float64, device=device)
    total_tensor = torch.tensor([total], dtype=torch.float64, device=device)
    dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)

    if rank == 0:
        accuracy = 100.0 * correct_tensor.item() / max(total_tensor.item(), 1.0)
        save_json(checkpoint_dir / "eval_accuracy.json", {"dataset": args.dataset_name, "accuracy": accuracy})

    dist.destroy_process_group()


def run_train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    args.gpu_ids = resolve_gpu_ids(args)
    if args.gpu_ids:
        args.device = f"cuda:{args.gpu_ids[0]}"
    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = args.master_port

    world_size = len(args.gpu_ids) if args.gpu_ids else 1
    peft_path = str(args.resume_peft) if args.resume_peft else None
    compressor_path = str(args.resume_encoder) if args.resume_encoder else None
    optimizer_path = str(args.resume_optimizer) if args.resume_optimizer else None

    mp.spawn(train_worker, args=(world_size, args, peft_path, compressor_path, optimizer_path), nprocs=world_size, join=True)
    mp.spawn(eval_worker, args=(world_size, args), nprocs=world_size, join=True)

    payload_path = args.output_dir / args.dataset_name / "step_last" / "eval_accuracy.json"
    if payload_path.exists():
        with open(payload_path, "r", encoding="utf-8") as handle:
            print(json.dumps(json.load(handle), indent=2))


def run_evaluate(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    args.gpu_ids = resolve_gpu_ids(args)
    if args.gpu_ids:
        args.device = f"cuda:{args.gpu_ids[0]}"
    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = args.master_port

    world_size = len(args.gpu_ids) if args.gpu_ids else 1
    mp.spawn(eval_worker, args=(world_size, args), nprocs=world_size, join=True)

    with open(args.checkpoint_dir / "eval_accuracy.json", "r", encoding="utf-8") as handle:
        print(json.dumps(json.load(handle), indent=2))


def main() -> None:
    args = parse_args()
    if args.command == "train":
        run_train(args)
        return
    run_evaluate(args)


if __name__ == "__main__":
    main()
