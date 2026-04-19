from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from k_token_merging import (
    build_training_batch,
    load_compressor,
    load_embedding_table,
    load_peft_model,
    load_tokenizer_and_model,
    perplexity,
    resolve_pad_token_id,
    save_artifacts,
    save_json,
    set_seed,
)
from k_token_merging.data import PromptExample, TextPairDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the CommitPackFT benchmark from K-Token Merging.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    add_shared_args(train_parser)
    train_parser.add_argument("--language", default="python")
    train_parser.add_argument("--test-size", type=int, default=5600)
    train_parser.add_argument("--batch-size", type=int, default=12)
    train_parser.add_argument("--eval-batch-size", type=int, default=24)
    train_parser.add_argument("--epochs", type=int, default=1)
    train_parser.add_argument("--learning-rate", type=float, default=1e-4)
    train_parser.add_argument("--grad-accum-steps", type=int, default=1)
    train_parser.add_argument("--save-steps", type=int, default=500)
    train_parser.add_argument("--output-dir", type=Path, default=Path("outputs/commitpackft"))
    train_parser.add_argument("--resume-peft", type=Path)
    train_parser.add_argument("--resume-encoder", type=Path)

    eval_parser = subparsers.add_parser("evaluate")
    add_shared_args(eval_parser)
    eval_parser.add_argument("--language", default="python")
    eval_parser.add_argument("--test-size", type=int, default=5600)
    eval_parser.add_argument("--batch-size", type=int, default=24)
    eval_parser.add_argument("--checkpoint-dir", type=Path, required=True)
    return parser.parse_args()


def add_shared_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--embedding-file", type=Path, required=True)
    parser.add_argument("--merge-factor", type=int, default=2)
    parser.add_argument("--embedding-dim", type=int, default=896)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)


def format_example(example: dict) -> PromptExample:
    prompt = (
        "Update the old python code content based on the instrutions.\n"
        f"Old Content: {example['old_contents']}\n"
        f"Instructions: {example['subject']}\n"
        "New Content:\n"
    )
    return PromptExample(prompt=prompt, answer=example["new_contents"])


def load_examples(language: str, test_size: int, seed: int) -> tuple[list[PromptExample], list[PromptExample]]:
    dataset = load_dataset("bigcode/commitpackft", language, trust_remote_code=True)
    split_dataset = dataset["train"].train_test_split(test_size=test_size, seed=seed)
    train_examples = [format_example(item) for item in split_dataset["train"]]
    test_examples = [format_example(item) for item in split_dataset["test"]]
    return train_examples, test_examples


def run_train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    train_examples, test_examples = load_examples(args.language, args.test_size, args.seed)

    tokenizer, base_model = load_tokenizer_and_model(args.model_name, args.device)
    peft_model = load_peft_model(base_model, peft_path=args.resume_peft, trainable=True)
    compressor = load_compressor(
        embedding_dim=args.embedding_dim,
        merge_factor=args.merge_factor,
        device=args.device,
        checkpoint_path=args.resume_encoder,
    )
    embedding_table = load_embedding_table(args.embedding_file, device=args.device)
    embedding_layer = peft_model.get_input_embeddings()
    pad_token_id = resolve_pad_token_id(tokenizer)

    train_loader = DataLoader(TextPairDataset(train_examples), batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(TextPairDataset(test_examples), batch_size=args.eval_batch_size, shuffle=False)
    optimizer = AdamW(list(peft_model.parameters()) + list(compressor.parameters()), lr=args.learning_rate)

    peft_model.train()
    compressor.train()
    global_step = 0
    optimizer.zero_grad()

    for epoch in range(args.epochs):
        progress = tqdm(train_loader, desc=f"commitpack epoch {epoch}", leave=False)
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
            loss = loss / args.grad_accum_steps
            loss.backward()

            if (global_step + 1) % args.grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            global_step += 1
            progress.set_postfix(loss=f"{loss.item():.4f}")

            if args.save_steps and global_step % args.save_steps == 0:
                checkpoint_dir = args.output_dir / args.language / "step_last"
                save_artifacts(
                    output_dir=checkpoint_dir,
                    peft_model=peft_model,
                    compressor=compressor,
                    optimizer=optimizer,
                    metadata={"epoch": epoch, "step": global_step, "loss": float(loss.item())},
                )

        if global_step % args.grad_accum_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

    ppl = perplexity(
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
    checkpoint_dir = args.output_dir / args.language / "step_last"
    save_artifacts(
        output_dir=checkpoint_dir,
        peft_model=peft_model,
        compressor=compressor,
        optimizer=optimizer,
        metadata={
            "language": args.language,
            "perplexity": ppl,
            "merge_factor": args.merge_factor,
            "epochs": args.epochs,
        },
    )


def run_evaluate(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    _, test_examples = load_examples(args.language, args.test_size, args.seed)

    tokenizer, base_model = load_tokenizer_and_model(args.model_name, args.device)
    peft_model = load_peft_model(base_model, peft_path=args.checkpoint_dir, trainable=False)
    compressor = load_compressor(
        embedding_dim=args.embedding_dim,
        merge_factor=args.merge_factor,
        device=args.device,
        checkpoint_path=args.checkpoint_dir / "encoder.pth",
    )
    embedding_table = load_embedding_table(args.embedding_file, device=args.device)
    embedding_layer = peft_model.get_input_embeddings()
    pad_token_id = resolve_pad_token_id(tokenizer)

    eval_loader = DataLoader(TextPairDataset(test_examples), batch_size=args.batch_size, shuffle=False)
    ppl = perplexity(
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
    payload = {"language": args.language, "perplexity": ppl}
    save_json(args.checkpoint_dir / "eval_perplexity.json", payload)
    print(json.dumps(payload, indent=2))


def main() -> None:
    args = parse_args()
    if args.command == "train":
        run_train(args)
        return
    run_evaluate(args)


if __name__ == "__main__":
    main()
