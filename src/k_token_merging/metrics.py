from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .compression import build_prefill_embeddings, build_training_batch


def classification_accuracy(
    dataloader: DataLoader,
    tokenizer,
    peft_model: torch.nn.Module,
    compressor: torch.nn.Module,
    embedding_table: torch.Tensor,
    embedding_layer: torch.nn.Module,
    merge_factor: int,
    embedding_dim: int,
    pad_token_id: int,
    device: torch.device | str,
    max_new_tokens: int = 1,
) -> float:
    peft_model.eval()
    compressor.eval()
    correct = 0
    total = 0

    for prompts, answers in tqdm(dataloader, desc="eval", leave=False):
        batch_size = len(answers)
        generated_embeds = build_prefill_embeddings(
            prompts=list(prompts),
            tokenizer=tokenizer,
            embedding_table=embedding_table,
            compressor=compressor,
            merge_factor=merge_factor,
            embedding_dim=embedding_dim,
            pad_token_id=pad_token_id,
            device=device,
        )

        generated_ids = torch.full(
            (batch_size, max_new_tokens),
            tokenizer.eos_token_id,
            dtype=torch.long,
            device=device,
        )

        for step in range(max_new_tokens):
            with torch.no_grad():
                attention_mask = torch.ones(generated_embeds.shape[:-1], dtype=torch.long, device=device)
                outputs = peft_model(inputs_embeds=generated_embeds, attention_mask=attention_mask)
                next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1)
                generated_ids[:, step] = next_token_id
                next_embed = embedding_layer(next_token_id).view(batch_size, 1, -1)
                generated_embeds = torch.cat([generated_embeds, next_embed], dim=1)

        decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        decoded = [item.strip() for item in decoded]
        gold = [item.strip() for item in answers]
        correct += sum(pred == ref for pred, ref in zip(decoded, gold))
        total += len(gold)

    return 100.0 * correct / max(total, 1)


def perplexity(
    dataloader: DataLoader,
    tokenizer,
    peft_model: torch.nn.Module,
    compressor: torch.nn.Module,
    embedding_table: torch.Tensor,
    embedding_layer: torch.nn.Module,
    merge_factor: int,
    embedding_dim: int,
    pad_token_id: int,
    device: torch.device | str,
) -> float:
    peft_model.eval()
    compressor.eval()
    total_nll = 0.0
    total_tokens = 0

    for prompts, answers in tqdm(dataloader, desc="eval", leave=False):
        inputs_embeds, attention_mask, labels = build_training_batch(
            prompts=list(prompts),
            answers=list(answers),
            tokenizer=tokenizer,
            embedding_table=embedding_table,
            compressor=compressor,
            embedding_layer=embedding_layer,
            merge_factor=merge_factor,
            embedding_dim=embedding_dim,
            pad_token_id=pad_token_id,
            device=device,
        )

        with torch.no_grad():
            outputs = peft_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits.float()

        shift_labels = F.pad(labels, (0, 1), value=tokenizer.eos_token_id)[..., 1:].contiguous()
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
        )

        mask = shift_labels.view(-1) != -100
        total_nll += loss[mask].sum().item()
        total_tokens += int(mask.sum().item())

    if total_tokens == 0:
        return float("inf")
    return math.exp(total_nll / total_tokens)
