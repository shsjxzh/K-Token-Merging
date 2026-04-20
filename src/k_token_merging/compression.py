from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def load_embedding_table(path: str | Path, device: torch.device | str | None = None) -> torch.Tensor:
    with open(path, "rb") as handle:
        embedding_table = pickle.load(handle)

    if torch.is_tensor(embedding_table):
        pass
    elif isinstance(embedding_table, np.ndarray):
        embedding_table = torch.from_numpy(embedding_table)
    else:
        embedding_table = torch.from_numpy(np.asarray(embedding_table))
    embedding_table = embedding_table.float()
    if device is not None:
        embedding_table = embedding_table.to(device)
    return embedding_table


def pad_to_multiple(input_ids: torch.Tensor, multiple: int, pad_token_id: int) -> torch.Tensor:
    trailing = input_ids.shape[1] % multiple
    if trailing == 0:
        return input_ids

    pad_width = multiple - trailing
    pad_column = torch.full(
        (input_ids.shape[0], pad_width),
        pad_token_id,
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    return torch.cat([input_ids, pad_column], dim=1)


def compress_prompt_input_ids(
    input_ids: torch.Tensor,
    embedding_table: torch.Tensor,
    compressor: torch.nn.Module,
    merge_factor: int,
    embedding_dim: int,
    pad_token_id: int,
) -> torch.Tensor:
    padded_input_ids = pad_to_multiple(input_ids, merge_factor, pad_token_id)
    embeds = F.embedding(padded_input_ids, embedding_table)
    embeds = embeds.view(-1, merge_factor * embedding_dim)
    return compressor(embeds).view(padded_input_ids.size(0), -1, embedding_dim)


def build_training_batch(
    prompts: list[str],
    answers: list[str],
    tokenizer,
    embedding_table: torch.Tensor,
    compressor: torch.nn.Module,
    embedding_layer: torch.nn.Module,
    merge_factor: int,
    embedding_dim: int,
    pad_token_id: int,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    original_padding_side = tokenizer.padding_side

    tokenizer.padding_side = "left"
    prompt_enc = tokenizer(prompts, return_tensors="pt", padding=True)
    tokenizer.padding_side = "right"
    answer_enc = tokenizer(answers, return_tensors="pt", padding=True)
    tokenizer.padding_side = original_padding_side

    prompt_ids = prompt_enc["input_ids"].to(device)
    answer_ids = answer_enc["input_ids"].to(device)
    answer_attention = answer_enc["attention_mask"].to(device)
    answer_padded_length = answer_ids.shape[1]

    compressed = compress_prompt_input_ids(
        input_ids=prompt_ids,
        embedding_table=embedding_table,
        compressor=compressor,
        merge_factor=merge_factor,
        embedding_dim=embedding_dim,
        pad_token_id=pad_token_id,
    )

    answer_embeds = embedding_layer(answer_ids)
    full_embeddings = [torch.cat([comp, ans], dim=0) for comp, ans in zip(compressed, answer_embeds)]
    inputs_embeds = torch.stack(full_embeddings)

    labels = torch.full(
        (inputs_embeds.size(0), inputs_embeds.size(1)),
        -100,
        dtype=torch.long,
        device=device,
    )
    labels[:, -answer_padded_length:] = answer_ids
    labels[:, -answer_padded_length:][answer_attention == 0] = -100

    attention_mask = torch.ones_like(labels)
    attention_mask[:, -answer_padded_length:] = answer_attention
    return inputs_embeds, attention_mask, labels


def build_prefill_embeddings(
    prompts: list[str],
    tokenizer,
    embedding_table: torch.Tensor,
    compressor: torch.nn.Module,
    merge_factor: int,
    embedding_dim: int,
    pad_token_id: int,
    device: torch.device | str,
) -> torch.Tensor:
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    prompt_enc = tokenizer(prompts, return_tensors="pt", padding=True)
    tokenizer.padding_side = original_padding_side

    prompt_ids = prompt_enc["input_ids"].to(device)
    with torch.no_grad():
        return compress_prompt_input_ids(
            input_ids=prompt_ids,
            embedding_table=embedding_table,
            compressor=compressor,
            merge_factor=merge_factor,
            embedding_dim=embedding_dim,
            pad_token_id=pad_token_id,
        )
