from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from .encoder import AverageInitializedEncoder


DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "lm_head",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def load_tokenizer_and_model(
    model_name: str,
    device: torch.device | str,
    trust_remote_code: bool = True,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    ).to(device)
    return tokenizer, model


def resolve_pad_token_id(tokenizer) -> int:
    space_tokens = tokenizer(" ", add_special_tokens=False)["input_ids"]
    if space_tokens:
        return space_tokens[0]
    if tokenizer.pad_token_id is not None:
        return tokenizer.pad_token_id
    if tokenizer.eos_token_id is not None:
        return tokenizer.eos_token_id
    raise ValueError("Unable to determine a padding token id.")


def build_default_lora_config(
    rank: int = 4,
    alpha: int = 16,
    dropout: float = 0.05,
) -> LoraConfig:
    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=DEFAULT_TARGET_MODULES,
    )


def load_peft_model(
    model: torch.nn.Module,
    peft_path: str | Path | None = None,
    trainable: bool = True,
    lora_config: LoraConfig | None = None,
) -> torch.nn.Module:
    if peft_path:
        return PeftModel.from_pretrained(model, peft_path, is_trainable=trainable)

    lora_config = lora_config or build_default_lora_config()
    peft_model = get_peft_model(model, lora_config)
    for name, parameter in peft_model.named_parameters():
        if "lora" in name.lower():
            parameter.requires_grad = True
    return peft_model


def load_compressor(
    embedding_dim: int,
    merge_factor: int,
    device: torch.device | str,
    checkpoint_path: str | Path | None = None,
) -> AverageInitializedEncoder:
    compressor = AverageInitializedEncoder(
        embedding_dim=embedding_dim,
        merge_factor=merge_factor,
    ).to(device)
    if checkpoint_path:
        compressor_state = torch.load(checkpoint_path, map_location=device)
        normalized_state = {key.replace("module.", ""): value for key, value in compressor_state.items()}
        compressor.load_state_dict(normalized_state)
    return compressor


def maybe_load_optimizer_state(
    optimizer: torch.optim.Optimizer,
    checkpoint_path: str | Path | None,
    device: torch.device | str,
) -> None:
    if checkpoint_path and Path(checkpoint_path).exists():
        optimizer.load_state_dict(torch.load(checkpoint_path, map_location=device))


def save_artifacts(
    output_dir: str | Path,
    peft_model: torch.nn.Module,
    compressor: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    metadata: dict | None = None,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(output_dir)
    torch.save(compressor.state_dict(), output_dir / "encoder.pth")
    if optimizer is not None:
        torch.save(optimizer.state_dict(), output_dir / "optimizer.pt")
    if metadata is not None:
        save_json(output_dir / "metrics.json", metadata)

