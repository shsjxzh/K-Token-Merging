from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract a base model embedding table.")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--output-file", default="artifacts/qwen2.5_0.5b_embeddings_id_full.pkl")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=resolve_dtype(args.dtype),
        device_map=None,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(args.device)
    model.eval()

    embedding_layer = model.get_input_embeddings()
    weights = embedding_layer.weight.detach().to("cpu").float()
    id_to_embedding = [weights[i].numpy() for i in range(weights.size(0))]

    with open(output_path, "wb") as handle:
        pickle.dump(id_to_embedding, handle, protocol=4)

    print(f"Saved {len(id_to_embedding)} embeddings to {output_path.resolve()}")


if __name__ == "__main__":
    main()
