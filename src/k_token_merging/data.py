from __future__ import annotations

from dataclasses import dataclass

from torch.utils.data import Dataset


@dataclass(frozen=True)
class PromptExample:
    prompt: str
    answer: str


class TextPairDataset(Dataset):
    def __init__(self, examples: list[PromptExample]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[str, str]:
        example = self.examples[index]
        return example.prompt, example.answer

