from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class GenerationResult:
    answer: str
    input_tokens: int
    output_tokens: int


class BaseGenerator(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> GenerationResult:
        raise NotImplementedError
