from abc import ABC, abstractmethod

from datasets import Dataset


class BaseLabeler(ABC):
    def __init__(self, output_column_name: str = "output", budget: int = 1_000_000):
        self.output_column_name = output_column_name
        self.budget = budget if budget is not None else 1e15
        self.is_out_of_budget = False

    @abstractmethod
    def __call__(self, dataset: Dataset) -> Dataset:
        pass
