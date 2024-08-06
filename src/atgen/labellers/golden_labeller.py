from datasets import Dataset

from .base_labeller import BaseLabeler


# Labels rows by using their label (should be used when evaluating strategies)
class GoldenLabeler(BaseLabeler):
    def __call__(self, dataset: Dataset) -> Dataset:
        assert (
            self.output_column_name in dataset.column_names
        ), "Column with labels was not found in dataset"
        return dataset
