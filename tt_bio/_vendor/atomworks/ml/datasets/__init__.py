import logging

# HACK: Re-export all dataset classes for backward compatibility
# In the future, we will just import from their respective modules (e.g., from .pandas_dataset import PandasDataset)
from .base import ExampleIDMixin, MolecularDataset
from .concat_dataset import ConcatDatasetWithID, FallbackDatasetWrapper, get_row_and_index_by_example_id
from .file_dataset import FileDataset
from .pandas_dataset import PandasDataset, StructuralDatasetWrapper

logger = logging.getLogger("datasets")

__all__ = [
    "ConcatDatasetWithID",
    "ExampleIDMixin",
    "FallbackDatasetWrapper",
    "FileDataset",
    "MolecularDataset",
    "PandasDataset",
    "StructuralDatasetWrapper",
    "get_row_and_index_by_example_id",
    "logger",
]
