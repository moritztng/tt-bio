"""Base classes for AtomWorks molecular datasets."""

import logging
import os
import socket
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from tt_bio._vendor.atomworks.ml.transforms.base import TransformedDict
from tt_bio._vendor.atomworks.ml.utils.debug import save_failed_example_to_disk
from tt_bio._vendor.atomworks.ml.utils.rng import capture_rng_states

logger = logging.getLogger("datasets")


class ExampleIDMixin(ABC):
    """Mixin providing example ID functionality to a Dataset.

    Provides methods for converting between example IDs and indices, and checking
    if an example ID exists in the dataset.
    """

    @abstractmethod
    def __contains__(self, example_id: str) -> bool:
        """Check if the dataset contains the example ID.

        Args:
            example_id: The ID to check for.

        Returns:
            True if the ID exists in the dataset.
        """
        pass

    @abstractmethod
    def id_to_idx(self, example_id: str | list[str]) -> int | list[int]:
        """Convert example ID(s) to index(es).

        Args:
            example_id: Single ID or list of IDs to convert.

        Returns:
            Corresponding index or list of indices.
        """
        pass

    @abstractmethod
    def idx_to_id(self, idx: int | list[int]) -> str | list[str]:
        """Convert index(es) to example ID(s).

        Args:
            idx: Single index or list of indices to convert.

        Returns:
            Corresponding ID or list of IDs.
        """
        pass


class MolecularDataset(Dataset):
    """Base class for AtomWorks molecular datasets.

    Handles Transform pipelines and loader functionality for molecular data.
    Subclasses implement :meth:`__getitem__` with their own data access patterns.
    """

    def __init__(
        self,
        *,
        name: str,
        transform: Callable | None = None,
        loader: Callable | None = None,
        save_failed_examples_to_dir: str | Path | None = None,
    ):
        """Initialize MolecularDataset.

        Args:
            name: Descriptive name for this dataset. Used for debugging and some
                downstream functions when using nested datasets.
            transform: Transform function or pipeline to apply to loaded data.
                Should accept the output of the loader and return featurized data.
            loader: Optional function to process raw dataset output into Transform-ready
                format. For example, parsing structural files or gathering columns
                into structured data.
            save_failed_examples_to_dir: Optional directory path where failed examples
                will be saved for debugging. Includes RNG state and error information.
        """
        self.loader = loader

        self.transform = transform
        self.name = name
        self.save_failed_examples_to_dir = Path(save_failed_examples_to_dir) if save_failed_examples_to_dir else None

    def _apply_loader(self, raw_data: Any) -> Any:
        """Apply the loader function to raw data with timing and debugging.

        Args:
            raw_data: The raw data to process.

        Returns:
            Processed data ready for transforms.
        """
        if self.loader is None:
            return raw_data

        # Apply loader function with timing
        _start_load_time = time.time()
        data = self.loader(raw_data)
        _stop_load_time = time.time()

        # Add timing information if data supports it (preserving TransformDataset behavior)
        if isinstance(data, dict):
            data = TransformedDict(data)
            data.__transform_history__.append(
                {
                    "name": "apply loader",
                    "instance": hex(id(self.loader)),
                    "start_time": _start_load_time,
                    "end_time": _stop_load_time,
                    "processing_time": _stop_load_time - _start_load_time,
                }
            )

        return data

    def _apply_transform(self, data: Any, example_id: str | None = None, idx: int | None = None) -> Any:
        """Apply the Transform pipeline with error handling and debugging support.

        Args:
            data: The loaded data ready for transforms.
            example_id: Optional example ID for debugging purposes. If not provided,
                will generate one using dataset name and index.
            idx: Optional dataset index for error reporting.

        Returns:
            Transformed data.

        Raises:
            KeyboardInterrupt: Always re-raised if encountered.
            Exception: Any exception from the transform pipeline is re-raised.
        """
        if self.transform is None:
            return data

        # Generate default example_id from idx and dataset name if not provided
        if example_id is None and idx is not None:
            example_id = f"{self.name}_{idx}"

        # Get process id and hostname for debugging
        if example_id:
            logger.debug(f"({socket.gethostname()}:{os.getpid()}) Processing example: {example_id}")

        try:
            # Capture RNG state for reproducibility before applying Transforms
            rng_state_dict = capture_rng_states(include_cuda=False)
            data = self.transform(data)
            return data

        except KeyboardInterrupt:
            # Always re-raise keyboard interrupts
            raise
        except Exception as e:
            logger.error(e)

            if self.save_failed_examples_to_dir and example_id:
                save_failed_example_to_disk(
                    example_id=example_id,
                    error_msg=e,
                    rng_state_dict=rng_state_dict,
                    data={},  # We do not save the data by default, since it may be large
                    fail_dir=self.save_failed_examples_to_dir,
                )

            # Re-raise the original exception
            raise

    def __getitem__(self, index: int) -> Any:
        """Return a fully-featurized data example given an index.

        Subclasses should implement this method to:
            1. Query the underlying data source for raw data at the given index
            2. Optionally pre-process data to prepare for the Transform pipeline
            3. Feed the input dictionary through a Transform pipeline

        Typical output for an activity prediction network:
            Step 1: ``{"path": "/path/to/dataset", "class_label": 5}``
            Step 2: ``{"atom_array": AtomArray, "extra_info": {"class_label": 5}}``
            Step 3: ``{"features": torch.Tensor, "class_label": torch.Tensor}``

        Args:
            index: The index of the example to retrieve.

        Returns:
            Fully-featurized data example.
        """
        raise NotImplementedError

    def __len__(self) -> int:
        """Return the number of examples in the dataset.

        Returns:
            The dataset length.
        """
        raise NotImplementedError
