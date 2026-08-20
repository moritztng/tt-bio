"""Pandas DataFrame-based dataset implementation."""

import logging
import warnings
from collections.abc import Callable
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tt_bio._vendor.atomworks.common import as_list
from tt_bio._vendor.atomworks.constants import NA_VALUES
from tt_bio._vendor.atomworks.ml.utils.io import read_parquet_with_metadata

from .base import ExampleIDMixin, MolecularDataset

logger = logging.getLogger("datasets")


class PandasDataset(MolecularDataset, ExampleIDMixin):
    """Dataset for tabular data stored as pandas DataFrames.

    Inherits all functionality from :class:`MolecularDataset` with additional
    DataFrame-specific features for filtering and ID-based access.
    """

    def __init__(
        self,
        *,
        data: pd.DataFrame | PathLike,
        name: str,
        id_column: str | None = "example_id",
        filters: list[str] | None = None,
        columns_to_load: list[str] | None = None,
        # MolecularDataset parameters
        transform: Callable | None = None,
        loader: Callable | None = None,
        save_failed_examples_to_dir: str | Path | None = None,
        load_kwargs: dict | tuple | None = None,
    ):
        """Initialize PandasDataset.

        Args:
            data: Either a pandas DataFrame or path to a CSV/Parquet file containing
                the tabular data. Each row represents one example.
            name: Descriptive name for this dataset. Used for debugging and some
                downstream functions when using nested datasets.
            id_column: Optional column name to use as the DataFrame index for
                example ID lookups. If provided, this column will be set as the index.
            filters: Optional list of pandas query strings to filter the data.
                Applied in order during initialization.
            columns_to_load: Optional list of column names to load when reading
                from a file. If None, all columns are loaded. Can dramatically reduce
                memory usage and load time if loading from a columnar format like Parquet.
            transform: Transform pipeline to apply to loaded data.
            loader: Optional function to process raw DataFrame rows into Transform-ready format.
            save_failed_examples_to_dir: Optional directory path where failed examples
                will be saved for debugging. Includes RNG state and error information.
            load_kwargs: Additional keyword arguments passed to pandas' read functions
                (read_csv, read_parquet) when loading from file.

        Examples:
            Load from DataFrame:
                >>> df = pd.DataFrame({"path": [...], "label": [...]})
                >>> dataset = PandasDataset(data=df, name="my_dataset")

            Load from file with filtering:
                >>> dataset = PandasDataset(data="data.csv", name="filtered_dataset", filters=["label > 0", "path.str.contains('.pdb')"])
        """
        super().__init__(
            name=name,
            transform=transform,
            loader=loader,
            save_failed_examples_to_dir=save_failed_examples_to_dir,
        )

        # Load data from path if needed
        if isinstance(data, PathLike | str):
            data = self._load_from_path(data, columns_to_load, **(load_kwargs or {}))
        self.data = data

        # Apply filters
        self.filters = filters
        self._already_filtered = False
        if filters:
            self._apply_filters(filters)
        self._already_filtered = True

        # Set index column if specified
        if id_column is not None:
            assert id_column in self.data.columns, f"Column {id_column} not found in dataset."
            self.data.set_index(id_column, inplace=True, drop=False, verify_integrity=True)

    def _load_from_path(
        self, path: PathLike | str, columns_to_load: list[str] | None = None, **load_kwargs: Any
    ) -> pd.DataFrame:
        """Load data from file path.

        Args:
            path: Path to the file to load.
            columns_to_load: Optional list of column names to load.
            **load_kwargs: Additional arguments for pandas read functions.

        Returns:
            Loaded DataFrame.

        Raises:
            ValueError: If file type is unsupported.
        """
        path = Path(path)
        # Convert OmegaConf ListConfig to plain list if needed
        if columns_to_load is not None:
            columns_to_load = as_list(columns_to_load)
        if path.suffix == ".csv":
            data = pd.read_csv(path, usecols=columns_to_load, keep_default_na=False, na_values=NA_VALUES, **load_kwargs)
        elif path.suffix == ".parquet":
            data = read_parquet_with_metadata(path, columns=columns_to_load, **load_kwargs)
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")
        return data

    def __getitem__(self, idx: int) -> Any:
        """Get an example by index, applying specified loader and Transforms.

        Args:
            idx: The index of the example to retrieve.

        Returns:
            Transformed data from the row.
        """
        raw_data = self.data.iloc[idx]
        example_id = self._get_example_id(idx)
        data = self._apply_loader(raw_data)
        return self._apply_transform(data, example_id=example_id, idx=idx)

    def __len__(self) -> int:
        """Return the number of rows in the dataset."""
        return len(self.data)

    def __contains__(self, example_id: str) -> bool:
        """Check if the dataset contains the example ID."""
        return example_id in self.data.index

    def _id_to_index_single(self, example_id: str) -> int:
        """Convert single example ID to index."""
        return self.data.index.get_loc(example_id)

    def _id_to_index_multiple(self, example_ids: list[str]) -> list[int]:
        """Convert multiple example IDs to indices."""
        idxs = np.arange(len(self.data))
        return [idxs[self.data.index.get_loc(example_id)] for example_id in example_ids]

    def id_to_idx(self, example_id: str | list[str]) -> int | list[int]:
        """Convert an example ID to the corresponding local index."""
        if np.isscalar(example_id):
            return self._id_to_index_single(example_id)
        elif isinstance(example_id, list | np.ndarray | tuple):
            return self._id_to_index_multiple(example_id)
        else:
            raise ValueError(f"Invalid type for example_id: {type(example_id)}")

    def idx_to_id(self, idx: int | list[int]) -> str | np.ndarray:
        """Convert a local index to the corresponding example ID."""
        _return_single = False
        if np.isscalar(idx) or (isinstance(idx, np.ndarray) and idx.shape == ()):
            _return_single = True
            idx = idx.item() if isinstance(idx, np.ndarray) else idx
            idx = slice(idx, idx + 1)
        ids = self.data.iloc[idx].index.values
        return ids[0] if _return_single else ids

    def _apply_filters(self, filters: list[str]) -> pd.DataFrame:
        """Apply filters to the data based on the provided list of query strings.

        For documentation on pandas query syntax, see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.query.html

        Args:
            filters: List of query strings to apply to the data.

        Raises:
            ValueError: If the data is not initialized or if a query removes all rows.
            Warning: If a query does not remove any rows.

        Example:
            >>> queries = [
            >>>     "deposition_date < '2020-01-01'",
            >>>     "resolution < 2.5 and ~method.str.contains('NMR')",
            >>>     "cluster.notnull()",
            >>>     "method in ['X-RAY_DIFFRACTION', 'ELECTRON_MICROSCOPY']"
            >>> ]
            >>> dataset = PandasDataset(data="data.csv", name="filtered_dataset", filters=queries)
        """
        assert not self._already_filtered, "Filters cannot be applied after initialization."

        # Apply queries one by one, confirming the impact of each
        for query in filters:
            self._apply_query(query)

    def _apply_query(self, query: str) -> None:
        """Apply a single query to the data.

        Args:
            query: The pandas query string to apply.
        """
        # Filter using query and validate impact
        original_num_rows = len(self.data)
        self.data = self.data.query(query)
        filtered_num_rows = len(self.data)
        self._validate_filter_impact(query, original_num_rows, filtered_num_rows)

    def _validate_filter_impact(self, query: str, original_num_rows: int, filtered_num_rows: int) -> None:
        """Validate the impact of the filter.

        Args:
            query: The query that was applied.
            original_num_rows: Number of rows before filtering.
            filtered_num_rows: Number of rows after filtering.

        Raises:
            ValueError: If the query removes all rows.
        """
        rows_removed = original_num_rows - filtered_num_rows
        percent_removed = (rows_removed / original_num_rows) * 100
        percent_remaining = (filtered_num_rows / original_num_rows) * 100

        if filtered_num_rows == original_num_rows:
            logger.warning(f"Query '{query}' on dataset {self.name} did not remove any rows.")
        elif filtered_num_rows == 0:
            raise ValueError(f"Query '{query}' on dataset {self.name} removed all rows.")
        else:
            logger.info(
                f"\n+-------------------------------------------+\n"
                f"Query '{query}' on dataset {self.name}:\n"
                f"  - Started with: {original_num_rows:,} rows\n"
                f"  - Removed: {rows_removed:,} rows ({percent_removed:.2f}%)\n"
                f"  - Remaining: {filtered_num_rows:,} rows ({percent_remaining:.2f}%)\n"
                f"+-------------------------------------------+\n"
            )

    def _get_example_id(self, idx: int) -> str:
        """Get example ID from index - returns the index value from the DataFrame.

        Args:
            idx: The index of the row.

        Returns:
            The index value as a string.
        """
        return str(self.data.iloc[idx].name)  # .name gets the index value


# Backwards Compatibility
# TODO: Deprecate
def StructuralDatasetWrapper(  # noqa: N802
    dataset_parser: Callable,
    transform: Callable | None = None,
    dataset: PandasDataset | None = None,
    cif_parser_args: dict | None = None,
    save_failed_examples_to_dir: str | Path | None = None,
    **kwargs,
) -> PandasDataset:
    """Backwards-compatible wrapper for the deprecated StructuralDatasetWrapper.

    This function is deprecated and will be removed in a future version.
    Use :class:`PandasDataset` with the appropriate loader function instead.

    Args:
        dataset_parser: The dataset parser to use (e.g., PNUnitsDFParser, InterfacesDFParser).
        transform: Transform pipeline to apply to loaded data.
        dataset: The underlying PandasDataset containing the tabular data.
        cif_parser_args: Arguments to pass to the CIF parser.
        save_failed_examples_to_dir: Directory to save failed examples for debugging.
        **kwargs: Additional arguments passed to PandasDataset.

    Returns:
        PandasDataset instance configured with the deprecated parameters.

    Raises:
        ValueError: If dataset parameter is required but not provided.
    """
    from tt_bio._vendor.atomworks.ml.datasets.parsers import load_example_from_metadata_row

    warnings.warn(
        "StructuralDatasetWrapper is deprecated. Use PandasDataset with a loader function instead. "
        "See atomworks.ml.datasets.loaders for functional alternatives to dataset parsers.",
        DeprecationWarning,
        stacklevel=2,
    )

    if dataset is None:
        raise ValueError("dataset parameter is required for StructuralDatasetWrapper")

    # Create loader from deprecated parameters
    def loader(row: pd.Series) -> dict[str, Any]:
        return load_example_from_metadata_row(row, dataset_parser, cif_parser_args=cif_parser_args or {})

    # Create a new PandasDataset with the loader
    return PandasDataset(
        data=dataset.data,
        name=dataset.name if hasattr(dataset, "name") else "structural_dataset",
        transform=transform,
        loader=loader,
        save_failed_examples_to_dir=save_failed_examples_to_dir,
        **kwargs,
    )
