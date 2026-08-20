"""Compression utilities for file operations."""

__all__ = ["compress_file", "is_compressed_file", "maybe_decompress_file", "transfer_with_compression"]

import gzip
import shutil
from os import PathLike
from pathlib import Path

import zstandard as zstd


def is_compressed_file(path: PathLike | str) -> bool:
    """Check if a file is compressed based on its extension."""
    path_str = str(path)
    return path_str.endswith(".gz") or path_str.endswith(".gzip") or path_str.endswith(".zst")


def compress_file(
    input_file: PathLike | str, output_file: PathLike | str | None = None, remove_original: bool = False
) -> Path:
    """Compress a file using gzip or zstd.

    Args:
        input_file: Path to the input file to compress.
        output_file: Path to the output compressed file. If None, uses input_file path + ".gz".
        remove_original: Whether to remove the original file after compression.

    Returns:
        Path to the compressed output file.

    Raises:
        OSError: If files cannot be read/written.
        FileNotFoundError: If input file doesn't exist.

    Examples:
        >>> # Compress a file, keeping the original
        >>> compress_file("data.txt")
        PosixPath('data.txt.gz')

        >>> # Compress with custom output path, removing original
        >>> compress_file("data.txt", "compressed/data.txt.gz", remove_original=True)
        PosixPath('compressed/data.txt.gz')

        >>> # Compress with zstd
        >>> compress_file("data.txt", "compressed/data.txt.zst")
        PosixPath('compressed/data.txt.zst')
    """
    input_path = Path(input_file)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Set default output path if not specified
    if output_file is None:
        output_path = input_path.with_suffix(f"{input_path.suffix}.gz")
    else:
        output_path = Path(output_file)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine compression format based on output extension
    if str(output_path).endswith(".zst"):
        # Use zstd compression
        cctx = zstd.ZstdCompressor()
        with open(input_path, "rb") as f_in, open(output_path, "wb") as f_out:
            cctx.copy_stream(f_in, f_out)
    else:
        # Use gzip compression (default)
        with open(input_path, "rb") as f_in, gzip.open(output_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    # Remove original file if requested
    if remove_original:
        input_path.unlink()

    return output_path


def maybe_decompress_file(
    input_file: PathLike | str, output_file: PathLike | str | None = None, remove_original: bool = False
) -> Path:
    """Decompress a file only if it's compressed, otherwise return the original path.

    Args:
        input_file: Path to the input file (may or may not be compressed).
        output_file: Path to the output decompressed file (if None, auto-generates).
        remove_original: Whether to remove the original file after decompression.

    Returns:
        Path to the uncompressed file (either the original or the decompressed one).

    Raises:
        OSError: If files cannot be read/written.
        FileNotFoundError: If input file doesn't exist.
    """
    input_path = Path(input_file)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # If not compressed, just return the input path - no operation needed
    if not is_compressed_file(input_path):
        return input_path

    # Handle decompression
    if output_file is None:
        # Handle .ext.gz -> .ext or .ext.zst -> .ext
        if input_path.suffix in (".gz", ".zst"):
            output_path = input_path.with_suffix("")
        else:
            # Handle .gzip
            output_path = Path(str(input_path).replace(".gzip", ""))
    else:
        output_path = Path(output_file)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Decompress the file based on format
    if str(input_path).endswith(".zst"):
        # Use zstd decompression
        dctx = zstd.ZstdDecompressor()
        with open(input_path, "rb") as f_in, open(output_path, "wb") as f_out:
            dctx.copy_stream(f_in, f_out)
    else:
        # Use gzip decompression
        with gzip.open(input_path, "rb") as f_in, open(output_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    # Remove original if requested
    if remove_original:
        input_path.unlink()

    return output_path


def transfer_with_compression(input_file: PathLike | str, output_file: PathLike | str, move: bool = False) -> Path:
    """Copy or move a file with automatic compression/decompression based on file extensions.

    Args:
        input_file: Path to the source file
        output_file: Path to the destination file
        move: If True, removes the input file after operation (default: False)

    Returns:
        Path to the output file

    Raises:
        FileNotFoundError: If input file doesn't exist
        OSError: If files cannot be read/written
    """
    input_path = Path(input_file)
    output_path = Path(output_file)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Case 1: Input is compressed, output is not compressed -> decompress
    if is_compressed_file(input_path) and not is_compressed_file(output_path):
        return maybe_decompress_file(input_path, output_path, remove_original=move)

    # Case 2: Input is not compressed, output is compressed -> compress
    elif not is_compressed_file(input_path) and is_compressed_file(output_path):
        return compress_file(input_path, output_path, remove_original=move)

    # Case 3: Both compressed or both uncompressed -> simple copy/move
    else:
        if move:
            shutil.move(input_path, output_path)
        else:
            shutil.copy2(input_path, output_path)
        return output_path
