from typing import List, Tuple

import torch
from torch import Tensor
from torch.nn.functional import pad


def pad_dim(data: Tensor, dim: int, pad_len: float, value: float = 0) -> Tensor:
    """Pad a tensor along a given dimension.

    Parameters
    ----------
    data : Tensor
        The input tensor.
    dim : int
        The dimension to pad.
    pad_len : float
        The padding length.
    value : int, optional
        The value to pad with.

    Returns
    -------
    Tensor
        The padded tensor.

    """
    if pad_len == 0:
        return data

    total_dims = len(data.shape)
    padding = [0] * (2 * (total_dims - dim))
    padding[2 * (total_dims - 1 - dim) + 1] = pad_len
    return pad(data, tuple(padding), value=value)


def pad_to_max(data: List[Tensor], value: float = 0) -> Tuple[Tensor, Tensor]:
    """Pad the data in all dimensions to the maximum found.

    Parameters
    ----------
    data : List[Tensor]
        List of tensors to pad.
    value : float
        The value to use for padding.

    Returns
    -------
    Tensor
        The padded tensor.
    Tensor
        The padding mask.

    """
    if isinstance(data[0], str):
        return data, 0

    # Check if all have the same shape - early return
    first_shape = data[0].shape
    if all(d.shape == first_shape for d in data):
        return torch.stack(data, dim=0), 0

    # Get the maximum in each dimension - vectorized approach
    num_dims = len(first_shape)
    max_dims = []

    # Vectorized max calculation
    for i in range(num_dims):
        max_dims.append(max(d.shape[i] for d in data))

    # Pre-allocate output tensors for better memory efficiency
    batch_size = len(data)
    target_shape = [batch_size] + max_dims

    # Create output tensor directly with target shape
    padded_data = torch.full(
        target_shape, value, dtype=data[0].dtype, device=data[0].device
    )
    padding_masks = torch.zeros(target_shape, dtype=torch.bool, device=data[0].device)

    # Fill in data and masks efficiently using slice assignment
    for i, d in enumerate(data):
        # Create slice tuples for current tensor
        slices = [slice(None)] * (num_dims + 1)
        slices[0] = i  # batch dimension
        for j in range(num_dims):
            slices[j + 1] = slice(0, d.shape[j])

        # Assign data and mask in one operation
        padded_data[tuple(slices)] = d
        padding_masks[tuple(slices)] = True

    return padded_data, padding_masks
