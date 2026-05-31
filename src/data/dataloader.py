"""Dataloader assembly helpers.

Created: 2026-05-31
Purpose: Build PyTorch dataloaders for reserve-learning datasets.
"""

from __future__ import annotations

from torch.utils.data import DataLoader

from src.data.dataset import ReserveDataset


def create_dataloader(
    dataset: ReserveDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader:
    """Construct a standard PyTorch dataloader.

    Args:
        dataset: Dataset to expose through the dataloader.
        batch_size: Batch size for iteration.
        shuffle: Whether to shuffle examples.
        num_workers: Number of worker processes for loading.

    Returns:
        DataLoader: Configured dataloader instance.
    """

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=False)
