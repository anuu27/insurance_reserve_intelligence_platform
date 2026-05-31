"""Dataloader assembly helpers."""

from __future__ import annotations

from torch.utils.data import DataLoader

from insurance_reserve_intelligence_platform.data.dataset import ReserveDataset


def create_dataloader(
    dataset: ReserveDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader:
    """Construct a standard PyTorch dataloader."""

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=False)
