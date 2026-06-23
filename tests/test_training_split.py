from __future__ import annotations

from torch.utils.data import Dataset

from scripts.train import split_dataset_by_group


class GroupedDataset(Dataset):
    def __init__(self) -> None:
        self.group_ids = ["a", "a", "b", "b", "c", "c", "d", "d"]

    def __len__(self) -> int:
        return len(self.group_ids)

    def __getitem__(self, index: int) -> int:
        return index


def test_scene_disjoint_split_has_no_group_leakage() -> None:
    dataset = GroupedDataset()
    train, val = split_dataset_by_group(dataset, seed=42, val_fraction=0.25)
    train_groups = {dataset.group_ids[index] for index in train.indices}
    val_groups = {dataset.group_ids[index] for index in val.indices}
    assert train_groups
    assert val_groups
    assert train_groups.isdisjoint(val_groups)
