"""Independently seeded training permutations with recorded sample order."""
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Sampler


class RecordedPermutationSampler(Sampler):
    def __init__(self, dataset, seed, path=None):
        self.size = len(dataset)
        self.seed = seed
        self.path = Path(path) if path is not None else None
        self.iteration = 0
        self.records = []

    def __len__(self):
        return self.size

    def __iter__(self):
        index = self.iteration
        self.iteration += 1
        order = torch.randperm(
            self.size,
            generator=torch.Generator().manual_seed(self.seed + 1000003 * index),
        )
        count = 0
        try:
            for value in order.tolist():
                count += 1
                yield value
        finally:
            record = dict(
                iterator_index=index,
                permutation_seed=self.seed + 1000003 * index,
                planned_length=self.size,
                yielded=count,
                first_16=order[:16].tolist(),
                yielded_sha256=hashlib.sha256(
                    order[:count].numpy().tobytes()
                ).hexdigest(),
            )
            self.records.append(record)
            if self.path is not None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a") as stream:
                    stream.write(json.dumps(record) + "\n")


def paired_loaders(original, seed, sampler_path=None):
    loaders = []
    for split, old in enumerate(original):
        sampler = (
            RecordedPermutationSampler(old.dataset, seed, sampler_path)
            if split == 0
            else None
        )
        loaders.append(
            DataLoader(
                old.dataset,
                batch_size=old.batch_size,
                sampler=sampler,
                shuffle=False,
                num_workers=old.num_workers,
                collate_fn=old.collate_fn,
                generator=torch.Generator().manual_seed(seed + 1000 + split),
            )
        )
    return loaders
