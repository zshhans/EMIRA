from pathlib import Path

import torch
from torch.utils.data import Dataset

FILES = {
    "physionet": ("set-a_None.pt", "set-b_None.pt", "set-c_None.pt"),
    "mimic": ("mimic.pt",),
    "activity": ("data.pt",),
    "ushcn": ("ushcn.pt",),
}


class ProcessedDataset(Dataset):
    """Load the preprocessed t-PatchGNN records without altering their order."""

    def __init__(self, dataset, root):
        paths = [Path(root) / "processed" / name for name in FILES[dataset]]
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing preprocessed data: {path}. See README.md."
                )
        self.data = []
        for path in paths:
            self.data.extend(torch.load(path, map_location="cpu", weights_only=False))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
