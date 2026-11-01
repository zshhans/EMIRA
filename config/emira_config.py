from dataclasses import asdict, dataclass
import json
from pathlib import Path

DATASETS = ("physionet", "mimic", "activity", "ushcn")
DATA_FOLDERS = dict(zip(DATASETS, ("physionet", "mimic3", "activity", "ushcn")))


@dataclass
class EMIRAConfig:
    dataset: str
    datapath: str
    num_chn: int
    history: int
    n_layers1: int
    n_layers2: int
    d_state1: int
    dt_rank1: int
    lr: float
    lr_factor: float
    weight_decay: float
    fusion_gate_weight_decay: float
    drop: float
    fusion_init: float
    grad_clip_val: float
    share_head: bool
    seed: int = 30
    deterministic: bool = False
    batch_size: int = 32
    max_epochs: int = 50
    dim_t: int = 1
    d_model1: int = 32
    d_model2: int = 64
    expand1: int = 4
    expand2: int = 4
    d_state2: int = 1
    dt_rank2: int = 16
    n_months: int = 48
    pred_window: int = 1

    @staticmethod
    def forward_fn(model, batch):
        x, y, t, x_mask, y_mask = batch
        return model(x, t, x_mask, y_mask), y, y_mask

    def get_model(self):
        from models.emira import Model

        return Model(**asdict(self))


def dataset_config(dataset, data_root="dataset", seed=30, deterministic=False):
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}")
    profiles = json.loads(Path(__file__).with_name("datasets.json").read_text())
    values = profiles[dataset]
    return EMIRAConfig(
        dataset=dataset,
        datapath=str(Path(data_root) / DATA_FOLDERS[dataset]),
        seed=seed,
        deterministic=deterministic,
        pred_window=1000 if dataset == "activity" else 1,
        **values,
    )
