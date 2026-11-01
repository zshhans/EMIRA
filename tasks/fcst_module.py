from dataclasses import asdict

import torch
from torch import Tensor
from torch.nn.functional import mse_loss
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
from torchmetrics import Metric, MetricCollection


def mae_loss(pred, tgt, **kwargs):
    return torch.abs(pred - tgt)


class Error(Metric):
    is_differentiable = True
    higher_is_better = False
    full_state_update = False
    error: Tensor
    total: Tensor

    def __init__(self, num_chn, fn, **kwargs):
        super().__init__(**kwargs)
        self.add_state("error", default=torch.zeros(num_chn), dist_reduce_fx="sum")
        self.add_state("total", default=torch.zeros(num_chn), dist_reduce_fx="sum")
        self.fn = fn

    def update(self, pred: Tensor, tgt: Tensor, mask: Tensor) -> None:
        error = (self.fn(pred, tgt, reduction="none") * mask).sum([0, 1])
        count = mask.sum([0, 1])

        self.error += error
        self.total += count

    def compute(self) -> Tensor:
        num_chn = self.total.count_nonzero()
        error = self.error / (self.total + 1e-8) / (num_chn + 1e-8)
        return error.sum()


class ForecastingModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(asdict(config))
        self.model = config.get_model()
        self.forward_fn = config.forward_fn
        metrics = MetricCollection(
            {
                "mse": Error(config.num_chn, mse_loss),
                "mae": Error(config.num_chn, mae_loss),
            }
        )
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

    def training_step(self, batch, batch_idx):
        pred, target, mask = self.forward_fn(self.model, batch)
        error = (mse_loss(pred, target, reduction="none") * mask).sum([0, 1])
        count = mask.sum([0, 1])
        return (error / (count + 1e-8) / (count.count_nonzero() + 1e-8)).sum()

    def validation_step(self, batch, batch_idx):
        pred, target, mask = self.forward_fn(self.model, batch)
        self.val_metrics.update(pred, target, mask)
        self.log_dict(self.val_metrics)

    def on_validation_epoch_end(self):
        self.log("lr", self.optimizers().param_groups[0]["lr"])

    def test_step(self, batch, batch_idx):
        pred, target, mask = self.forward_fn(self.model, batch)
        self.test_metrics.update(pred, target, mask)
        self.log_dict(self.test_metrics)

    def configure_optimizers(self):
        shared, gates = [], []
        for name, parameter in self.named_parameters():
            (gates if ".fusion_gate." in name else shared).append(parameter)
        optimizer = torch.optim.AdamW(
            [
                dict(params=shared, weight_decay=self.hparams.weight_decay),
                dict(params=gates, weight_decay=self.hparams.fusion_gate_weight_decay),
            ],
            lr=self.hparams.lr,
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.hparams.lr_factor,
            patience=3,
            min_lr=3e-5,
        )
        return dict(
            optimizer=optimizer,
            lr_scheduler=dict(scheduler=scheduler, monitor="val_mse"),
        )
