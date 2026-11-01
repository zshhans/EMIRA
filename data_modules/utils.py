from functools import partial

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import pytorch_lightning as pl

from .tpatch.processed import ProcessedDataset
from .tpatch.utils import split_dataset, variable_time_collate_fn
from .tpatch.activity import Activity_time_chunk
from .tpatch.ushcn import USHCN_time_chunk, USHCN_variable_time_collate_fn
from .paired_sampling import paired_loaders


def get_dataloaders(config, num_workers=8, sampler_path=None):
    pl.seed_everything(2026)
    dataset = ProcessedDataset(config.dataset, config.datapath)
    test, train, val, dmin, dmax, tmax = split_dataset(config, dataset)
    if config.dataset == "ushcn":
        train, val, test = [
            USHCN_time_chunk(part, config) for part in (train, val, test)
        ]
        collate = partial(USHCN_variable_time_collate_fn, args=config, tmax=tmax)
    else:
        if config.dataset == "activity":
            train, val, test = [
                Activity_time_chunk(part, config) for part in (train, val, test)
            ]
            tmax = torch.tensor(config.history + config.pred_window)
        collate = partial(
            variable_time_collate_fn, args=config, dmin=dmin, dmax=dmax, tmax=tmax
        )
    loaders = [
        DataLoader(
            part,
            batch_size=config.batch_size,
            shuffle=(index == 0),
            num_workers=num_workers,
            collate_fn=partial(compact_fn, fn=collate),
        )
        for index, part in enumerate((train, val, test))
    ]
    return paired_loaders(loaders, config.seed, sampler_path)


def compact_fn(batch, fn):
    x, xt, x_mask, y, yt, y_mask = fn(batch)
    bsz = x.shape[0]
    x = torch.cat([x, x_mask], dim=-1)
    y = torch.cat([y, y_mask], dim=-1)
    xt_mask = x_mask.any(2)
    yt_mask = y_mask.any(2)
    x_list = []
    y_list = []
    t_list = []
    for i in range(bsz):
        x_list.append(
            torch.cat([x[i, xt_mask[i]], torch.zeros_like(y[i, yt_mask[i]])], dim=0)
        )
        y_list.append(
            torch.cat([torch.zeros_like(x[i, xt_mask[i]]), y[i, yt_mask[i]]], dim=0)
        )
        t_list.append(torch.cat([xt[i, xt_mask[i]], yt[i, yt_mask[i]]], dim=0))
    x = pad_sequence(x_list, batch_first=True)
    y = pad_sequence(y_list, batch_first=True)
    t = pad_sequence(t_list, batch_first=True)
    x, x_mask = torch.chunk(x, 2, dim=-1)
    y, y_mask = torch.chunk(y, 2, dim=-1)
    if t.dim() == 2:
        t = t.unsqueeze(-1)

    return x, y, t, x_mask, y_mask
