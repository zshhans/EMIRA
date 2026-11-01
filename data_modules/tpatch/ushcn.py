import torch
from torch.nn.utils.rnn import pad_sequence
from . import utils


def USHCN_time_chunk(data, args):
    chunk_data = []

    for b, (record_id, tt, vals, mask) in enumerate(data):
        for st in range(
            0, args.n_months - args.history - args.pred_window + 1, args.pred_window
        ):
            et = st + args.history + args.pred_window
            if et == args.n_months:
                indices = torch.where((tt >= st) & (tt <= et))[0]
            else:
                indices = torch.where((tt >= st) & (tt < et))[0]

            t_bias = torch.tensor(st).to(tt.device)
            chunk_data.append(
                (record_id, tt[indices] - t_bias, vals[indices], mask[indices], t_bias)
            )

    return chunk_data


def USHCN_variable_time_collate_fn(batch, args, dmin=None, dmax=None, tmax=None):
    """Collate monthly windows while restoring absolute timestamps."""

    observed_tp = []
    observed_data = []
    observed_mask = []
    predicted_tp = []
    predicted_data = []
    predicted_mask = []

    for b, (record_id, tt, vals, mask, t_bias) in enumerate(batch):
        n_observed_tp = torch.lt(tt, args.history).sum()
        tt = tt + t_bias
        observed_tp.append(tt[:n_observed_tp])
        observed_data.append(vals[:n_observed_tp])
        observed_mask.append(mask[:n_observed_tp])

        predicted_tp.append(tt[n_observed_tp:])
        predicted_data.append(vals[n_observed_tp:])
        predicted_mask.append(mask[n_observed_tp:])

    observed_tp = pad_sequence(observed_tp, batch_first=True)
    observed_data = pad_sequence(observed_data, batch_first=True)
    observed_mask = pad_sequence(observed_mask, batch_first=True)
    predicted_tp = pad_sequence(predicted_tp, batch_first=True)
    predicted_data = pad_sequence(predicted_data, batch_first=True)
    predicted_mask = pad_sequence(predicted_mask, batch_first=True)

    observed_tp = utils.normalize_masked_tp(observed_tp, att_min=0, att_max=tmax)
    predicted_tp = utils.normalize_masked_tp(predicted_tp, att_min=0, att_max=tmax)

    return (
        observed_data,
        observed_tp,
        observed_mask,
        predicted_data,
        predicted_tp,
        predicted_mask,
    )
