import torch
from torch.nn.utils.rnn import pad_sequence
from . import utils


def Activity_time_chunk(data, args):
    chunk_data = []
    history = args.history
    pred_window = args.pred_window
    for b, (record_id, tt, vals, mask) in enumerate(data):
        t_max = int(tt.max())
        for st in range(0, t_max - history, pred_window):
            et = st + history + pred_window
            if et >= t_max:
                idx = torch.where((tt >= st) & (tt <= et))[0]
            else:
                idx = torch.where((tt >= st) & (tt < et))[0]
            new_id = f"{record_id}_{st//pred_window}"
            chunk_data.append((new_id, tt[idx] - st, vals[idx], mask[idx]))

    return chunk_data
