import torch
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from config.emira_config import EMIRAConfig


def normalize_masked_data(data, mask, att_min, att_max):
    scale = att_max - att_min
    scale = scale + (scale == 0) * 1e-8

    if (scale != 0.0).all():
        data_norm = (data - att_min) / scale
    else:
        raise Exception("Zero!")

    data_norm[mask == 0] = 0

    if torch.isnan(data_norm).any():
        raise Exception("nans!")

    return data_norm


def normalize_masked_tp(data, att_min, att_max):
    scale = att_max - att_min
    scale = scale + (scale == 0) * 1e-8

    if (scale != 0.0).all():
        data_norm = (data - att_min) / scale
    else:
        raise Exception("Zero!")

    if torch.isnan(data_norm).any():
        raise Exception("nans!")

    return data_norm


def variable_time_collate_fn(batch, args, dmin, dmax, tmax):
    """Split records at the history boundary and normalize masked values."""

    observed_tp = []
    observed_data = []
    observed_mask = []
    predicted_tp = []
    predicted_data = []
    predicted_mask = []

    for b, (record_id, tt, vals, mask) in enumerate(batch):
        n_observed_tp = torch.lt(tt, args.history).sum()
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

    observed_data = normalize_masked_data(
        observed_data, observed_mask, att_min=dmin, att_max=dmax
    )
    predicted_data = normalize_masked_data(
        predicted_data, predicted_mask, att_min=dmin, att_max=dmax
    )

    observed_tp = normalize_masked_tp(observed_tp, att_min=0, att_max=tmax)
    predicted_tp = normalize_masked_tp(predicted_tp, att_min=0, att_max=tmax)

    return (
        observed_data,
        observed_tp,
        observed_mask,
        predicted_data,
        predicted_tp,
        predicted_mask,
    )


def get_data_min_max(records):
    inf = torch.Tensor([float("Inf")])[0]

    data_min, data_max, time_max = None, None, -inf

    for b, (record_id, tt, vals, mask) in enumerate(records):
        n_features = vals.size(-1)

        batch_min = []
        batch_max = []
        for i in range(n_features):
            non_missing_vals = vals[:, i][mask[:, i] == 1]
            if len(non_missing_vals) == 0:
                batch_min.append(inf)
                batch_max.append(-inf)
            else:
                batch_min.append(torch.min(non_missing_vals))
                batch_max.append(torch.max(non_missing_vals))

        batch_min = torch.stack(batch_min)
        batch_max = torch.stack(batch_max)

        if (data_min is None) and (data_max is None):
            data_min = batch_min
            data_max = batch_max
        else:
            data_min = torch.min(data_min, batch_min)
            data_max = torch.max(data_max, batch_max)

        time_max = torch.max(time_max, tt.max())

    return data_min, data_max, time_max


def split_dataset(config: EMIRAConfig, total_dataset):
    seen_data, test_data = train_test_split(
        total_dataset, train_size=0.8, random_state=42, shuffle=True
    )
    train_data, val_data = train_test_split(
        seen_data, train_size=0.75, random_state=42, shuffle=False
    )
    config.num_chn = train_data[0][2].size(-1)
    dmin, dmax, tmax = get_data_min_max(seen_data)
    return test_data, train_data, val_data, dmin, dmax, tmax
