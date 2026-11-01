import torch


@torch.no_grad()
def token_predecessors(x_mask, y_mask, coords, padding_mask):
    """Previous valid observation OR query in the existing time-major packing.

    All dense intermediates are integer metadata [B,T,C]. Values never decide
    validity. Each source has at most one successor, including q1 -> q2.
    """
    b, t, c = x_mask.shape
    if y_mask.shape != x_mask.shape or padding_mask.shape != coords.shape[:2]:
        raise ValueError("masks and packed coordinates must have matching shapes")
    valid = (x_mask > 0) | (y_mask > 0)
    ranks = valid.flatten(1).long().cumsum(1).sub(1).view(b, t, c)
    positions = torch.where(valid, ranks, -1)
    last = positions.cummax(dim=1).values
    previous = torch.cat([torch.full_like(last[:, :1], -1), last[:, :-1]], dim=1)
    tc = coords[..., 1:].long()
    flat = tc[..., 0] * c + tc[..., 1]
    source = previous.flatten(1).gather(1, flat)
    receiver = valid.flatten(1).gather(1, flat) & padding_mask
    return torch.where(receiver, source, -1).to(torch.int32).contiguous()
