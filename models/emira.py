import math
import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from einops import repeat
from .emira_mamba import EmiraMamba
from .observation_routes import token_predecessors


class ForecastingHead(nn.Module):
    def __init__(self, d_model, num_chn):
        super().__init__()
        self.weights = nn.Parameter(torch.Tensor(num_chn, d_model, 1))
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        self.bias = nn.Parameter(torch.Tensor(1, 1, num_chn, 1))
        nn.init.zeros_(self.bias)

    def forward(self, x):
        output = torch.einsum("btcd,cdo->btco", x, self.weights) + self.bias
        return output


class Model(nn.Module):
    def __init__(
        self,
        num_chn,
        dim_t,
        n_layers1,
        d_model1,
        d_state1,
        expand1,
        dt_rank1,
        n_layers2,
        d_model2,
        d_state2,
        expand2,
        dt_rank2,
        drop,
        share_head,
        fusion_init,
        **kwargs
    ):
        super().__init__()
        self.d_model1 = d_model1
        self.d_state1 = d_state1
        self.expand1 = expand1
        self.dt_rank1 = dt_rank1
        self.d_model2 = d_model2
        self.d_state2 = d_state2
        self.expand2 = expand2
        self.dt_rank2 = dt_rank2
        self.t_emb = nn.Linear(dim_t, d_model1 * 2, bias=False)
        self.mask_emb = nn.Sequential(
            nn.Linear(num_chn * 2, d_model1, bias=False),
        )
        self.x_emb = nn.Sequential(
            nn.Linear(2 + num_chn, d_model2 * expand2),
            nn.SiLU(),
            nn.Linear(d_model2 * expand2, d_model2),
        )

        self.ctx_proj = nn.Linear(d_model1, d_model2 * expand2, bias=False)
        self.query_proj = nn.Linear(d_model1 * 2, d_state2, bias=False)

        self.mamba1 = EmiraMamba(n_layers1, d_model1, d_state1, expand1, dt_rank1, drop)
        self.mamba2 = EmiraMamba(
            n_layers2,
            d_model2,
            d_state2,
            expand2,
            dt_rank2,
            drop,
            state_fusion=True,
            fusion_init=fusion_init,
        )

        if share_head:
            self.head = nn.Linear(d_model2, 1)
        else:
            self.head = ForecastingHead(d_model2, num_chn)

    def _temporal_encoding(self, t: Tensor):
        t = t - t[:, [0]]
        t = F.silu(self.t_emb(t))
        trend, cyclic = t.chunk(2, dim=-1)
        return trend + torch.sin(cyclic)

    def forward(self, x, t, x_mask, y_mask):
        dev = x.device
        bsz, max_lens, nc = x_mask.shape
        x_lens = x_mask.any(2).sum(1)
        y_lens = y_mask.any(2).sum(1)
        xy_lens = x_lens + y_lens
        xy_mask = x_mask + y_mask
        xyt = self._temporal_encoding(t)

        ctx = xyt + self.mask_emb(torch.cat([x_mask, y_mask], dim=-1))
        idx_t = repeat(torch.arange(max_lens, device=dev), "t -> b t", b=bsz)
        idx_flip = torch.where(
            idx_t < xy_lens[..., None], xy_lens[..., None] - idx_t - 1, idx_t
        )
        idx_flip = repeat(idx_flip, "b t -> b t d", d=self.d_model1)
        ctx_flip = self.mamba1(torch.gather(ctx, 1, idx_flip))
        ctx = torch.gather(ctx_flip, 1, idx_flip)
        ctx = repeat(self.ctx_proj(ctx), "b t d -> b t c d", c=nc)

        query = torch.einsum("btd,btc->btcd", xyt, y_mask)

        query_max = query.max(dim=1).values
        query_avg = query.sum(dim=1) / (y_mask.sum(dim=1)[..., None] + 1e-8)

        query = self.query_proj(torch.cat([query_max, query_avg], dim=-1))
        query = repeat(query, "b c d -> b t c d", t=max_lens)

        c_onehot = repeat(
            torch.eye(nc, device=dev), "c d -> b t c d", b=bsz, t=max_lens
        )
        xy = self.x_emb(torch.cat([x[..., None], y_mask[..., None], c_onehot], -1))

        idx_b = repeat(torch.arange(bsz, device=dev), "b -> b t c 1", t=max_lens, c=nc)
        idx_t = repeat(idx_t, "b t -> b t c 1", c=nc)
        idx_c = repeat(torch.arange(nc, device=dev), "c -> b t c 1", b=bsz, t=max_lens)

        xy = torch.cat([xy, ctx, query, idx_b, idx_t, idx_c], dim=-1)
        dim_xy = xy.shape[-1]

        num_obs = xy_mask.sum(dim=[1, 2]).to(int)
        max_num_obs = num_obs.max().item()
        if max_num_obs == 0:
            return self.head(xy.new_zeros(bsz, max_lens, nc, self.d_model2)).squeeze(-1)
        xy_valid = xy.view(-1, dim_xy)[xy_mask.view(-1).bool()]
        idx_b_flat = xy_valid[:, -3].to(int)
        idx_l = torch.cat([torch.arange(l, device=dev) for l in num_obs])
        xy_flat = xy.new_zeros(bsz, max_num_obs, dim_xy) * torch.nan
        xy_flat.index_put_((idx_b_flat, idx_l), xy_valid)
        padding_mask = ~torch.isnan(xy_flat[..., 0])

        xy_flat, ctx, query, coords = xy_flat.nan_to_num(0, 0, 0).split(
            [self.d_model2, self.d_model2 * self.expand2, self.d_state2, 3], -1
        )

        fusion_sources = token_predecessors(x_mask, y_mask, coords, padding_mask)
        fusion_routes = None
        if x.is_cuda:
            from .k1_scan import prepare_k1_routes

            fusion_routes = prepare_k1_routes(fusion_sources)
        xy_flat = self.mamba2(xy_flat, ctx, query, fusion_sources, fusion_routes)[
            padding_mask
        ]

        coords = coords[padding_mask].transpose(0, 1).to(int)
        xy = xy_flat.new_zeros(bsz, max_lens, nc, self.d_model2)
        xy.index_put_((coords[0], coords[1], coords[2]), xy_flat)

        xy = self.head(xy).squeeze(-1)
        return xy
