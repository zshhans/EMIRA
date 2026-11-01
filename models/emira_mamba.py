import math

import torch
from torch import nn
from torch.nn import functional as F
from einops import repeat

from .pscan import pscan
from .input_gate_scan import selective_scan_input_convex, mask_missing_predecessors


def selective_scan(x, dt, A, B, C, D, z, bias):
    if x.is_cuda:
        if torch.are_deterministic_algorithms_enabled():
            from .k1_scan.deterministic_ops import deterministic_native_scan as scan
        else:
            from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as scan
        u, bb, cc, zz = (v.transpose(1, 2).contiguous() for v in (x, B, C, z))
        return scan(u, dt, A, bb, cc, D, zz, bias, delta_softplus=True).transpose(1, 2)
    delta = F.softplus(dt.transpose(1, 2) + bias)
    states = pscan(
        torch.exp(delta.unsqueeze(-1) * A),
        delta.unsqueeze(-1) * B.unsqueeze(2) * x.unsqueeze(-1),
    )
    y = (states @ C.unsqueeze(-1)).squeeze(3)
    return (y + D * x) * F.silu(z)


class EmiraMambaBlock(nn.Module):
    def __init__(
        self, d_model, d_state, expand, dt_rank, state_fusion=False, fusion_init=0.05
    ):
        super().__init__()
        self.d_inner = expand * d_model
        self.d_state = d_state
        self.dt_rank = dt_rank
        self.state_fusion = state_fusion
        if state_fusion:
            # Gate creation must preserve the seeded backbone initialization.
            with torch.random.fork_rng(devices=[]):
                self.fusion_gate = nn.Linear(self.d_inner, 1)
            nn.init.zeros_(self.fusion_gate.weight)
            nn.init.constant_(
                self.fusion_gate.bias, math.log(fusion_init / (1 - fusion_init))
            )
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.x_proj = nn.Linear(self.d_inner, dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(dt_rank, self.d_inner, bias=True)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        nn.init.uniform_(self.dt_proj.weight, -(dt_rank**-0.5), dt_rank**-0.5)
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001))
            + math.log(0.001)
        ).clamp(min=1e-4)
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        self.dt_proj.bias._no_reinit = True
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True

    def forward(
        self,
        hidden_states,
        ctx=None,
        query=None,
        fusion_sources=None,
        fusion_routes=None,
    ):
        x, z = self.in_proj(hidden_states).chunk(2, dim=2)
        A = -torch.exp(self.A_log.float())
        features = x if ctx is None else x + ctx
        dt, B, C = self.x_proj(features).split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = self.dt_proj.weight @ dt.transpose(1, 2)
        if query is not None:
            C = C * query
        args = (x, dt, A, B, C, self.D.float(), z, self.dt_proj.bias.float())
        if self.state_fusion:
            with torch.autocast(device_type=x.device.type, enabled=False):
                gate = (
                    F.linear(
                        x.float(),
                        self.fusion_gate.weight.float(),
                        self.fusion_gate.bias.float(),
                    )
                    .squeeze(-1)
                    .sigmoid()
                )
            gate = mask_missing_predecessors(gate, fusion_sources)
            y = selective_scan_input_convex(
                *args,
                gate=gate,
                sources=fusion_sources,
                routes=fusion_routes,
                cuda=x.is_cuda
            )
        else:
            y = selective_scan(*args)
        return self.out_proj(y)


class GatedMLP(nn.Module):
    def __init__(self, d_model, expand):
        super().__init__()
        self.fc1 = nn.Linear(d_model, 2 * d_model * expand)
        self.fc2 = nn.Linear(d_model * expand, d_model)

    def forward(self, x):
        y, gate = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(y * F.silu(gate))


class ResidualBlock(nn.Module):
    def __init__(
        self,
        d_model,
        d_state,
        expand,
        dt_rank,
        drop,
        state_fusion=False,
        fusion_init=0.05,
    ):
        super().__init__()
        self.mixer = EmiraMambaBlock(
            d_model, d_state, expand, dt_rank, state_fusion, fusion_init
        )
        self.mlp = GatedMLP(d_model, expand)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(drop)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x, ctx=None, query=None, fusion_sources=None, fusion_routes=None):
        x = x + self.drop1(
            self.mixer(self.norm1(x), ctx, query, fusion_sources, fusion_routes)
        )
        return x + self.drop2(self.mlp(self.norm2(x)))


class EmiraMamba(nn.Module):
    def __init__(
        self,
        n_layers,
        d_model,
        d_state,
        expand,
        dt_rank,
        drop,
        state_fusion=False,
        fusion_init=0.05,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                ResidualBlock(
                    d_model, d_state, expand, dt_rank, drop, state_fusion, fusion_init
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, x, ctx=None, query=None, fusion_sources=None, fusion_routes=None):
        for layer in self.layers:
            x = layer(x, ctx, query, fusion_sources, fusion_routes)
        return x
