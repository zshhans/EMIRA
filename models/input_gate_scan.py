import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable

from .pscan import pscan


def mask_missing_predecessors(gate, sources):
    """Zero the coefficient on absent edges; valid zero-valued states stay valid."""
    if gate.shape != sources.shape:
        raise ValueError("gate and sources must have matching [B,L] shapes")
    return gate.masked_fill(sources < 0, 0.0)


class _InputConvex(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, u, dt, A, B, C, D, z, bias, gate, src, inv, slots, count, training
    ):
        from .k1_scan.loader import packed_light_extension
        from .k1_scan.packed_light_ops import layout, DEFAULT_CONFIG
        from .k1_scan.speed_ops import kernel_extension

        args = [
            v.contiguous() if v is not None else None
            for v in (u, dt, A, B, C, D, z, bias, gate)
        ]
        u, dt, A, B, C, D, z, bias, gate = args
        config = DEFAULT_CONFIG
        threads, items = layout(u.shape[-1], config[1])
        ext = packed_light_extension(config[1], config[2])
        self_c = (C.float() * (1 - gate[:, None])).to(C.dtype).contiguous()

        _, _, neighbor_c, extra_c = ext.prepare(B, C, inv, gate, threads, items)
        if training:
            pb, pc, _, _ = ext.prepare(B, self_c, inv, gate, threads, items)
        fw = kernel_extension((config[0], config[1], 2, config[3]), True)
        out, checkpoints, cache, pre = fw.fwd(
            u,
            dt,
            A,
            B[:, None],
            self_c[:, None],
            D,
            z,
            bias,
            True,
            src,
            inv,
            slots,
            gate,
            count,
            extra_c,
            training,
        )
        if training:
            ctx.save_for_backward(
                *args, src, inv, slots, pb, pc, neighbor_c, checkpoints, cache, pre
            )
            ctx.config, ctx.layout = config, (threads, items)
            ctx.deterministic = torch.are_deterministic_algorithms_enabled()
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        from .k1_scan.loader import packed_light_extension, deterministic_extension

        (
            u,
            dt,
            A,
            B,
            C,
            D,
            z,
            bias,
            gate,
            src,
            inv,
            slots,
            pb,
            pc,
            neighbor_c,
            checkpoints,
            cache,
            pre,
        ) = ctx.saved_tensors
        ext = (
            deterministic_extension if ctx.deterministic else packed_light_extension
        )(ctx.config[1], ctx.config[2])
        groups = u.shape[1] if ctx.deterministic else ctx.config[3]
        grads = ext.bwd(
            u,
            dt,
            A,
            pb[:, None],
            pc[:, None],
            D,
            z,
            bias,
            dy.contiguous(),
            checkpoints,
            pre if z is not None else None,
            None,
            True,
            False,
            src,
            inv,
            slots,
            gate,
            cache,
            neighbor_c,
            groups,
        )
        du, ddt, da, gb, gc, dD, dbias = grads[:7]
        db, dc, dg = ext.finish_gradients(gb, gc, grads[-1], src, C, gate, *ctx.layout)

        absent = torch.full_like(src, -1)
        _, dc_self, _ = ext.finish_gradients(
            gb, gc, grads[-1], absent, C, gate, *ctx.layout
        )
        dc = (dc.float() - gate[:, None] * dc_self.float()).to(C.dtype)
        dg = dg - (C.float() * dc_self.float()).sum(1)
        return (
            du,
            ddt,
            da,
            db,
            dc,
            dD if D is not None else None,
            grads[7] if z is not None else None,
            dbias if bias is not None else None,
            dg,
            None,
            None,
            None,
            None,
            None,
        )


def selective_scan_input_convex(
    x, dt, A, B, C, D, z, bias, *, gate, sources, routes=None, cuda=False
):
    """g[B,L] multiplies predecessor state, shared over channels/state dims."""
    if gate.shape != sources.shape or gate.shape != x.shape[:2]:
        raise ValueError("gate and sources must match token shape [B,L]")
    with torch.autocast(device_type=x.device.type, enabled=False):
        if cuda:
            from .k1_scan.deterministic_ops import _validate
            from .k1_scan.speed_routing import cached_k1_routes, K1Routes

            u = x.transpose(1, 2)
            bb, cc = (v.transpose(1, 2).to(x.dtype) for v in (B, C))
            zz = None if z is None else z.transpose(1, 2).to(x.dtype)
            delta = dt.to(x.dtype)
            _validate(u, delta, A, bb, cc, D, zz, bias)
            route = cached_k1_routes(sources, "gpu") if routes is None else routes
            if (
                not isinstance(route, K1Routes)
                or route.source.shape != sources.shape
                or route.source.device != x.device
            ):
                raise ValueError("matching prepared K1Routes required")
            arguments = (u, delta, A, bb, cc, D, zz, bias, gate.float())
            training = torch.is_grad_enabled() and any(
                t is not None and t.requires_grad for t in arguments
            )
            return _InputConvex.apply(*arguments, *route.tensors(), training).transpose(
                1, 2
            )

        delta = F.softplus(dt.transpose(1, 2).float() + bias.float())
        states = pscan(
            torch.exp(delta[..., None] * A.float()),
            delta[..., None] * B.float().unsqueeze(2) * x.float().unsqueeze(-1),
        )
        index = sources.clamp_min(0).long()[..., None, None].expand_as(states)
        neighbor = states.gather(1, index) * (sources >= 0)[..., None, None]
        fused = (1 - gate[..., None, None]) * states + gate[..., None, None] * neighbor
        y = (fused * C.float().unsqueeze(2)).sum(-1)
        if D is not None:
            y = y + D.float() * x.float()
        if z is not None:
            y = y * F.silu(z.float())
        return y.to(x.dtype)
