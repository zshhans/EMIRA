import torch
from torch.autograd.function import once_differentiable
from .loader import deterministic_extension
from .packed_light_ops import layout


def _validate(u, dt, A, B, C, D, z, bias):
    if not u.is_cuda or u.ndim != 3 or any(n < 1 for n in u.shape):
        raise ValueError("nonempty CUDA [batch,channels,length] input required")
    if (
        A.dtype != torch.float32
        or A.ndim != 2
        or not 1 <= A.shape[1] <= 256
        or A.shape[0] != u.shape[1]
    ):
        raise ValueError("real FP32 A [channels,state<=256] required")
    if (
        B.ndim != 3
        or C.shape != B.shape
        or B.shape != (u.shape[0], A.shape[1], u.shape[2])
    ):
        raise ValueError("single-group variable B/C [batch,state,length] required")
    if dt.shape != u.shape or (z is not None and z.shape != u.shape):
        raise ValueError("delta and optional z must match u")
    if any(t.device != u.device for t in (dt, A, B, C, D, z, bias) if t is not None):
        raise ValueError("all tensors must share a CUDA device")
    if u.dtype not in (torch.float32, torch.float16, torch.bfloat16) or any(
        t.dtype != u.dtype for t in (dt, B, C, z) if t is not None
    ):
        raise ValueError("u/delta/variable B/C/z require matching FP32/FP16/BF16 types")
    for t in (D, bias):
        if t is not None and (t.shape != (u.shape[1],) or t.dtype != torch.float32):
            raise ValueError("D and delta_bias must be FP32 [channels]")


class _DeterministicNative(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, dt, A, B, C, D, z, bias, softplus, return_last):
        from mamba_ssm.ops.selective_scan_interface import selective_scan_cuda

        ctx.ndim_B, ctx.ndim_C = B.ndim, C.ndim
        if B.ndim == 4:
            if B.shape[1] != 1:
                raise ValueError("only group1 variable B supported")
            B = B[:, 0]
        if C.ndim == 4:
            if C.shape[1] != 1:
                raise ValueError("only group1 variable C supported")
            C = C[:, 0]
        _validate(u, dt, A, B, C, D, z, bias)
        u, dt, A, B, C, D, z, bias = [
            v.contiguous() if v is not None else None
            for v in (u, dt, A, B, C, D, z, bias)
        ]

        out, checkpoints, *rest = selective_scan_cuda.fwd(
            u, dt, A, B[:, None], C[:, None], D, z, bias, softplus
        )
        ctx.save_for_backward(
            u, dt, A, B, C, D, z, bias, checkpoints, out if z is not None else None
        )
        ctx.softplus = softplus
        result = out if z is None else rest[0]
        return (result, checkpoints[:, :, -1, 1::2]) if return_last else result

    @staticmethod
    @once_differentiable
    def backward(ctx, dy, *ignored_last_state_grad):
        u, dt, A, B, C, D, z, bias, checkpoints, out = ctx.saved_tensors
        threads, items = layout(u.shape[-1], 1)
        ext = deterministic_extension(1, 2)
        absent = torch.full(
            (u.shape[0], u.shape[2]), -1, device=u.device, dtype=torch.int32
        )
        w = torch.zeros(absent.shape, device=u.device, dtype=torch.float32)
        pb, pc, neighbor_c, _ = ext.prepare(B, C, absent, w, threads, items)
        cache = torch.empty(
            (u.shape[0], u.shape[1], 0), device=u.device, dtype=torch.float32
        )
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
            out.float().contiguous() if out is not None else None,
            None,
            ctx.softplus,
            False,
            absent,
            absent,
            absent,
            w,
            cache,
            neighbor_c,
            u.shape[1],
        )
        du, ddt, da, gb, gc, dD, dbias = grads[:7]
        db, dc, _ = ext.finish_gradients(
            gb, gc, grads[-1], absent, C, w, threads, items
        )
        if ctx.ndim_B == 4:
            db = db[:, None]
        if ctx.ndim_C == 4:
            dc = dc[:, None]
        return (
            du,
            ddt,
            da,
            db,
            dc,
            dD if D is not None else None,
            grads[7] if z is not None else None,
            dbias if bias is not None else None,
            None,
            None,
        )


def deterministic_native_scan(
    u,
    delta,
    A,
    B,
    C,
    D=None,
    z=None,
    delta_bias=None,
    delta_softplus=False,
    return_last_state=False,
):
    """mamba_ssm-compatible real group1 API; last-state gradient is ignored as upstream."""
    return _DeterministicNative.apply(
        u, delta, A, B, C, D, z, delta_bias, delta_softplus, return_last_state
    )
