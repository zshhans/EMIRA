"""GPU-validated route snapshots, shared by all fused layers of one forward."""
from dataclasses import dataclass
import weakref

import torch
from .loader import routing_extension


@dataclass(frozen=True)
class K1Routes:
    source: torch.Tensor
    inverse: torch.Tensor
    slots: torch.Tensor
    count: int
    strategy: str = "gpu"

    def tensors(self):
        return self.source, self.inverse, self.slots, self.count


def prepare_k1_routes(sources, strategy="gpu"):
    if strategy != "gpu":
        raise ValueError("the model backend supports GPU routes only")
    if (
        sources.ndim != 2
        or sources.dtype not in (torch.int32, torch.int64)
        or not sources.is_cuda
    ):
        raise ValueError("CUDA int32/int64 [B,L] sources required")
    with torch.inference_mode(False):
        src, inv, slots, errors = routing_extension().build_routes(sources.contiguous())
        flag = int(errors.item())
        if flag & 1:
            raise ValueError("source outside its sample")
        if flag & 2:
            raise ValueError("source must precede target")
        if flag & 4:
            raise ValueError("a source may have only one successor")
        count = sources.shape[1] if sources.shape[1] > 2048 else 0
    return K1Routes(src, inv, slots, count)


_cache = {}


def cached_k1_routes(sources, strategy="gpu"):
    version = None if torch.is_inference(sources) else sources._version
    key = (id(sources), strategy)
    old = _cache.get(key)
    if version is not None and old and old[0]() is sources and old[1] == version:
        return old[2]
    routes = prepare_k1_routes(sources, strategy)
    if version is not None:
        ref = weakref.ref(sources, lambda _: _cache.pop(key, None))
        _cache[key] = (ref, version, routes)
    return routes
