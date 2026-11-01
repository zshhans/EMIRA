from functools import lru_cache
from pathlib import Path
from torch.utils.cpp_extension import load

ROOT = Path(__file__).parent


@lru_cache(None)
def speed_extension(fwd_wide=0, bwd_wide=0, min_blocks=2):
    if fwd_wide not in (0, 1) or bwd_wide not in (0, 1) or min_blocks not in (1, 2):
        raise ValueError("invalid speed-kernel layout")
    root = ROOT / "speed"
    tag = f"f{fwd_wide}b{bwd_wide}m{min_blocks}"
    symbols = [
        "selective_scan_fwd",
        "selective_scan_bwd",
        "selective_scan_fwd_cuda",
        "selective_scan_bwd_cuda",
        "selective_scan_fwd_kernel",
        "selective_scan_bwd_kernel",
        "selective_scan_fwd_launch",
        "selective_scan_bwd_launch",
        "set_ssm_params_fwd",
        "set_ssm_params_bwd",
        "emira_prepare_e",
        "emira_finish_gradients",
    ]
    names = [f"-D{s}=emira_speed_{tag}_{s}" for s in symbols]
    defines = [
        f"-DEMIRA_SPEED_FWD_WIDE={fwd_wide}",
        f"-DEMIRA_SPEED_BWD_WIDE={bwd_wide}",
        f"-DEMIRA_SPEED_MIN_BLOCKS={min_blocks}",
    ] + names
    return load(
        name=f"emira_model_k1_speed_{tag}",
        sources=[str(root / "selective_scan.cpp"), str(ROOT / "speed_routes.cu")]
        + [str(p) for p in sorted(root.glob("*.cu"))],
        extra_cflags=["-O3", "-std=c++17", "-fvisibility=hidden"] + defines,
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            "--use_fast_math",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
        ]
        + defines,
        verbose=True,
    )


@lru_cache(None)
def routing_extension():
    return load(
        name="emira_model_k1_gpu_routes",
        sources=[str(ROOT / "gpu_routing.cu")],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "-std=c++17"],
        verbose=True,
    )


@lru_cache(None)
def packed_light_extension(wide=1, min_blocks=2):
    if wide not in (0, 1) or min_blocks not in (1, 2, 3):
        raise ValueError("invalid packed light layout")
    root = ROOT / "packed_light"
    tag = f"b{wide}m{min_blocks}"
    symbols = [
        "selective_scan_bwd",
        "selective_scan_bwd_cuda",
        "selective_scan_bwd_kernel",
        "selective_scan_bwd_launch",
        "set_ssm_params_fwd",
        "set_ssm_params_bwd",
        "emira_pack",
        "emira_unpack_gradients",
    ]
    names = [f"-D{s}=emira_packed_light_{tag}_{s}" for s in symbols]
    defines = [
        f"-DEMIRA_SPEED_BWD_WIDE={wide}",
        f"-DEMIRA_SPEED_MIN_BLOCKS={min_blocks}",
    ] + names
    return load(
        name=f"emira_model_k1_packed_light_{tag}",
        sources=[str(root / "selective_scan.cpp"), str(ROOT / "packed_light_routes.cu")]
        + [str(p) for p in sorted(root.glob("bwd_*.cu"))],
        extra_cflags=["-O3", "-std=c++17", "-fvisibility=hidden"] + defines,
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            "--use_fast_math",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
        ]
        + defines,
        verbose=True,
    )


@lru_cache(None)
def deterministic_extension(wide=1, min_blocks=2):
    root = ROOT / "deterministic"
    tag = f"b{wide}m{min_blocks}"
    symbols = [
        "selective_scan_bwd",
        "selective_scan_bwd_cuda",
        "selective_scan_bwd_kernel",
        "selective_scan_bwd_launch",
        "set_ssm_params_fwd",
        "set_ssm_params_bwd",
        "emira_pack",
        "emira_unpack_gradients",
    ]
    defines = [
        f"-DEMIRA_SPEED_BWD_WIDE={wide}",
        f"-DEMIRA_SPEED_MIN_BLOCKS={min_blocks}",
    ]
    defines += [f"-D{s}=emira_deterministic_{tag}_{s}" for s in symbols]
    return load(
        name=f"emira_deterministic_{tag}",
        sources=[
            str(root / "selective_scan.cpp"),
            str(root / "routes.cu"),
            str(root / "reduce.cu"),
        ]
        + [str(p) for p in sorted(root.glob("bwd_*.cu"))],
        extra_cflags=["-O3", "-std=c++17", "-fvisibility=hidden"] + defines,
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            "--use_fast_math",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
        ]
        + defines,
        verbose=True,
    )
