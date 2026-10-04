import enum
import gc
from functools import cache

import torch

from app.settings import SETTINGS


class HWVendor(enum.StrEnum):
    "Accelerator type"

    CUDA = enum.auto()
    "NVIDIA"
    ROCm = enum.auto()
    "AMD"
    MPS = enum.auto()
    "Apple Silicon"
    XPU = enum.auto()
    "Intel GPU"
    CPU = enum.auto()
    "Fallback. Check your drivers?"

    @property
    def device_type(self) -> str:
        match self:
            case HWVendor.CUDA | HWVendor.ROCm:
                return "cuda"  # ROCm builds of torch also expose the "cuda" device type
            case HWVendor.MPS:
                return "mps"
            case HWVendor.XPU:
                return "xpu"
            case HWVendor.CPU:
                return "cpu"


@cache
def vendor() -> HWVendor:
    "Get vendor for installed Torch version"
    if torch.cuda.is_available():
        return HWVendor.ROCm if torch.version.hip else HWVendor.CUDA
    elif torch.backends.mps.is_available():
        return HWVendor.MPS
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        return HWVendor.XPU
    else:
        return HWVendor.CPU


def device() -> torch.device:
    "Device from settings, or auto-detected from the installed torch."
    if SETTINGS.DEVICE != "auto":
        dev = torch.device(SETTINGS.DEVICE)
        if dev.type != "cpu" and dev.type != vendor().device_type:
            raise RuntimeError(f"DEVICE={SETTINGS.DEVICE} requested but this machine has {vendor()}")
        return dev
    dev_type = vendor().device_type
    return torch.device(dev_type, 0) if dev_type in ("cuda", "xpu") else torch.device(dev_type)


def dtype() -> torch.dtype:
    "Half precision on accelerators, full precision on CPU."
    match vendor():
        case HWVendor.CPU:
            return torch.float32  # half precision on CPU is slow or unsupported
        case _:
            return torch.bfloat16


def memory_info() -> tuple[int, int]:
    "(used, total) bytes on the inference device. MPS total is the unified-memory budget Metal allows."
    match device().type:
        case "cuda":
            free, total = torch.cuda.mem_get_info(device())
            return total - free, total
        case "mps":
            return (
                torch.mps.driver_allocated_memory(),
                torch.mps.recommended_max_memory(),
            )
        case "xpu":
            free, total = torch.xpu.mem_get_info(device())
            return total - free, total
        case _:
            return 0, 0


def gib(n: float) -> str:
    return f"{n / 1024**3:.2f} GiB"


def empty_cache() -> None:
    "Release cached allocator memory."
    gc.collect()
    match device().type:
        case "cuda":
            torch.cuda.empty_cache()
        case "mps":
            torch.mps.empty_cache()
        case "xpu":
            torch.xpu.empty_cache()
