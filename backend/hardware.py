"""Local hardware telemetry (CPU, RAM, NVIDIA GPU VRAM) with short-lived caching."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from typing import Any

import psutil

_CACHE_TTL_SEC = 1.5
_last_snapshot: dict[str, Any] | None = None
_last_snapshot_ts: float = 0.0
_NVIDIA_SMI_MISSING: bool | None = None

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _query_nvidia_smi() -> dict[str, Any]:
    global _NVIDIA_SMI_MISSING
    if _NVIDIA_SMI_MISSING is True:
        return {"gpu_available": False}
    if _NVIDIA_SMI_MISSING is None:
        if not shutil.which("nvidia-smi"):
            _NVIDIA_SMI_MISSING = True
            return {"gpu_available": False}
        _NVIDIA_SMI_MISSING = False
    try:
        res = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2.0,
            creationflags=_NO_WINDOW,
        )
        if res.returncode != 0 or not res.stdout.strip():
            return {"gpu_available": False}
        line = res.stdout.strip().splitlines()[0]
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            return {"gpu_available": False}
        vram_used = float(parts[1])
        vram_total = float(parts[2])
        vram_percent = round((vram_used / vram_total) * 100.0, 1) if vram_total > 0 else 0.0
        return {
            "gpu_available": True,
            "gpu_name": parts[0],
            "vram_used_mb": int(round(vram_used)),
            "vram_total_mb": int(round(vram_total)),
            "vram_percent": vram_percent,
            "gpu_util_percent": int(float(parts[3])),
            "gpu_temp_c": int(float(parts[4])),
        }
    except Exception:
        return {"gpu_available": False}


def get_system_stats(force_refresh: bool = False) -> dict[str, Any]:
    """Return RAM, CPU and NVIDIA VRAM telemetry, cached briefly to avoid polling overhead."""
    global _last_snapshot, _last_snapshot_ts
    now = time.monotonic()
    if not force_refresh and _last_snapshot is not None and (now - _last_snapshot_ts) < _CACHE_TTL_SEC:
        return dict(_last_snapshot)

    vm = psutil.virtual_memory()
    cpu_percent = psutil.cpu_percent(interval=None)
    gpu = _query_nvidia_smi()

    snapshot: dict[str, Any] = {
        "ram_used_gb": round(vm.used / (1024**3), 2),
        "ram_total_gb": round(vm.total / (1024**3), 2),
        "ram_percent": round(vm.percent, 1),
        "cpu_percent": round(cpu_percent, 1),
        "cpu_count": os.cpu_count() or 4,
        "gpu_available": gpu.get("gpu_available", False),
        "gpu_name": gpu.get("gpu_name", "No detectada"),
        "vram_used_mb": gpu.get("vram_used_mb", 0),
        "vram_total_mb": gpu.get("vram_total_mb", 0),
        "vram_percent": gpu.get("vram_percent", 0.0),
        "gpu_util_percent": gpu.get("gpu_util_percent", 0),
        "gpu_temp_c": gpu.get("gpu_temp_c", 0),
    }
    _last_snapshot = snapshot
    _last_snapshot_ts = now
    return dict(snapshot)
