from __future__ import annotations

import datetime as dt
import os
import shlex
import shutil
import subprocess
import threading
from typing import Any, Dict, List, Optional, Tuple

import psutil

MB = 1024 * 1024
_NA_STRINGS = {"", "n/a", "na", "not supported", "not available", "--"}

_NVML = None
_NVML_READY = False
_NVML_INITIALIZED = False
_NVML_LOCK = threading.Lock()


def _ensure_nvml():
    """Lazy-load and initialize pynvml; returns the module on success or None otherwise."""
    global _NVML, _NVML_READY, _NVML_INITIALIZED
    with _NVML_LOCK:
        if _NVML_INITIALIZED:
            return _NVML if _NVML_READY else None
        _NVML_INITIALIZED = True
        try:
            import pynvml as nvml  # type: ignore

            nvml.nvmlInit()
        except Exception:  # noqa: BLE001
            _NVML_READY = False
            _NVML = None
        else:
            _NVML_READY = True
            _NVML = nvml
        return _NVML if _NVML_READY else None


def _process_info(pid: int) -> Optional[Dict[str, Any]]:
    try:
        proc = psutil.Process(pid)
    except (psutil.NoSuchProcess, psutil.ZombieProcess, psutil.AccessDenied):
        return None
    try:
        with proc.oneshot():
            username = proc.username()
            name = proc.name()
            cmdline = " ".join(shlex.quote(part) for part in proc.cmdline()) or name
            cpu_percent = proc.cpu_percent(interval=None)
            mem = proc.memory_info()
            threads = proc.num_threads()
            create_time = proc.create_time()
    except (psutil.NoSuchProcess, psutil.ZombieProcess, psutil.AccessDenied):
        return None
    return {
        "pid": pid,
        "user": username,
        "name": name,
        "cmdline": cmdline,
        "cpu_percent": round(cpu_percent, 2),
        "ram_mb": round(mem.rss / MB, 2),
        "threads": threads,
        "create_time": dt.datetime.fromtimestamp(create_time).isoformat(),
    }


def _safe_float(value: Optional[str]) -> Optional[float]:
    """Convert NVIDIA-SMI numeric string to float, tolerating N/A and suffixed units."""
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        return None
    lower = normalized.lower()
    if lower in _NA_STRINGS:
        return None
    for suffix in ("MiB", "GiB", "W", "%"):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)].strip()
            break
    try:
        return float(normalized)
    except ValueError:
        return None


def _collect_with_nvml() -> Optional[Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]]:
    nvml = _ensure_nvml()
    if not nvml:
        return None
    try:
        device_count = nvml.nvmlDeviceGetCount()
    except Exception:  # noqa: BLE001
        return None

    gpus: List[Dict[str, Any]] = []
    processes: List[Dict[str, Any]] = []
    for index in range(device_count):
        try:
            handle = nvml.nvmlDeviceGetHandleByIndex(index)
            name = nvml.nvmlDeviceGetName(handle)
            uuid = nvml.nvmlDeviceGetUUID(handle)
            memory = nvml.nvmlDeviceGetMemoryInfo(handle)
            util = nvml.nvmlDeviceGetUtilizationRates(handle)
        except Exception:  # noqa: BLE001
            continue

        def _decode(value):
            return value.decode("utf-8") if isinstance(value, bytes) else str(value)

        gpu_entry = {
            "index": index,
            "name": _decode(name),
            "uuid": _decode(uuid),
            "memory_total_mb": round(memory.total / MB, 2),
            "memory_used_mb": round(memory.used / MB, 2),
            "memory_free_mb": round(memory.free / MB, 2),
            "utilization_gpu": getattr(util, "gpu", 0),
            "utilization_mem": getattr(util, "memory", 0),
            "temperature": None,
            "power_w": None,
        }
        try:
            gpu_entry["temperature"] = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
        except Exception:  # noqa: BLE001
            pass
        try:
            gpu_entry["power_w"] = round(nvml.nvmlDeviceGetPowerUsage(handle) / 1000, 2)
        except Exception:  # noqa: BLE001
            pass

        gpus.append(gpu_entry)

        pid_mem: Dict[int, int] = {}
        for getter_name in (
            "nvmlDeviceGetGraphicsRunningProcesses_v2",
            "nvmlDeviceGetComputeRunningProcesses_v2",
        ):
            getter = getattr(nvml, getter_name, None)
            if getter is None:
                continue
            try:
                proc_entries = getter(handle) or []
            except Exception:  # noqa: BLE001
                continue
            for proc_entry in proc_entries:
                pid = getattr(proc_entry, "pid", None)
                used_mem = getattr(proc_entry, "usedGpuMemory", 0) or 0
                if pid is None:
                    continue
                pid_mem[pid] = max(used_mem, pid_mem.get(pid, 0))

        for pid, used_bytes in pid_mem.items():
            info = _process_info(pid)
            if not info:
                continue
            process_entry = {
                **info,
                "gpu_index": index,
                "gpu_uuid": gpu_entry["uuid"],
                "gpu_memory_mb": round(used_bytes / MB, 2),
            }
            processes.append(process_entry)

    return gpus, processes


def _collect_with_nvidia_smi() -> Optional[Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]]:
    if shutil.which("nvidia-smi") is None:
        return None
    gpu_fields = [
        "name",
        "uuid",
        "memory.total",
        "memory.used",
        "memory.free",
        "utilization.gpu",
        "utilization.memory",
        "temperature.gpu",
        "power.draw",
    ]
    base_cmd = ["nvidia-smi", f"--query-gpu={','.join(gpu_fields)}", "--format=csv,noheader,nounits"]
    try:
        gpu_output = subprocess.check_output(base_cmd, encoding="utf-8")
    except (OSError, subprocess.CalledProcessError):
        return None

    gpus: List[Dict[str, Any]] = []
    for idx, line in enumerate(gpu_output.strip().splitlines()):
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != len(gpu_fields):
            continue
        memory_total = _safe_float(parts[2])
        memory_used = _safe_float(parts[3])
        memory_free = _safe_float(parts[4])
        gpus.append(
            {
                "index": idx,
                "name": parts[0],
                "uuid": parts[1],
                "memory_total_mb": memory_total,
                "memory_used_mb": memory_used,
                "memory_free_mb": memory_free,
                "utilization_gpu": _safe_float(parts[5]),
                "utilization_mem": _safe_float(parts[6]),
                "temperature": _safe_float(parts[7]),
                "power_w": _safe_float(parts[8]),
            }
        )

    proc_fields = ["pid", "process_name", "used_memory", "gpu_uuid"]
    proc_cmd = ["nvidia-smi", f"--query-compute-apps={','.join(proc_fields)}", "--format=csv,noheader,nounits"]
    processes: List[Dict[str, Any]] = []
    try:
        proc_output = subprocess.check_output(proc_cmd, encoding="utf-8")
    except (subprocess.CalledProcessError, OSError):
        proc_output = ""
    for line in proc_output.strip().splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != len(proc_fields):
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        info = _process_info(pid)
        if not info:
            continue
        used_mem = _safe_float(parts[2])
        if used_mem is None:
            continue
        processes.append(
            {
                **info,
                "gpu_memory_mb": used_mem,
                "gpu_uuid": parts[3],
                "gpu_index": next((gpu["index"] for gpu in gpus if gpu["uuid"] == parts[3]), None),
            }
        )

    return gpus, processes


def _collect_host_stats() -> Dict[str, Any]:
    vm = psutil.virtual_memory()
    sm = psutil.swap_memory()
    cpu_percent = psutil.cpu_percent(interval=None)
    load_1, load_5, load_15 = os.getloadavg() if hasattr(os, "getloadavg") else (0.0, 0.0, 0.0)
    return {
        "cpu_percent": round(cpu_percent, 2),
        "memory_percent": round(vm.percent, 2),
        "memory_used_gb": round(vm.used / (1024**3), 2),
        "memory_total_gb": round(vm.total / (1024**3), 2),
        "swap_percent": round(sm.percent, 2),
        "swap_used_gb": round(sm.used / (1024**3), 2),
        "swap_total_gb": round(sm.total / (1024**3), 2),
        "load_1": round(load_1, 2),
        "load_5": round(load_5, 2),
        "load_15": round(load_15, 2),
        "timestamp": dt.datetime.now().isoformat(),
    }


def collect_gpu_metrics() -> Dict[str, Any]:
    """Collect GPU/host utilization data for the monitoring dashboard."""
    provider = "nvml"
    payload = _collect_with_nvml()
    if payload is None:
        provider = "nvidia-smi"
        payload = _collect_with_nvidia_smi()
    metrics = {
        "timestamp": dt.datetime.now().isoformat(),
        "host": _collect_host_stats(),
        "gpus": [],
        "processes": [],
        "provider": provider,
        "message": "",
    }
    if payload is None:
        metrics["provider"] = "unavailable"
        metrics["message"] = "当前环境未检测到可用的 NVIDIA GPU 或 NVML 驱动。"
        return metrics
    gpus, processes = payload
    metrics["gpus"] = gpus
    metrics["processes"] = processes
    metrics["message"] = (
        "已连接到 NVIDIA Management Library (NVML)。" if provider == "nvml" else "使用 nvidia-smi 命令进行数据采集。"
    )
    return metrics


__all__ = ["collect_gpu_metrics"]
