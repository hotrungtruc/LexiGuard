"""
Hardware Detection & Compute Optimization Module
=================================================
Centralized single-source-of-truth for hardware profiling, CPU thread allocation,
and GPU/CPU acceleration tuning across LexiGuard.

Features:
- Physical vs. Logical Core Detection (prevents thread contention in llama.cpp/Ollama)
- System RAM Assessment & Dual-LLM Co-residency Recommendation
- GPU/CUDA/MPS Detection & VRAM Sizing
- Dynamic Thread Allocation for Ollama, OCR, and Background Ingestion
"""

import os
import sys
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field
from loguru import logger


class HardwareProfile(BaseModel):
    """
    Normalized hardware capability profile of the host machine.
    """
    # CPU Metrics
    cpu_physical_cores: int = Field(default=4, description="Count of physical CPU cores (excluding HyperThreading/SMT)")
    cpu_logical_cores: int = Field(default=4, description="Count of logical threads")
    
    # Memory Metrics (in GB)
    total_ram_gb: float = Field(default=8.0, description="Total physical system RAM in Gigabytes")
    available_ram_gb: float = Field(default=4.0, description="Currently available RAM in Gigabytes")
    
    # GPU / Accelerator Metrics
    gpu_available: bool = Field(default=False, description="Whether a compatible hardware accelerator is present")
    gpu_type: str = Field(default="none", description="Accelerator category: 'cuda', 'mps', or 'none'")
    gpu_name: Optional[str] = Field(default=None, description="Model identifier of the primary GPU")
    gpu_vram_gb: Optional[float] = Field(default=None, description="Total dedicated VRAM in Gigabytes")
    gpu_device_count: int = Field(default=0, description="Total count of available GPUs")

    # Dynamic Optimization Recommendations
    optimal_ollama_threads: int = Field(default=4, description="Optimal threads for CPU LLM inference (mapped to physical cores)")
    optimal_ocr_threads: int = Field(default=4, description="Optimal worker threads for Docling / PDF OCR extraction")
    optimal_device: str = Field(default="cpu", description="Primary compute device recommendation: 'cuda', 'mps', or 'cpu'")
    optimal_ollama_max_loaded_models: int = Field(default=1, description="Number of models Ollama should keep in cache concurrently")

    def get_ollama_env_recommendations(self) -> Dict[str, str]:
        """
        Generates recommended environment variables for launching the Ollama daemon.
        """
        return {
            "OLLAMA_NUM_PARALLEL": "1",
            "OLLAMA_MAX_LOADED_MODELS": str(self.optimal_ollama_max_loaded_models),
            "OLLAMA_ORIGINS": "*",
            "OLLAMA_NUM_THREADS": str(self.optimal_ollama_threads),
        }

    def summary(self) -> Dict[str, Any]:
        """Provides a structured dictionary summary of hardware metrics."""
        return {
            "cpu": {
                "physical_cores": self.cpu_physical_cores,
                "logical_cores": self.cpu_logical_cores,
                "optimal_ollama_threads": self.optimal_ollama_threads,
                "optimal_ocr_threads": self.optimal_ocr_threads,
            },
            "ram": {
                "total_gb": self.total_ram_gb,
                "available_gb": self.available_ram_gb,
                "recommended_max_models": self.optimal_ollama_max_loaded_models,
            },
            "accelerator": {
                "available": self.gpu_available,
                "type": self.gpu_type,
                "name": self.gpu_name,
                "vram_gb": self.gpu_vram_gb,
                "selected_device": self.optimal_device,
            },
        }


def detect_hardware_profile() -> HardwareProfile:
    """
    Performs introspective detection of host system hardware capabilities
    and computes optimal compute parameters.
    """
    # 1. CPU Detection
    logical_cores = os.cpu_count() or 4
    physical_cores = logical_cores

    try:
        import psutil
        detected_phys = psutil.cpu_count(logical=False)
        if detected_phys and detected_phys > 0:
            physical_cores = detected_phys
        detected_log = psutil.cpu_count(logical=True)
        if detected_log and detected_log > 0:
            logical_cores = detected_log
    except Exception as e:
        logger.debug(f"psutil core detection fallback: {e}")
        # Heuristic fallback: typical consumer hyperthreaded CPU has 2 threads per core
        if logical_cores > 2:
            physical_cores = max(1, logical_cores // 2)

    # 2. RAM Detection
    total_ram = 8.0
    avail_ram = 4.0
    try:
        import psutil
        vmem = psutil.virtual_memory()
        total_ram = round(vmem.total / (1024 ** 3), 2)
        avail_ram = round(vmem.available / (1024 ** 3), 2)
    except Exception as e:
        logger.debug(f"psutil memory detection fallback: {e}")

    # 3. GPU / Accelerator Detection
    gpu_available = False
    gpu_type = "none"
    gpu_name = None
    gpu_vram = None
    device_count = 0

    try:
        import torch

        # Check NVIDIA CUDA
        if torch.cuda.is_available():
            gpu_available = True
            gpu_type = "cuda"
            device_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            gpu_vram = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2)
        # Check Apple Silicon MPS
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            gpu_available = True
            gpu_type = "mps"
            gpu_name = "Apple Silicon MPS"
            device_count = 1
    except Exception as e:
        logger.debug(f"PyTorch accelerator detection fallback: {e}")

    # 4. Compute Optimal Parameters
    # A. Ollama CPU Threads:
    # Rule: Match physical cores to avoid hyperthreading cache thrashing.
    # Allow user override via env var OLLAMA_NUM_THREADS if explicitly defined.
    env_ollama_threads = os.environ.get("OLLAMA_NUM_THREADS")
    if env_ollama_threads and env_ollama_threads.isdigit():
        optimal_ollama_threads = max(1, int(env_ollama_threads))
    else:
        optimal_ollama_threads = max(1, physical_cores)

    # B. OCR Worker Threads:
    # Bound to [1, min(physical_cores, 8)] to avoid choking the machine.
    env_ocr_threads = os.environ.get("OCR_NUM_THREADS")
    if env_ocr_threads and env_ocr_threads.isdigit():
        optimal_ocr_threads = max(1, int(env_ocr_threads))
    else:
        optimal_ocr_threads = max(1, min(physical_cores, 8))

    # C. Primary Device Recommendation:
    env_device = os.environ.get("DEVICE") or os.environ.get("FORCE_DEVICE")
    if env_device and env_device.lower() in ("cuda", "mps", "cpu"):
        optimal_device = env_device.lower()
    else:
        if gpu_available and (gpu_type == "mps" or (gpu_vram is not None and gpu_vram >= 2.0)):
            optimal_device = gpu_type
        else:
            optimal_device = "cpu"

    # D. Ollama Max Loaded Models (Dual-LLM co-residency):
    # 3B (~2.3GB) + 7B (~4.8GB) = ~7.1GB. If system RAM >= 14GB, allow 2 models simultaneously.
    if total_ram >= 14.0:
        optimal_max_models = 2
    else:
        optimal_max_models = 1

    profile = HardwareProfile(
        cpu_physical_cores=physical_cores,
        cpu_logical_cores=logical_cores,
        total_ram_gb=total_ram,
        available_ram_gb=avail_ram,
        gpu_available=gpu_available,
        gpu_type=gpu_type,
        gpu_name=gpu_name,
        gpu_vram_gb=gpu_vram,
        gpu_device_count=device_count,
        optimal_ollama_threads=optimal_ollama_threads,
        optimal_ocr_threads=optimal_ocr_threads,
        optimal_device=optimal_device,
        optimal_ollama_max_loaded_models=optimal_max_models,
    )

    return profile


# Global Singleton Hardware Profile
_GLOBAL_HARDWARE_PROFILE: Optional[HardwareProfile] = None


def get_hardware_profile(refresh: bool = False) -> HardwareProfile:
    """
    Returns the singleton HardwareProfile instance, cached on first call.
    Pass `refresh=True` to force re-detection.
    """
    global _GLOBAL_HARDWARE_PROFILE
    if _GLOBAL_HARDWARE_PROFILE is None or refresh:
        _GLOBAL_HARDWARE_PROFILE = detect_hardware_profile()
        logger.info(
            f"Hardware detected: {len(sys.argv) > 0 and _GLOBAL_HARDWARE_PROFILE.cpu_physical_cores} physical cores, "
            f"{_GLOBAL_HARDWARE_PROFILE.total_ram_gb}GB RAM, "
            f"Device: {_GLOBAL_HARDWARE_PROFILE.optimal_device} "
            f"({_GLOBAL_HARDWARE_PROFILE.gpu_name or 'CPU'}), "
            f"Ollama threads: {_GLOBAL_HARDWARE_PROFILE.optimal_ollama_threads}"
        )
    return _GLOBAL_HARDWARE_PROFILE


# Exported singleton instance
hardware_profile = get_hardware_profile()
