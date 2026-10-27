"""
Unit & Integration Tests for Centralized Hardware Detection & Compute Profile
============================================================================
Validates:
- Real hardware detection (CPU cores, RAM, GPU/CUDA)
- Auto-tuning recommendations for Ollama threads, OCR threads, and max loaded models
- Environment variable overrides
- Resilient fallback logic when dependencies or accelerators are absent
- Central integration with Settings and DoclingParser
"""

from unittest.mock import patch

from src.core.config import settings
from src.core.hardware import (
    HardwareProfile,
    detect_hardware_profile,
    get_hardware_profile,
    hardware_profile,
)
from src.ingestion.docling_parser import DoclingParser


def test_real_hardware_detection_profile():
    """Verify real system detection succeeds and outputs valid metrics."""
    profile = detect_hardware_profile()
    assert isinstance(profile, HardwareProfile)
    assert profile.cpu_physical_cores >= 1
    assert profile.cpu_logical_cores >= profile.cpu_physical_cores
    assert profile.total_ram_gb > 0.0
    assert profile.optimal_ollama_threads >= 1
    assert profile.optimal_ocr_threads >= 1
    assert profile.optimal_device in ("cuda", "mps", "cpu")
    assert profile.optimal_ollama_max_loaded_models in (1, 2)

    summary = profile.summary()
    assert "cpu" in summary
    assert "ram" in summary
    assert "accelerator" in summary


def test_hardware_singleton_caching():
    """Ensure get_hardware_profile returns the cached singleton."""
    p1 = get_hardware_profile()
    p2 = get_hardware_profile()
    assert p1 is p2
    assert p1 is hardware_profile


def test_ollama_env_recommendations_structure():
    """Verify dictionary formatting of recommended Ollama environment variables."""
    profile = get_hardware_profile()
    env_vars = profile.get_ollama_env_recommendations()

    assert "OLLAMA_NUM_PARALLEL" in env_vars
    assert env_vars["OLLAMA_NUM_PARALLEL"] == "1"
    assert "OLLAMA_MAX_LOADED_MODELS" in env_vars
    assert "OLLAMA_ORIGINS" in env_vars
    assert "OLLAMA_NUM_THREADS" in env_vars
    assert int(env_vars["OLLAMA_NUM_THREADS"]) == profile.optimal_ollama_threads


def test_hardware_env_var_overrides(monkeypatch):
    """Verify that manual environment variables take precedence when user overrides."""
    monkeypatch.setenv("OLLAMA_NUM_THREADS", "14")
    monkeypatch.setenv("OCR_NUM_THREADS", "6")
    monkeypatch.setenv("DEVICE", "cpu")

    custom_profile = detect_hardware_profile()
    assert custom_profile.optimal_ollama_threads == 14
    assert custom_profile.optimal_ocr_threads == 6
    assert custom_profile.optimal_device == "cpu"


def test_hardware_fallback_without_accelerator_and_psutil():
    """Verify fallback heuristics when psutil raises and GPU is absent."""
    with (
        patch("psutil.cpu_count", side_effect=RuntimeError("psutil unavailable")),
        patch("psutil.virtual_memory", side_effect=RuntimeError("vmem unavailable")),
        patch("torch.cuda.is_available", return_value=False),
        patch("os.cpu_count", return_value=8),
    ):
        profile = detect_hardware_profile()
        # Hyperthreading heuristic: 8 // 2 = 4
        assert profile.cpu_physical_cores in (4, 8)
        assert profile.optimal_device == "cpu"
        assert profile.gpu_available is False
        assert profile.optimal_ollama_threads >= 1


def test_settings_hardware_integration():
    """Ensure Settings loads compute parameters directly from the centralized profile."""
    assert settings.OLLAMA_NUM_THREADS == hardware_profile.optimal_ollama_threads
    assert settings.OCR_NUM_THREADS == hardware_profile.optimal_ocr_threads
    assert settings.DEVICE == hardware_profile.optimal_device
    assert hasattr(settings, "hardware")
    assert settings.hardware.optimal_ollama_threads == hardware_profile.optimal_ollama_threads


def test_docling_parser_adopts_central_profile():
    """Ensure DoclingParser defaults to centralized profile when device='auto' and threads=None."""
    parser = DoclingParser(device="auto", num_threads=None)
    assert parser.num_threads == hardware_profile.optimal_ocr_threads
    assert parser.device == hardware_profile.optimal_device
