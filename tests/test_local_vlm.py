"""
Unit Tests for LocalVLMEngine (On-Premise VLM)
"""

import pytest
import torch
from src.ingestion.local_vlm import LocalVLMEngine, get_local_vlm
from src.core.config import settings

def test_local_vlm_engine_initialization():
    engine = LocalVLMEngine()
    assert engine is not None
    assert engine.hf_token == settings.HF_TOKEN
    assert engine.quantization == "4bit"
    if torch.cuda.is_available():
        assert engine.device == "cuda"

def test_get_local_vlm_singleton():
    engine1 = get_local_vlm()
    engine2 = get_local_vlm()
    assert engine1 is engine2
