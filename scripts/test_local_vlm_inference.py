"""
Local VLM Inference & Benchmark Runner
======================================
Tests local Hugging Face Vision-Language Model execution on NVIDIA RTX 4050 (6GB VRAM)
using 4-bit NF4 Quantization.
"""

import sys
import time
from pathlib import Path
import torch
from loguru import logger

from src.core.config import settings
from src.ingestion.local_vlm import LocalVLMEngine

def main():
    logger.info("=== LexiGuard Local VLM Benchmark & Verification ===")
    
    # 1. Hardware Inspection
    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"
    total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3) if cuda_available else 0
    
    print(f"[*] GPU Acceleration: {cuda_available}")
    print(f"[*] Device Name:      {device_name}")
    print(f"[*] Total VRAM:       {total_vram:.2f} GB")
    print(f"[*] HF Account:       {settings.HF_USERNAME}")
    print(f"[*] Target Model:     {settings.LOCAL_VLM_MODEL}")
    print(f"[*] Quantization:     {settings.LOCAL_VLM_QUANTIZATION} (bitsandbytes NF4)")
    print("-" * 60)

    # 2. Select Sample Image
    sample_images = [
        Path("data/sample_contracts/SCAN CV GUI BO VV TU VAN PHAP LUAT.jpg"),
        Path("data/sample_contracts/luu-ban-nhap-tu-dong-5.jpg")
    ]
    test_image = None
    for p in sample_images:
        if p.exists():
            test_image = p
            break

    if not test_image:
        print("[!] No sample contract image found in data/sample_contracts/")
        return

    print(f"[*] Processing Sample Image: {test_image.name}")
    
    # 3. Initialize & Load Local VLM
    engine = LocalVLMEngine(
        model_id=settings.LOCAL_VLM_MODEL,
        quantization="4bit"
    )

    print("[*] Loading model weights into VRAM (with 4-bit NF4 quantization)...")
    t0 = time.time()
    try:
        engine._load_model()
        load_time = time.time() - t0
        allocated_vram = torch.cuda.memory_allocated(0) / (1024**3) if cuda_available else 0
        print(f"[+] Model loaded successfully in {load_time:.2f}s!")
        print(f"[+] VRAM Footprint: {allocated_vram:.2f} GB / {total_vram:.2f} GB")
        print(f"[+] Active Model:  {engine.model_id}")
    except Exception as e:
        print(f"[-] Model loading failed: {e}")
        return

    # 4. Run Image Transcription
    print("-" * 60)
    print("[*] Running local OCR / transcription inference...")
    t_start = time.time()
    extracted_text = engine.transcribe_image(test_image)
    infer_time = time.time() - t_start

    print(f"[+] Inference completed in {infer_time:.2f}s!")
    print("=" * 60)
    print("EXTRACTED TEXT:")
    print("=" * 60)
    print(extracted_text)
    print("=" * 60)

if __name__ == "__main__":
    main()
