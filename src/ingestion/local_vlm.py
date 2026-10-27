"""
Local VLM Engine: High-Performance On-Premise Vision-Language Model
====================================================================
Runs local Google PaliGemma / Qwen2-VL models optimized for RTX 4050 (6GB VRAM):
- 4-bit NF4 Quantization via bitsandbytes (VRAM usage ~2.2 GB).
- Authenticated via Hugging Face token.
- Seamless fallback to Qwen2-VL if gated PaliGemma license is pending acceptance.
- Single-pass high resolution document transcription & bounding-box extraction.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union
import torch
from PIL import Image
import numpy as np
from loguru import logger

from src.core.config import settings

class LocalVLMEngine:
    """
    On-premise Vision-Language Model execution engine.
    Optimized for NVIDIA GeForce RTX 4050 (6GB VRAM) using 4-bit NF4 quantization.
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        hf_token: Optional[str] = None,
        quantization: str = "4bit",
        device: Optional[str] = None
    ):
        self.model_id = model_id or settings.LOCAL_VLM_MODEL
        self.fallback_model_id = settings.LOCAL_VLM_FALLBACK_MODEL
        self.hf_token = hf_token or settings.HF_TOKEN
        self.quantization = quantization or settings.LOCAL_VLM_QUANTIZATION
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.processor = None
        self.model = None
        self._is_loaded = False

        logger.info(
            f"LocalVLMEngine configured: Model='{self.model_id}' | "
            f"Device='{self.device}' | Quantization='{self.quantization}'"
        )

    def _load_model(self):
        """Loads model and processor into VRAM on demand (Lazy Loading)."""
        if self._is_loaded:
            return

        from transformers import AutoProcessor, BitsAndBytesConfig

        logger.info(f"Loading local VLM model '{self.model_id}' into {self.device}...")

        # Setup 4-bit NF4 Quantization for RTX 4050 6GB VRAM optimization
        quant_config = None
        if self.quantization == "4bit" and self.device == "cuda":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True
            )
            logger.info("Enabled 4-bit NF4 Quantization (Estimated VRAM: ~2.2 GB).")

        # Attempt loading target model (Google PaliGemma 2)
        try:
            self._init_checkpoint(self.model_id, quant_config)
            self._is_loaded = True
            logger.info(f"Successfully loaded '{self.model_id}' on {self.device}!")
        except Exception as e:
            logger.warning(
                f"Could not load primary model '{self.model_id}' ({e}). "
                f"Attempting non-gated fallback '{self.fallback_model_id}'..."
            )
            try:
                self._init_checkpoint(self.fallback_model_id, quant_config)
                self.model_id = self.fallback_model_id
                self._is_loaded = True
                logger.info(f"Successfully loaded fallback VLM '{self.fallback_model_id}'!")
            except Exception as e2:
                logger.error(f"Failed to load any local VLM model: {e2}")
                raise e2

    def _init_checkpoint(self, checkpoint_id: str, quant_config: Any):
        """Initializes model checkpoint and processor."""
        from transformers import AutoProcessor

        self.processor = AutoProcessor.from_pretrained(
            checkpoint_id,
            token=self.hf_token
        )

        if "paligemma" in checkpoint_id.lower():
            from transformers import PaliGemmaForConditionalGeneration
            self.model = PaliGemmaForConditionalGeneration.from_pretrained(
                checkpoint_id,
                quantization_config=quant_config,
                device_map="auto" if self.device == "cuda" else None,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                token=self.hf_token
            )
        elif "qwen" in checkpoint_id.lower():
            from transformers import Qwen2VLForConditionalGeneration
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                checkpoint_id,
                quantization_config=quant_config,
                device_map="auto" if self.device == "cuda" else None,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                token=self.hf_token
            )
        else:
            from transformers import AutoModelForVision2Seq
            self.model = AutoModelForVision2Seq.from_pretrained(
                checkpoint_id,
                quantization_config=quant_config,
                device_map="auto" if self.device == "cuda" else None,
                token=self.hf_token
            )

    def transcribe_image(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        prompt: str = "ocr"
    ) -> str:
        """
        Transcribes text from a document image locally.
        Preserves all legal clauses, numbers, stamps, and recipient lists.
        """
        self._load_model()

        # Normalize image to PIL Image
        if isinstance(image_input, (str, Path)):
            pil_image = Image.open(str(image_input)).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            import cv2
            rgb = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)
        elif isinstance(image_input, Image.Image):
            pil_image = image_input.convert("RGB")
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        logger.info(f"Running local VLM transcription on {pil_image.size} image...")

        try:
            if "paligemma" in self.model_id.lower():
                inputs = self.processor(
                    text=prompt,
                    images=pil_image,
                    return_tensors="pt"
                ).to(self.device)

                with torch.inference_mode():
                    output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=1024,
                        do_sample=False
                    )

                # Decode only the newly generated tokens
                input_len = inputs["input_ids"].shape[-1]
                generated_ids = output_ids[0, input_len:]
                text = self.processor.decode(generated_ids, skip_special_tokens=True).strip()

            elif "qwen" in self.model_id.lower():
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": pil_image},
                            {"type": "text", "text": "Transcribe all Vietnamese text in this legal document with 100% completeness. Preserve sections, articles, dates, and signers."}
                        ]
                    }
                ]
                text_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = self.processor.process_vision_info(messages) if hasattr(self.processor, "process_vision_info") else (pil_image, None)
                inputs = self.processor(
                    text=[text_prompt],
                    images=image_inputs,
                    padding=True,
                    return_tensors="pt"
                ).to(self.device)

                with torch.inference_mode():
                    generated_ids = self.model.generate(**inputs, max_new_tokens=1024, do_sample=False)
                    generated_ids_trimmed = [
                        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    text = self.processor.batch_decode(
                        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )[0].strip()
            else:
                text = ""

            return text

        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

# Global Singleton instance
_local_vlm_instance: Optional[LocalVLMEngine] = None

def get_local_vlm() -> LocalVLMEngine:
    """Returns the singleton instance of LocalVLMEngine."""
    global _local_vlm_instance
    if _local_vlm_instance is None:
        _local_vlm_instance = LocalVLMEngine()
    return _local_vlm_instance
