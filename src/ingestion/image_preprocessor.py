"""
Image Preprocessor: Intelligent Preprocessing & Multi-Page Stitching Splitter
=============================================================================
Prepares noisy, low-resolution, or stitched document images for optimal OCR:
- Auto-detects multi-page stitched collages (horizontal & vertical splits).
- Upscales low-DPI images using edge-preserving interpolation.
- Enhances contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).
- Mitigates Moiré pattern screen noise and shadows.
"""

from pathlib import Path
from typing import List, Tuple, Union
import cv2
import numpy as np
from PIL import Image
from loguru import logger

class ImagePreprocessor:
    """Intelligent document image enhancer and multi-page segmentation preprocessor."""

    def __init__(self, target_dpi_width: int = 2000):
        self.target_dpi_width = target_dpi_width
        logger.info(f"ImagePreprocessor initialized (Target width: {self.target_dpi_width}px).")

    def process(self, image_path: Union[str, Path]) -> List[Tuple[np.ndarray, dict]]:
        """
        Loads an image, analyzes its aspect ratio, splits multi-page collages if present,
        and applies super-resolution enhancement and contrast normalization.
        Returns a list of (processed_cv2_image, metadata_dict) for each detected sub-page.
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        # Read image with OpenCV (handles Vietnamese/Unicode paths safely)
        img_array = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError(f"Could not decode image at {path}")

        h, w = img.shape[:2]
        aspect_ratio = w / float(h)
        logger.info(f"Loaded image: {path.name} | Shape: {w}x{h} px | Aspect Ratio: {aspect_ratio:.2f}")

        sub_pages: List[Tuple[np.ndarray, dict]] = []

        # 1. Check for Horizontal Multi-Page Collage (e.g. 3 pages side-by-side, ratio > 1.4)
        if aspect_ratio >= 1.4:
            # Determine number of pages (e.g. 2 or 3 pages side-by-side)
            num_pages = 3 if aspect_ratio >= 1.8 else 2
            logger.info(f"Detected horizontal panoramic collage ({num_pages} pages side-by-side). Splitting...")
            page_w = w // num_pages
            for i in range(num_pages):
                x_start = i * page_w
                x_end = w if i == num_pages - 1 else (i + 1) * page_w
                crop = img[:, x_start:x_end]
                enhanced = self.enhance_page(crop)
                sub_pages.append((
                    enhanced,
                    {
                        "source": path.name,
                        "sub_page_index": i + 1,
                        "total_sub_pages": num_pages,
                        "crop_bbox": [x_start, 0, x_end, h],
                        "split_type": "horizontal"
                    }
                ))

        # 2. Check for Vertical Multi-Page Collage (e.g. 2 pages stacked vertically, ratio < 0.65)
        elif aspect_ratio <= 0.65 and h >= 550:
            # Check for horizontal dividing line / page break in middle
            mid_start = int(h * 0.45)
            mid_end = int(h * 0.55)
            gray = cv2.cvtColor(img[mid_start:mid_end, :], cv2.COLOR_BGR2GRAY)
            # Find row with maximum whitespace or boundary
            row_means = np.mean(gray, axis=1)
            split_rel_y = np.argmax(row_means)
            split_y = mid_start + split_rel_y

            logger.info(f"Detected vertical stacked collage (2 pages). Splitting at Y={split_y}...")
            # Page 1 (Top)
            top_crop = img[:split_y, :]
            sub_pages.append((
                self.enhance_page(top_crop),
                {
                    "source": path.name,
                    "sub_page_index": 1,
                    "total_sub_pages": 2,
                    "crop_bbox": [0, 0, w, split_y],
                    "split_type": "vertical"
                }
            ))
            # Page 2 (Bottom)
            bottom_crop = img[split_y:, :]
            sub_pages.append((
                self.enhance_page(bottom_crop),
                {
                    "source": path.name,
                    "sub_page_index": 2,
                    "total_sub_pages": 2,
                    "crop_bbox": [0, split_y, w, h],
                    "split_type": "vertical"
                }
            ))

        # 3. Standard Single Page
        else:
            enhanced = self.enhance_page(img)
            sub_pages.append((
                enhanced,
                {
                    "source": path.name,
                    "sub_page_index": 1,
                    "total_sub_pages": 1,
                    "crop_bbox": [0, 0, w, h],
                    "split_type": "none"
                }
            ))

        logger.info(f"Preprocessing completed for {path.name}: {len(sub_pages)} page(s) ready for OCR.")
        return sub_pages

    def enhance_page(self, img: np.ndarray) -> np.ndarray:
        """
        Enhances document image:
        - Upscales resolution if width < target_dpi_width.
        - Applies CLAHE contrast enhancement on luminance.
        - De-noises high-frequency Moiré artifacts.
        """
        h, w = img.shape[:2]

        # 1. Upscale if image is too small for character recognition
        if w < self.target_dpi_width:
            scale_factor = min(self.target_dpi_width / float(w), 3.0)
            new_w = int(w * scale_factor)
            new_h = int(h * scale_factor)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        # 2. Contrast & Lighting Normalization (CLAHE in LAB color space)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_l = clahe.apply(l_chan)
        lab_enhanced = cv2.merge((enhanced_l, a_chan, b_chan))
        enhanced_bgr = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

        # 3. Mild Bilateral Filter to smooth Moiré raster noise while preserving sharp font edges
        smoothed = cv2.bilateralFilter(enhanced_bgr, d=5, sigmaColor=35, sigmaSpace=35)

        return smoothed

    def evaluate_quality(self, img: np.ndarray) -> dict:
        """
        Assesses document image quality to route to Branch 1 (Full Docling) or Branch 2 (Gemini VLM).
        Metrics evaluated:
        - Sharpness: Laplacian variance (blur detection).
        - Contrast: Standard deviation of grayscale pixel intensities.
        - Resolution: Pixel dimensions (w, h).
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        h, w = gray.shape[:2]
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        contrast_std = float(gray.std())

        is_sharp = laplacian_var >= 80.0
        is_high_contrast = contrast_std >= 35.0
        is_sufficient_res = w >= 1200 and h >= 1600

        is_high_quality = bool(is_sharp and is_high_contrast and is_sufficient_res)

        reason = (
            "High resolution, sharp focus, and sufficient contrast"
            if is_high_quality
            else f"Degraded image: blur_var={laplacian_var:.1f} (min 80), contrast={contrast_std:.1f} (min 35), res={w}x{h} (min 1200x1600)"
        )

        return {
            "is_high_quality": is_high_quality,
            "blur_score": laplacian_var,
            "contrast_score": contrast_std,
            "resolution": (w, h),
            "reason": reason
        }

    def save_enhanced_image(self, img: np.ndarray, output_path: Union[str, Path]) -> Path:
        """Saves processed image with Unicode support."""
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        is_success, buffer = cv2.imencode(out_p.suffix, img)
        if is_success:
            with open(out_p, "wb") as f:
                f.write(buffer)
        return out_p
