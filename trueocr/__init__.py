"""TrueOCR: OCR tiếng Việt trên CPU với ONNX Runtime."""

from .ocr import OCRConfig, OCRResult, TrueOCR, load_model
from .recognizer import Prediction, Recognizer

__all__ = ["OCRConfig", "OCRResult", "Prediction", "Recognizer", "TrueOCR", "load_model"]
__version__ = "0.1.0"
