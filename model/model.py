import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import numpy as np
import onnxruntime as ort

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelLoadError(Exception):
    """Custom exception for model loading errors."""
    pass


class PredictionError(Exception):
    """Custom exception for prediction errors."""
    pass


class ONNXModel:
    """
    A wrapper class for ONNX model inference.
    
    This class handles loading and running inference on ONNX models,
    providing a clean interface for model operations.
    """
    
    def __init__(self, model_path: Optional[Path] = None) -> None:
        """
        Initialize the ONNX model wrapper.
        
        Args:
            model_path: Optional path to ONNX model file. If None, searches in assets directory.
            
        Raises:
            ModelLoadError: If model cannot be loaded.
        """
        self._session: Optional[ort.InferenceSession] = None
        self._model_path: Optional[Path] = None
        
        self._load_model(model_path)

    def _find_model_path(self) -> Path:
        """
        Find ONNX model file in the assets directory.
        
        Returns:
            Path to the first ONNX file found.
            
        Raises:
            ModelLoadError: If no ONNX files are found.
        """
        current_dir = Path(__file__).resolve().parent
        assets_dir = current_dir.parent / "assets"
        
        if not assets_dir.exists():
            raise ModelLoadError(f"Assets directory not found: {assets_dir}")
        
        model_files = list(assets_dir.glob("*.onnx"))
        
        if not model_files:
            raise ModelLoadError(f"No .onnx model files found in {assets_dir}")
        
        logger.info(f"Found ONNX model files: {[f.name for f in model_files]}")
        return model_files[0]

    def _load_model(self, model_path: Optional[Path] = None) -> None:
        """
        Load the ONNX model from the specified path.
        
        Args:
            model_path: Path to ONNX model file. If None, searches in assets directory.
            
        Raises:
            ModelLoadError: If model cannot be loaded.
        """
        try:
            # Determine model path
            if model_path is None:
                model_path = self._find_model_path()
            elif not model_path.exists():
                raise ModelLoadError(f"Model file not found: {model_path}")
            
            self._model_path = model_path
            
            # Create ONNX Runtime session
            self._session = ort.InferenceSession(
                str(model_path),
                providers=['CPUExecutionProvider']
            )
            
            self._log_model_info()
            logger.info(f"ONNX model loaded successfully: {model_path.name}")
            
        except ImportError as e:
            error_msg = "onnxruntime not installed. Install with: pip install onnxruntime"
            logger.error(error_msg)
            raise ModelLoadError(error_msg) from e
            
        except Exception as e:
            error_msg = f"Failed to load ONNX model: {e}"
            logger.error(error_msg, exc_info=True)
            raise ModelLoadError(error_msg) from e

    def _log_model_info(self) -> None:
        """Log basic model information."""
        if self._session is None:
            return
            
        try:
            input_info = self._session.get_inputs()[0]
            output_info = self._session.get_outputs()[0]
            
            logger.info(f"Model input - Name: {input_info.name}, Shape: {input_info.shape}")
            logger.info(f"Model output - Name: {output_info.name}, Shape: {output_info.shape}")
            
        except Exception as e:
            logger.warning(f"Could not log model info: {e}")

    def _validate_input(self, input_data: Union[np.ndarray, Any]) -> np.ndarray:
        """
        Validate and prepare input data for prediction.
        
        Args:
            input_data: Input data to validate and convert.
            
        Returns:
            Validated numpy array with correct dtype.
            
        Raises:
            PredictionError: If input validation fails.
        """
        try:
            # Convert to numpy array if needed
            if hasattr(input_data, 'numpy'):  # TensorFlow tensor
                input_data = input_data.numpy()
            elif not isinstance(input_data, np.ndarray):
                input_data = np.array(input_data)
            
            # Ensure correct data type
            input_data = input_data.astype(np.float32)
            
            return input_data
            
        except Exception as e:
            raise PredictionError(f"Input validation failed: {e}") from e

    def _format_output(self, raw_outputs: List[np.ndarray]) -> Dict[str, List[float]]:
        """
        Format model outputs into a structured dictionary.
        
        Args:
            raw_outputs: Raw outputs from ONNX model.
            
        Returns:
            Formatted output dictionary.
        """
        # Assuming first output contains the predictions
        outputs_array = raw_outputs[0][0]
        
        return {
            str(i): outputs_array[i].tolist() 
            for i in range(len(outputs_array))
        }

    def predict(self, input_data: Union[np.ndarray, Any]) -> Dict[str, List[float]]:
        """
        Make a prediction using the loaded ONNX model.

        Args:
            input_data: Input data for prediction. Should be compatible with model input shape.
            
        Returns:
            Dictionary containing model predictions.
            
        Raises:
            ModelLoadError: If model is not loaded.
            PredictionError: If prediction fails.
        """
        if not self.is_loaded:
            raise ModelLoadError(
                "ONNX model is not loaded. Ensure model file exists and can be loaded."
            )
        
        try:
            # Validate and prepare input
            validated_input = self._validate_input(input_data)
            
            # Get input name and run inference
            input_name = self._session.get_inputs()[0].name
            raw_outputs = self._session.run(None, {input_name: validated_input})
            
            # Format and return outputs
            return self._format_output(raw_outputs)

        except PredictionError:
            raise  # Re-raise prediction errors as-is
        except Exception as e:
            error_msg = f"Prediction failed: {e}"
            logger.error(error_msg, exc_info=True)
            raise PredictionError(error_msg) from e

    def print_model_summary(self) -> None:
        """Print detailed information about the loaded model."""
        if not self.is_loaded:
            print("No ONNX model loaded.")
            return

        print("ONNX Model Summary")
        print("=" * 50)
        print(f"Model file: {self._model_path.name if self._model_path else 'Unknown'}")
        print()
        
        # Input information
        print("Inputs:")
        for i, input_detail in enumerate(self._session.get_inputs(), 1):
            print(f"  {i}. Name: {input_detail.name}")
            print(f"     Shape: {input_detail.shape}")
            print(f"     Type: {input_detail.type}")
            print()
        
        # Output information
        print("Outputs:")
        for i, output_detail in enumerate(self._session.get_outputs(), 1):
            print(f"  {i}. Name: {output_detail.name}")
            print(f"     Shape: {output_detail.shape}")
            print(f"     Type: {output_detail.type}")
            print()
            
        print(f"Execution Providers: {', '.join(self._session.get_providers())}")

    @property
    def is_loaded(self) -> bool:
        """Check if the model is successfully loaded."""
        return self._session is not None

    @property
    def input_details(self) -> Optional[Dict[str, Any]]:
        """Get input details of the loaded model."""
        if not self.is_loaded:
            return None
            
        input_detail = self._session.get_inputs()[0]
        return {
            'name': input_detail.name,
            'shape': input_detail.shape,
            'type': input_detail.type
        }

    @property
    def output_details(self) -> Optional[Dict[str, Any]]:
        """Get output details of the loaded model."""
        if not self.is_loaded:
            return None
            
        output_detail = self._session.get_outputs()[0]
        return {
            'name': output_detail.name,
            'shape': output_detail.shape,
            'type': output_detail.type
        }

    @property
    def model_path(self) -> Optional[Path]:
        """Get the path to the loaded model file."""
        return self._model_path


# Maintain backwards compatibility
Model = ONNXModel