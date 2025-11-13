"""
Real-time ABCD Pattern Detector for OHLCV Data
"""
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple
import logging
import tempfile
import os
from tensorflow import keras
import mplfinance as mpf

from .config import BacktestConfig

logger = logging.getLogger(__name__)


class ABCDDetector:
    """Real-time ABCD pattern detector"""

    def __init__(
        self,
        model_path: str,
        config: Optional[BacktestConfig] = None
    ):
        """
        Initialize detector

        Args:
            model_path: Path to trained model
            config: Backtest configuration
        """
        self.config = config or BacktestConfig()
        self.model = keras.models.load_model(model_path)
        logger.info(f"Model loaded from {model_path}")

    def prepare_chart_image(
        self,
        df: pd.DataFrame,
        output_path: Optional[str] = None
    ) -> str:
        """
        Prepare chart image from OHLCV data

        Args:
            df: OHLCV DataFrame
            output_path: Output file path (if None, uses temp file)

        Returns:
            Path to saved image
        """
        if output_path is None:
            temp_file = tempfile.NamedTemporaryFile(
                suffix='.png',
                delete=False
            )
            output_path = temp_file.name
            temp_file.close()

        # Configure plot style
        mc = mpf.make_marketcolors(
            up='#00ff00',
            down='#ff0000',
            edge='inherit',
            wick='inherit',
            volume='in'
        )

        s = mpf.make_mpf_style(marketcolors=mc, gridstyle='', y_on_right=False)

        try:
            mpf.plot(
                df,
                type='candle',
                volume=True,
                style=s,
                savefig=dict(
                    fname=output_path,
                    dpi=100,
                    bbox_inches='tight',
                    pad_inches=0.1
                ),
                figsize=(8, 6),
                warn_too_much_data=10000
            )
        except Exception as e:
            logger.error(f"Failed to create chart image: {e}")
            raise

        return output_path

    def load_and_preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Load and preprocess image for model

        Args:
            image_path: Path to image file

        Returns:
            Preprocessed image array
        """
        from tensorflow.keras.preprocessing import image

        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        return img_array

    def detect_pattern(
        self,
        df: pd.DataFrame,
        cleanup: bool = True
    ) -> Tuple[float, bool]:
        """
        Detect ABCD pattern in OHLCV data

        Args:
            df: OHLCV DataFrame
            cleanup: Remove temporary files

        Returns:
            (confidence, is_pattern)
        """
        # Create chart image
        image_path = self.prepare_chart_image(df)

        try:
            # Load and preprocess
            img_array = self.load_and_preprocess_image(image_path)

            # Predict
            prediction = self.model.predict(img_array, verbose=0)
            confidence = float(prediction[0][0])
            is_pattern = confidence >= self.config.detection_threshold

            return confidence, is_pattern

        finally:
            # Cleanup temporary file
            if cleanup and os.path.exists(image_path):
                try:
                    os.remove(image_path)
                except Exception as e:
                    logger.warning(f"Failed to remove temp file {image_path}: {e}")

    def scan_dataframe(
        self,
        df: pd.DataFrame,
        window_size: int = 20,
        step: int = 1
    ) -> List[Dict]:
        """
        Scan OHLCV DataFrame for patterns using sliding window

        Args:
            df: OHLCV DataFrame
            window_size: Size of sliding window
            step: Step size for window

        Returns:
            List of detected patterns with timestamps and confidence
        """
        if len(df) < window_size:
            logger.warning(f"DataFrame too short ({len(df)} < {window_size})")
            return []

        detections = []

        for i in range(0, len(df) - window_size + 1, step):
            window_df = df.iloc[i:i+window_size]

            try:
                confidence, is_pattern = self.detect_pattern(window_df)

                if is_pattern:
                    detection = {
                        'start_idx': i,
                        'end_idx': i + window_size - 1,
                        'start_time': window_df.index[0],
                        'end_time': window_df.index[-1],
                        'confidence': confidence,
                        'price_at_detection': window_df['Close'].iloc[-1]
                    }
                    detections.append(detection)
                    logger.info(f"Pattern detected at {detection['end_time']}, "
                              f"confidence: {confidence:.3f}")

            except Exception as e:
                logger.error(f"Error detecting pattern at index {i}: {e}")
                continue

        logger.info(f"Scan complete: {len(detections)} patterns detected")
        return detections

    def detect_latest(
        self,
        df: pd.DataFrame,
        lookback: int = 20
    ) -> Optional[Dict]:
        """
        Detect pattern in the most recent bars

        Args:
            df: OHLCV DataFrame
            lookback: Number of recent bars to analyze

        Returns:
            Detection dictionary or None
        """
        if len(df) < lookback:
            logger.warning(f"Not enough data ({len(df)} < {lookback})")
            return None

        window_df = df.iloc[-lookback:]

        try:
            confidence, is_pattern = self.detect_pattern(window_df)

            if is_pattern:
                detection = {
                    'timestamp': window_df.index[-1],
                    'confidence': confidence,
                    'price': window_df['Close'].iloc[-1],
                    'lookback_period': lookback
                }
                return detection

            return None

        except Exception as e:
            logger.error(f"Error in latest detection: {e}")
            return None

    def get_pattern_statistics(
        self,
        detections: List[Dict]
    ) -> Dict:
        """
        Calculate statistics from detections

        Args:
            detections: List of detection dictionaries

        Returns:
            Statistics dictionary
        """
        if not detections:
            return {
                'count': 0,
                'avg_confidence': 0.0,
                'max_confidence': 0.0,
                'min_confidence': 0.0
            }

        confidences = [d['confidence'] for d in detections]

        stats = {
            'count': len(detections),
            'avg_confidence': np.mean(confidences),
            'max_confidence': np.max(confidences),
            'min_confidence': np.min(confidences),
            'std_confidence': np.std(confidences)
        }

        return stats

    def filter_overlapping_detections(
        self,
        detections: List[Dict],
        min_separation: int = 10
    ) -> List[Dict]:
        """
        Filter out overlapping detections

        Args:
            detections: List of detections
            min_separation: Minimum bars between detections

        Returns:
            Filtered list of detections
        """
        if not detections:
            return []

        # Sort by confidence
        sorted_detections = sorted(
            detections,
            key=lambda x: x['confidence'],
            reverse=True
        )

        filtered = []
        for detection in sorted_detections:
            # Check if too close to existing detections
            too_close = False
            for existing in filtered:
                if abs(detection['end_idx'] - existing['end_idx']) < min_separation:
                    too_close = True
                    break

            if not too_close:
                filtered.append(detection)

        # Sort by time
        filtered.sort(key=lambda x: x['end_idx'])

        logger.info(f"Filtered {len(detections)} -> {len(filtered)} detections")
        return filtered
