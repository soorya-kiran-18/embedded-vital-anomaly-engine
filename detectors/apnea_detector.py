from detectors.base_tflite_detector import BaseTFLiteDetector


class ApneaDetector(BaseTFLiteDetector):
    """Wrap the existing apnea TFLite model in the common detector interface."""

    def __init__(
        self,
        model_path: str = "models/apnea_detector_model.tflite",
        stats_path: str = "models/apnea_norm_stats.npz",
        dataset_fallback_path: str = "X.npy",
    ) -> None:
        super().__init__(
            model_path=model_path,
            stats_path=stats_path,
            dataset_fallback_path=dataset_fallback_path,
            window_size=30,
        )
