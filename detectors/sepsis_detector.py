from detectors.base_tflite_detector import BaseTFLiteDetector


class SepsisDetector(BaseTFLiteDetector):
    def __init__(
        self,
        model_path: str = "models/sepsis_student.tflite",
        stats_path: str = "models/sepsis_norm_stats.npz",
        dataset_fallback_path: str = "models/sepsis_X.npy",
    ) -> None:
        super().__init__(
            model_path=model_path,
            stats_path=stats_path,
            dataset_fallback_path=dataset_fallback_path,
            window_size=60,
        )
