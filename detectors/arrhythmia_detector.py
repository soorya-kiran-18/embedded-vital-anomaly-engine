from detectors.base_tflite_detector import BaseTFLiteDetector


class ArrhythmiaDetector(BaseTFLiteDetector):
    def __init__(
        self,
        model_path: str = "models/arrhythmia_student.tflite",
        stats_path: str = "models/arrhythmia_norm_stats.npz",
        dataset_fallback_path: str = "models/arrhythmia_X.npy",
    ) -> None:
        super().__init__(
            model_path=model_path,
            stats_path=stats_path,
            dataset_fallback_path=dataset_fallback_path,
            window_size=16,
        )
