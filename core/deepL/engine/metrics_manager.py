import numpy as np

class MetricsManager:
    def __init__(self):
        self._metrics: dict[str, list[float]] = {}
        self._keys: list[str] = []
    
    def register_metric(self, key: str, metric: float):
        self._metrics[key] = []
        self._keys.append(key)
        self._metrics[key].append(metric)
    
    def update_metric(self, key: str, metric: float):
        if key not in self._keys:
            self.register_metric(key, metric)
            return
        self._metrics[key].append(metric)
    
    def update(self, metric_dict: dict[str, float]):
        for key, metric in metric_dict.items():
            self.update_metric(key, metric)
    
    def get_metric_mean_std(self, key: str, threshold_key: str = None, filter_func = None) -> tuple[float, float]:
        assert key in self._keys, f"Key '{key}' not found."
        filter_metrics = self._metrics
        if threshold_key is not None and filter_func is not None:
            filter_metrics = self.filter_metrics(threshold_key, [key], filter_func)
        return np.mean(filter_metrics[key]), np.std(filter_metrics[key])

    def get_metric_mean(self, key: str, threshold_key: str = None, filter_func = None) -> float:
        assert key in self._keys, f"Key '{key}' not found."
        filter_metrics = self._metrics
        if threshold_key is not None and filter_func is not None:
            filter_metrics = self.filter_metrics(threshold_key, [key], filter_func)
        return np.mean(filter_metrics[key])

    def get_metrics_mean_std(self, keys: list[str] = None, threshold_key: str = None, filter_func = None) -> dict[str, tuple[float, float]]:
        if keys is None:
            keys = self._keys
        assert len(keys) > 0, "At least one key must be provided."
        filter_metrics = self._metrics
        mean_std_metrics = {}
        if threshold_key is not None and filter_func is not None:
            filter_metrics = self.filter_metrics(threshold_key, keys, filter_func)
            mean_std_metrics['RR'] = filter_metrics['RR']
            mean_std_metrics['threshold_key'] = threshold_key
            mean_std_metrics['filter_func'] = filter_func.threshold
        for key in keys:
            assert key in self._keys, f"Key '{key}' not found."
            if np.asarray(filter_metrics[key]).size == 0:
                mean_std_metrics[key] = "NoData"
            else:
                mean_std_metrics[key] = (np.mean(filter_metrics[key]), np.std(filter_metrics[key]))
        return mean_std_metrics

    def get_metrics_mean(self, keys: list[str] = None, threshold_key: str = None, filter_func = None) -> dict[str, float]:
        if keys is None:
            keys = self._keys
        assert len(keys) > 0, "At least one key must be provided."
        filter_metrics = self._metrics
        mean_metrics = {}
        if threshold_key is not None and filter_func is not None:
            filter_metrics = self.filter_metrics(threshold_key, keys, filter_func)
            mean_metrics['RR'] = filter_metrics['RR']
            mean_metrics['threshold_key'] = threshold_key
            mean_metrics['filter_func'] = filter_func.threshold
        for key in keys:
            assert key in self._keys, f"Key '{key}' not found."
            if np.asarray(filter_metrics[key]).size == 0:
                mean_metrics[key] = "NoData"
            else:
                mean_metrics[key] = np.mean(filter_metrics[key])
        return mean_metrics

    def get_metrics_median(self, keys: list[str] = None, threshold_key: str = None, filter_func = None) -> dict[str, float]:
        if keys is None:
            keys = self._keys
        assert len(keys) > 0, "At least one key must be provided."
        filter_metrics = self._metrics
        median_metrics = {}
        if threshold_key is not None and filter_func is not None:
            filter_metrics = self.filter_metrics(threshold_key, keys, filter_func)
            median_metrics['RR'] = filter_metrics['RR']
            median_metrics['threshold_key'] = threshold_key
            median_metrics['filter_func'] = filter_func.threshold
        for key in keys:
            assert key in self._keys, f"Key '{key}' not found."
            if np.asarray(filter_metrics[key]).size == 0:
                median_metrics[key] = "NoData"
            else:
                median_metrics[key] = np.median(filter_metrics[key])
        return median_metrics

    def get_metrics_mean_std_median(self, keys: list[str] = None, threshold_key: str = None, filter_func = None) -> dict[str, tuple[float, float, float]]:
        if keys is None:
            keys = self._keys
        assert len(keys) > 0, "At least one key must be provided."
        filter_metrics = self._metrics
        mean_std_median_metrics = {}
        if threshold_key is not None and filter_func is not None:
            filter_metrics = self.filter_metrics(threshold_key, keys, filter_func)
            mean_std_median_metrics['RR'] = filter_metrics['RR']
            mean_std_median_metrics['threshold_key'] = threshold_key
            mean_std_median_metrics['filter_func'] = filter_func.threshold
        for key in keys:
            assert key in self._keys, f"Key '{key}' not found."
            if np.asarray(filter_metrics[key]).size == 0:
                mean_std_median_metrics[key] = "NoData"
            else:
                mean_std_median_metrics[key] = (np.mean(filter_metrics[key]), np.std(filter_metrics[key]), np.median(filter_metrics[key]))
        return mean_std_median_metrics
        
    def filter_metrics(self, threshold_key: str, keys: list[str] = None, filter_func = None) -> dict[str, np.ndarray]:
        if keys is None:
            keys = self._keys
        assert len(keys) > 0, "At least one key must be provided."
        assert threshold_key in self._keys, f"Threshold key '{threshold_key}' not found."
        assert filter_func is not None, "Filter function must be provided."
        filter_indices = np.where(filter_func(np.array(self._metrics[threshold_key])))[0]
        filter_metrics = {}
        filter_metrics['RR'] = (len(filter_indices) / len(self._metrics[threshold_key])) * 100
        for key in keys:
            assert key in self._keys, f"Key '{key}' not found."
            filter_metrics[key] = np.array(self._metrics[key])[filter_indices]
        return filter_metrics
    
    def get_metrics(self):
        return self._metrics
    
    def clear(self):
        self._metrics = {}
        self._keys = []