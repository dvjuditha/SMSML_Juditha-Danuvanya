import time
import psutil
from prometheus_client import start_http_server, Counter, Gauge, Summary, Histogram

# Kriteria Advanced: 10 Metrics
# 1. Total Predictions
MODEL_PREDICTIONS_TOTAL = Counter('model_predictions_total', 'Total number of predictions made by the model', ['predicted_class'])
# 2. Total Errors
MODEL_ERRORS_TOTAL = Counter('model_errors_total', 'Total number of prediction errors')
# 3. Request Duration Summary
REQUEST_DURATION_SUMMARY = Summary('request_duration_seconds_summary', 'Time spent processing request summary')
# 4. Latency Histogram
PREDICTION_LATENCY_HISTOGRAM = Histogram('prediction_latency_seconds_histogram', 'Time spent predicting histogram', buckets=(0.01, 0.05, 0.1, 0.5, 1.0))
# 5. Active Requests Gauge
ACTIVE_REQUESTS = Gauge('active_requests_gauge', 'Number of currently active requests')
# 6. CPU Usage Gauge
SYSTEM_CPU_USAGE = Gauge('system_cpu_usage_percent', 'System CPU utilization as a percentage')
# 7. Memory Usage Gauge
SYSTEM_MEMORY_USAGE = Gauge('system_memory_usage_percent', 'System Memory utilization as a percentage')
# 8. Input Features Count
INPUT_FEATURES_COUNT = Counter('input_features_count_total', 'Total number of features processed')
# 9. Model Version Gauge
MODEL_VERSION = Gauge('model_version_info', 'Current model version', ['version'])
# 10. Memory Available
SYSTEM_MEMORY_AVAILABLE = Gauge('system_memory_available_bytes', 'System Memory available in bytes')

class MetricsCollector:
    def __init__(self, port=8000):
        self.port = port

    def start(self):
        start_http_server(self.port)
        print(f"Prometheus metrics exposed at http://localhost:{self.port}/metrics")
        MODEL_VERSION.labels(version="1.0.0").set(1)
        
        while True:
            self.collect_system_metrics()
            time.sleep(5)

    def collect_system_metrics(self):
        SYSTEM_CPU_USAGE.set(psutil.cpu_percent())
        mem = psutil.virtual_memory()
        SYSTEM_MEMORY_USAGE.set(mem.percent)
        SYSTEM_MEMORY_AVAILABLE.set(mem.available)

if __name__ == '__main__':
    collector = MetricsCollector(port=8000)
    collector.start()
