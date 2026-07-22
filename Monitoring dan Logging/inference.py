import time
import random
import argparse
from prometheus_exporter import (
    MetricsCollector,
    MODEL_PREDICTIONS_TOTAL,
    MODEL_ERRORS_TOTAL,
    REQUEST_DURATION_SUMMARY,
    PREDICTION_LATENCY_HISTOGRAM,
    ACTIVE_REQUESTS,
    INPUT_FEATURES_COUNT
)

def run_inference_simulation(num_requests, interval):
    print(f"Starting inference simulation for {num_requests} requests...")
    for i in range(num_requests):
        ACTIVE_REQUESTS.inc()
        start_time = time.time()
        
        try:
            # Simulate prediction logic (Breast Cancer: 30 features)
            INPUT_FEATURES_COUNT.inc(30)
            
            # Simulate latency
            processing_time = random.uniform(0.01, 0.05)
            time.sleep(processing_time)
            
            # Simulate prediction (0 for Benign, 1 for Malignant)
            pred = random.choice(['Benign', 'Malignant'])
            MODEL_PREDICTIONS_TOTAL.labels(predicted_class=pred).inc()
            
            # 1% chance of error
            if random.random() < 0.01:
                raise Exception("Simulated Model Error")
                
        except Exception as e:
            MODEL_ERRORS_TOTAL.inc()
            print(f"Error: {e}")
            
        finally:
            duration = time.time() - start_time
            REQUEST_DURATION_SUMMARY.observe(duration)
            PREDICTION_LATENCY_HISTOGRAM.observe(duration)
            ACTIVE_REQUESTS.dec()
            
        time.sleep(interval)
        if (i+1) % 10 == 0:
            print(f"Processed {i+1}/{num_requests} requests...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-requests", type=int, default=5000)
    parser.add_argument("--interval", type=float, default=0.5)
    args = parser.parse_args()

    # Start the prometheus exporter background thread
    import threading
    collector = MetricsCollector(port=8000)
    t = threading.Thread(target=collector.start, daemon=True)
    t.start()
    
    # Run the inference loop
    run_inference_simulation(args.n_requests, args.interval)
