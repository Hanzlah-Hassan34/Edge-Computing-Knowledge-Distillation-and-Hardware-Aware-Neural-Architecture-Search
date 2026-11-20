"""Benchmark the best Q2 ONNX model on CPU, including sampled process RSS."""

import argparse
import json
import os
import threading
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import psutil


HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = HERE / "05_ONNX" / "best_cifar10_nas_model.onnx"
DEFAULT_CONFIG = HERE / "03_Search_Results" / "best_architecture.json"
DEFAULT_OUTPUT = HERE / "06_Benchmarks" / "best_architecture_onnx_cpu_benchmark_remeasured.json"
MIB = 1024 ** 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    if not args.model.is_file() or not args.config.is_file():
        parser.error("ONNX model or architecture configuration is missing.")
    if args.repeats < 1:
        parser.error("--repeats must be positive")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    resolution = int(config["architecture_config"]["input_resolution"])
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(args.model), sess_options=options, providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    inputs = np.random.default_rng(42).standard_normal(
        (1, 3, resolution, resolution), dtype=np.float32
    )
    process = psutil.Process(os.getpid())
    baseline_rss = process.memory_info().rss
    peak_rss = [baseline_rss]
    stop_monitor = threading.Event()

    def sample_rss():
        while not stop_monitor.is_set():
            peak_rss[0] = max(peak_rss[0], process.memory_info().rss)
            stop_monitor.wait(0.005)

    monitor = threading.Thread(target=sample_rss, daemon=True)
    monitor.start()
    timings_ms = []
    try:
        for _ in range(args.warmup):
            session.run(None, {input_name: inputs})
        for _ in range(args.repeats):
            start = time.perf_counter()
            session.run(None, {input_name: inputs})
            timings_ms.append((time.perf_counter() - start) * 1000)
    finally:
        stop_monitor.set()
        monitor.join()
        peak_rss[0] = max(peak_rss[0], process.memory_info().rss)

    result = {
        "onnx_path": str(args.model),
        "onnx_size_mb": args.model.stat().st_size / MIB,
        "cpu_latency_median_ms": float(np.median(timings_ms)),
        "cpu_latency_p95_ms": float(np.percentile(timings_ms, 95)),
        "peak_process_rss_mb": peak_rss[0] / MIB,
        "inference_memory_delta_mb": max(0.0, peak_rss[0] - baseline_rss) / MIB,
        "memory_measurement": "psutil process RSS sampled every 5 ms",
        "batch_size": 1,
        "input_resolution": resolution,
        "cpu_threads": 1,
        "repeats": args.repeats,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
