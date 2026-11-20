"""Benchmark all Q1 student ONNX models on CPU and write a reproducible CSV."""

import argparse
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd


HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "04_ONNX_Models" / "fp32"
OUTPUT = HERE / "05_Results" / "metrics" / "onnx_cpu_benchmarks_remeasured.csv"
MODEL_NAMES = (
    "mobilenet_v2_baseline",
    "mobilenet_v2_distilled",
    "squeezenet_baseline",
    "squeezenet_distilled",
    "efficientnet_lite0_baseline",
    "efficientnet_lite0_distilled",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.repeats < 1 or args.batch_size < 1:
        parser.error("--repeats and --batch-size must be positive")

    rows = []
    for model_name in MODEL_NAMES:
        model_path = args.model_dir / f"{model_name}.onnx"
        if not model_path.is_file():
            raise FileNotFoundError(f"Missing ONNX model: {model_path}")
        session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        input_name = session.get_inputs()[0].name
        batch = np.random.default_rng(42).standard_normal(
            (args.batch_size, 3, 32, 32), dtype=np.float32
        )
        for _ in range(args.warmup):
            session.run(None, {input_name: batch})

        timings_ms = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            session.run(None, {input_name: batch})
            timings_ms.append((time.perf_counter() - start) * 1000)
        rows.append({
            "model": model_name,
            "onnx_path": str(model_path),
            "size_mb": model_path.stat().st_size / (1024 ** 2),
            "cpu_latency_mean_ms_per_batch": float(np.mean(timings_ms)),
            "cpu_latency_median_ms_per_batch": float(np.median(timings_ms)),
            "cpu_latency_p95_ms_per_batch": float(np.percentile(timings_ms, 95)),
            "batch_size": args.batch_size,
            "repeats": args.repeats,
        })
        print(
            f"{model_name}: mean={rows[-1]['cpu_latency_mean_ms_per_batch']:.3f} "
            f"ms/batch, size={rows[-1]['size_mb']:.2f} MiB"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
