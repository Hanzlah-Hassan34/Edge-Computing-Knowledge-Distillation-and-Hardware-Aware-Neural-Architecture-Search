"""Run one CIFAR-10 image through the selected NAS architecture using ONNX Runtime."""

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision import datasets, transforms


MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)
HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--data-root", type=Path, default=HERE / "data" / "cifar10")
    parser.add_argument(
        "--config", type=Path, default=HERE / "03_Search_Results" / "best_architecture.json"
    )
    parser.add_argument(
        "--onnx-model", type=Path, default=HERE / "05_ONNX" / "best_cifar10_nas_model.onnx"
    )
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    if not args.image.is_file():
        parser.error(f"Image does not exist: {args.image}")
    if not args.config.is_file() or not args.onnx_model.is_file():
        parser.error("Best architecture JSON or ONNX model is missing.")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    resolution = int(config["architecture_config"]["input_resolution"])
    class_names = datasets.CIFAR10(
        root=str(args.data_root), train=False, download=True
    ).classes
    preprocess = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    with Image.open(args.image) as image:
        input_batch = preprocess(image.convert("RGB")).unsqueeze(0).numpy()

    session = ort.InferenceSession(
        str(args.onnx_model), providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    logits = session.run(None, {input_name: input_batch})[0][0]
    logits = logits - np.max(logits)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum()
    top_indices = np.argsort(probabilities)[-min(args.top_k, len(class_names)):][::-1]

    for index in top_indices:
        print(f"{class_names[int(index)]:20s} {probabilities[index]:.4f}")


if __name__ == "__main__":
    main()
