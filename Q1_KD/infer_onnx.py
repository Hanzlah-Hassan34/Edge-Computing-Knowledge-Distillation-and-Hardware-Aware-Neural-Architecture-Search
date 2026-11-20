"""Run one CIFAR-100 image through any exported Q1 student ONNX model."""

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision import datasets, transforms


MODEL_NAMES = (
    "mobilenet_v2_baseline",
    "mobilenet_v2_distilled",
    "squeezenet_baseline",
    "squeezenet_distilled",
    "efficientnet_lite0_baseline",
    "efficientnet_lite0_distilled",
)
MEAN = (0.5071, 0.4867, 0.4408)
STD = (0.2675, 0.2565, 0.2761)
HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--data-root", type=Path, default=HERE / "data" / "cifar100")
    parser.add_argument(
        "--onnx-dir", type=Path, default=HERE / "04_ONNX_Models" / "fp32"
    )
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    if not args.image.is_file():
        parser.error(f"Image does not exist: {args.image}")

    model_path = args.onnx_dir / f"{args.model}.onnx"
    if not model_path.is_file():
        parser.error(f"ONNX model not found: {model_path}")

    class_names = datasets.CIFAR100(
        root=str(args.data_root), train=False, download=True
    ).classes
    preprocess = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    with Image.open(args.image) as image:
        input_batch = preprocess(image.convert("RGB")).unsqueeze(0).numpy()

    session = ort.InferenceSession(
        str(model_path), providers=["CPUExecutionProvider"]
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
