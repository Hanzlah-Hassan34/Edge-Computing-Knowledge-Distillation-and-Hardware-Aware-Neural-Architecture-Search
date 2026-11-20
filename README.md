# Assignment 4 — Edge Computing

This repository contains the complete implementations and executed results for:

- **Q1:** CIFAR-100 knowledge distillation.
- **Q2:** Hardware-aware CNN architecture search on CIFAR-10.

The Colab notebooks are the primary end-to-end entry points. The accompanying Python
files are executable exports of the notebook code, plus standalone ONNX inference and
benchmark utilities.

## Deliverable map

| Assignment item | Included files |
|---|---|
| 1. ResNet-50 teacher and training code | `Q1_KD/03_Checkpoints/teacher/`; `Q1_KD/Q1_Knowledge_Distillation_Complete (1).ipynb`; `Q1_KD/q1_train_and_deploy.py` |
| 2. MobileNetV2, SqueezeNet, EfficientNet-Lite0 | Q1 notebook and training export |
| 3. Three supervised baseline checkpoints | `Q1_KD/03_Checkpoints/baseline/` |
| 4. Distillation method and three student checkpoints | Q1 notebook and `Q1_KD/03_Checkpoints/distilled/` |
| 5–6. NAS search space, generator, and Optuna search | Q2 notebook and `Q2_NAS_CIFAR10/q2_nas_train_and_deploy.py` |
| 7. Best architecture and search history | `Q2_NAS_CIFAR10/03_Search_Results/` |
| 8. Trained best NAS architecture | `Q2_NAS_CIFAR10/04_Checkpoints/final/` |
| 9–10. Inference and ONNX deployment | `Q1_KD/infer_onnx.py`, `Q2_NAS_CIFAR10/infer_onnx.py`, and the ONNX folders |
| 11. Benchmark scripts and results | `Q1_KD/benchmark_onnx.py`, `Q2_NAS_CIFAR10/benchmark_onnx.py`, and the results folders |
| 12. Technical report | [`A4_Technical_Report.pdf`](./A4_Technical_Report.pdf) |
| 13. Instructions | This README and [`requirements.txt`](./requirements.txt) |

## Run in Google Colab

1. Open the relevant notebook in Colab:
   - [Q1 knowledge distillation](./Q1_KD/Q1_Knowledge_Distillation_Complete%20(1).ipynb)
   - [Q2 hardware-aware NAS](./Q2_NAS_CIFAR10/Q2_Hardware_Aware_NAS_Complete%20(1).ipynb)
2. Select a GPU runtime.
3. Run cells from top to bottom and approve the Google Drive mount.
4. The notebooks download CIFAR-100/CIFAR-10 and save checkpoints, metrics, and ONNX files under the corresponding project folders in Google Drive.

The saved notebooks in this bundle contain their executed cell outputs. They completed
without notebook cell errors. Q1 trained for 10 teacher and student epochs; Q2 searched
12 candidates (3 epochs each) and trained the selected architecture for 25 epochs.

### Executable Python exports

The training exports retain the same Colab workflow and Drive output paths:

```bash
python Q1_KD/q1_train_and_deploy.py
python Q2_NAS_CIFAR10/q2_nas_train_and_deploy.py
```

In Colab, these can also be run with `%run`. If run outside Colab, set `PROJECT_ROOT`
to a writable output directory first. These scripts retrain models; use the notebooks
when you want the cell-by-cell workflow.

## Inference and benchmarking

The ONNX inference scripts accept an image file and print top-k CIFAR class predictions.
They download the relevant CIFAR dataset if its class labels are not already cached:

```bash
python Q1_KD/infer_onnx.py --model efficientnet_lite0_distilled --image path/to/image.jpg
python Q2_NAS_CIFAR10/infer_onnx.py --image path/to/image.jpg
```

Re-run CPU benchmarks after installing requirements:

```bash
python Q1_KD/benchmark_onnx.py
python Q2_NAS_CIFAR10/benchmark_onnx.py
```

The Q1 benchmark uses batch size 8 to match the saved notebook benchmark. Q2 measures
single-image latency. Keep these units distinct when comparing the results; CPU timing
also varies with the host machine.

## Recorded results

| Model | CIFAR-100 test accuracy |
|---|---:|
| MobileNetV2 baseline | 26.14% |
| MobileNetV2 distilled | 30.88% |
| SqueezeNet baseline | 14.02% |
| SqueezeNet distilled | 17.70% |
| EfficientNet-Lite0 baseline | 25.34% |
| EfficientNet-Lite0 distilled | 32.23% |

The best Q2 architecture achieved **76.32% validation accuracy** and **75.75% test
accuracy**. It has 90,618 parameters; its ONNX file is about 0.347 MiB, with measured
single-image CPU p95 latency of about 0.294 ms in the recorded Colab run.

## Repository setup and large model files

Install dependencies with:

```bash
python -m pip install -r requirements.txt
```

The dataset is not included; torchvision downloads it when required. Model checkpoints
and ONNX files are tracked with Git LFS settings in `.gitattributes` because several
Q1 checkpoints are larger than GitHub's regular-file limit. Before pushing:

```bash
git lfs install
git add .gitattributes
git add .
git status
```

Check that the large `.pt` and `.onnx` artifacts are listed as LFS objects before
publishing the repository.
