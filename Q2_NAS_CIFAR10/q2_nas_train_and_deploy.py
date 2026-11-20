"""Executable code export from the accompanying Colab notebook.

Run in Google Colab or a compatible environment after installing requirements.txt.
The notebook remains the canonical, cell-by-cell execution interface.
"""
from IPython.display import display

# %% Notebook code cell 1

import gc
import json
import os
import platform
import random
import threading
import time
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna
import onnx
import onnxruntime as ort
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm.auto import tqdm

print("Python:", platform.python_version())
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("WARNING: Training on CPU will be slow. Select Runtime > Change runtime type > GPU.")


# %% Notebook code cell 3
try:
    from google.colab import drive
except ModuleNotFoundError:
    drive = None

if drive is not None:
    drive.mount("/content/drive")
    PROJECT_ROOT = Path("/content/drive/MyDrive/NAS_CIFAR10_Edge_Neural_Architecture_Search")
else:
    PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent))
SEARCH_ROOT = PROJECT_ROOT / "03_Search_Results"
CHECKPOINT_ROOT = PROJECT_ROOT / "04_Checkpoints"
FINAL_MODEL_ROOT = CHECKPOINT_ROOT / "final"
BENCHMARK_ROOT = PROJECT_ROOT / "06_Benchmarks"
ONNX_ROOT = PROJECT_ROOT / "05_ONNX"
MANIFEST_ROOT = PROJECT_ROOT / "08_Manifests"
DATA_ROOT = PROJECT_ROOT / "data" / "cifar10"

for folder in (
    SEARCH_ROOT, FINAL_MODEL_ROOT, BENCHMARK_ROOT, ONNX_ROOT,
    MANIFEST_ROOT, DATA_ROOT,
):
    folder.mkdir(parents=True, exist_ok=True)

SEED = 42
BATCH_SIZE = 128
NUM_WORKERS = 2
SEARCH_TRIALS = 12
SEARCH_EPOCHS = 3
SEARCH_TRAIN_SAMPLES = 10_000
SEARCH_VAL_SAMPLES = 2_000
FINAL_EPOCHS = 25
DEFAULT_LEARNING_RATE = 2e-3
INPUT_RESOLUTIONS = [24, 32, 40]

# These are explicit deployment constraints for a single-image CPU inference target.
MAX_CPU_P95_LATENCY_MS = 100.0
MAX_CHECKPOINT_SIZE_MB = 8.0
MAX_INFERENCE_MEMORY_DELTA_MB = 256.0
CPU_BENCHMARK_REPEATS = 50

NUM_CLASSES = 10
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()
torch.backends.cudnn.benchmark = torch.cuda.is_available()
torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

print("Project:", PROJECT_ROOT)
print("Training device:", DEVICE)
print(
    "Limits:",
    f"p95 CPU latency <= {MAX_CPU_P95_LATENCY_MS:.1f} ms,",
    f"checkpoint <= {MAX_CHECKPOINT_SIZE_MB:.1f} MB,",
    f"inference memory increase <= {MAX_INFERENCE_MEMORY_DELTA_MB:.1f} MB",
)


# %% Notebook code cell 5
train_raw = datasets.CIFAR10(root=str(DATA_ROOT), train=True, download=True)
test_raw = datasets.CIFAR10(root=str(DATA_ROOT), train=False, download=True)
class_names = train_raw.classes
targets = np.asarray(train_raw.targets, dtype=np.int64)
test_targets = np.asarray(test_raw.targets, dtype=np.int64)
assert len(train_raw) == 50_000 and len(test_raw) == 10_000
assert len(class_names) == NUM_CLASSES

SPLIT_PATH = MANIFEST_ROOT / f"cifar10_split_seed{SEED}.npz"
if SPLIT_PATH.exists():
    split = np.load(SPLIT_PATH)
    train_idx = split["train_indices"].astype(np.int64)
    val_idx = split["val_indices"].astype(np.int64)
    if int(split["seed"]) != SEED:
        raise ValueError(f"Split seed mismatch in {SPLIT_PATH}")
else:
    train_idx, val_idx = train_test_split(
        np.arange(len(train_raw)),
        test_size=0.10,
        random_state=SEED,
        stratify=targets,
    )
    train_idx = np.asarray(train_idx, dtype=np.int64)
    val_idx = np.asarray(val_idx, dtype=np.int64)
    np.savez_compressed(
        SPLIT_PATH,
        train_indices=train_idx,
        val_indices=val_idx,
        seed=SEED,
    )

assert len(train_idx) == 45_000 and len(val_idx) == 5_000
assert len(np.intersect1d(train_idx, val_idx)) == 0
assert np.all(np.bincount(targets[train_idx], minlength=NUM_CLASSES) == 4_500)
assert np.all(np.bincount(targets[val_idx], minlength=NUM_CLASSES) == 500)
assert np.all(np.bincount(test_targets, minlength=NUM_CLASSES) == 1_000)

search_train_idx, _ = train_test_split(
    train_idx,
    train_size=min(SEARCH_TRAIN_SAMPLES, len(train_idx)),
    random_state=SEED,
    stratify=targets[train_idx],
)
search_val_idx, _ = train_test_split(
    val_idx,
    train_size=min(SEARCH_VAL_SAMPLES, len(val_idx)),
    random_state=SEED + 1,
    stratify=targets[val_idx],
)

class TransformedSubset(Dataset):
    def __init__(self, base_dataset, indices, transform):
        self.base_dataset = base_dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        image, label = self.base_dataset[int(self.indices[item])]
        return self.transform(image), int(label)


def make_transforms(resolution, training):
    image_ops = [transforms.Resize((resolution, resolution))]
    if training:
        image_ops.append(transforms.RandomHorizontalFlip())
    image_ops.extend([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return transforms.Compose(image_ops)


def make_loaders(resolution, train_indices, validation_indices, batch_size=BATCH_SIZE):
    train_set = TransformedSubset(
        train_raw, train_indices, make_transforms(resolution, training=True)
    )
    validation_set = TransformedSubset(
        train_raw, validation_indices, make_transforms(resolution, training=False)
    )
    generator = torch.Generator().manual_seed(SEED)
    common = {
        "batch_size": batch_size,
        "num_workers": NUM_WORKERS,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": NUM_WORKERS > 0,
    }
    return (
        DataLoader(train_set, shuffle=True, generator=generator, **common),
        DataLoader(validation_set, shuffle=False, **common),
    )


print(
    f"CIFAR-10 split: train={len(train_idx):,}, validation={len(val_idx):,}, "
    f"test={len(test_raw):,}; search subset={len(search_train_idx):,}/"
    f"{len(search_val_idx):,}"
)


# %% Notebook code cell 7
ACTIVATIONS = {
    "relu": lambda: nn.ReLU(inplace=False),
    "relu6": lambda: nn.ReLU6(inplace=False),
    "hardswish": lambda: nn.Hardswish(inplace=False),
}

class SearchableConvBlock(nn.Module):
    def __init__(self, in_channels, config):
        super().__init__()
        out_channels = config["channels"]
        kernel_size = config["kernel_size"]
        stride = config["stride"]
        padding = kernel_size // 2
        activation = ACTIVATIONS[config["activation"]]

        if config["block_type"] == "depthwise_separable":
            self.block = nn.Sequential(
                nn.Conv2d(
                    in_channels, in_channels, kernel_size, stride, padding,
                    groups=in_channels, bias=False,
                ),
                nn.BatchNorm2d(in_channels),
                activation(),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                activation(),
            )
        elif config["block_type"] == "standard":
            self.block = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size, stride, padding,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                activation(),
            )
        else:
            raise ValueError(f"Unsupported block type: {config['block_type']}")

    def forward(self, x):
        return self.block(x)


class SearchableCNN(nn.Module):
    def __init__(self, architecture_config):
        super().__init__()
        self.architecture_config = architecture_config
        blocks = []
        in_channels = 3
        for block_config in architecture_config["blocks"]:
            blocks.append(SearchableConvBlock(in_channels, block_config))
            in_channels = block_config["channels"]
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(in_channels, NUM_CLASSES)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


def sample_architecture(trial):
    depth = trial.suggest_int("depth", 2, 5)
    blocks = []
    for index in range(depth):
        blocks.append({
            "channels": trial.suggest_categorical(
                f"channels_{index}", [16, 24, 32, 48, 64, 96, 128]
            ),
            "kernel_size": trial.suggest_categorical(
                f"kernel_size_{index}", [3, 5]
            ),
            "block_type": trial.suggest_categorical(
                f"block_type_{index}", ["standard", "depthwise_separable"]
            ),
            "activation": trial.suggest_categorical(
                f"activation_{index}", ["relu", "relu6", "hardswish"]
            ),
            "stride": 2 if index < 2 else 1,
        })
    return {
        "input_resolution": trial.suggest_categorical(
            "input_resolution", INPUT_RESOLUTIONS
        ),
        "blocks": blocks,
    }


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


smoke_config = {
    "input_resolution": 32,
    "blocks": [
        {"channels": 24, "kernel_size": 3, "block_type": "standard", "activation": "relu", "stride": 2},
        {"channels": 32, "kernel_size": 5, "block_type": "depthwise_separable", "activation": "relu6", "stride": 2},
        {"channels": 48, "kernel_size": 3, "block_type": "depthwise_separable", "activation": "hardswish", "stride": 1},
    ],
}
smoke_model = SearchableCNN(smoke_config)
with torch.no_grad():
    smoke_logits = smoke_model(torch.randn(2, 3, 32, 32))
assert smoke_logits.shape == (2, NUM_CLASSES)
print("Generator smoke test passed; parameters:", f"{count_parameters(smoke_model):,}")
del smoke_model


# %% Notebook code cell 9
def evaluate_model(model, loader):
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    with torch.inference_mode():
        for inputs, labels in loader:
            inputs = inputs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            logits = model(inputs)
            loss_sum += criterion(logits, labels).item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return {"loss": loss_sum / total, "accuracy": correct / total}


def train_architecture(
    model,
    train_loader,
    validation_loader,
    epochs,
    learning_rate,
    run_name,
    checkpoint_path=None,
):
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    criterion = nn.CrossEntropyLoss()
    best_accuracy = -1.0
    best_state = None
    history = []

    for epoch in range(epochs):
        model.train()
        loss_sum = 0.0
        correct = 0
        total = 0
        progress = tqdm(
            train_loader,
            desc=f"{run_name} {epoch + 1}/{epochs}",
            leave=False,
        )
        for inputs, labels in progress:
            inputs = inputs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            loss_sum += loss.item() * batch_size
            correct += (logits.detach().argmax(dim=1) == labels).sum().item()
            total += batch_size
            progress.set_postfix(loss=f"{loss.item():.3f}")

        validation = evaluate_model(model, validation_loader)
        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": loss_sum / total,
            "train_accuracy": correct / total,
            "val_loss": validation["loss"],
            "val_accuracy": validation["accuracy"],
        }
        history.append(epoch_metrics)
        print(
            f"{run_name} epoch {epoch + 1}: "
            f"train_acc={epoch_metrics['train_accuracy']:.4f}, "
            f"val_acc={epoch_metrics['val_accuracy']:.4f}"
        )

        if validation["accuracy"] > best_accuracy:
            best_accuracy = validation["accuracy"]
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            if checkpoint_path is not None:
                torch.save({
                    "model_state": best_state,
                    "architecture_config": model.architecture_config,
                    "epoch": epoch + 1,
                    "best_validation_accuracy": best_accuracy,
                    "history": history,
                }, checkpoint_path)

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best_accuracy


# %% Notebook code cell 11
MIB = 1024 ** 2

def measure_cpu_resources(model, resolution, repeats=CPU_BENCHMARK_REPEATS):
    model = model.to("cpu").eval()
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_path = Path(temp_dir) / "candidate_state.pt"
        torch.save(model.state_dict(), checkpoint_path)
        checkpoint_size_mb = checkpoint_path.stat().st_size / MIB

    inputs = torch.randn(1, 3, resolution, resolution)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
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
    latencies = []
    try:
        with torch.inference_mode():
            for _ in range(10):
                model(inputs)
            for _ in range(repeats):
                start = time.perf_counter()
                model(inputs)
                latencies.append((time.perf_counter() - start) * 1000)
    finally:
        stop_monitor.set()
        monitor.join()
        peak_rss[0] = max(peak_rss[0], process.memory_info().rss)
        torch.set_num_threads(previous_threads)

    return {
        "checkpoint_size_mb": checkpoint_size_mb,
        "cpu_latency_median_ms": float(np.median(latencies)),
        "cpu_latency_p95_ms": float(np.percentile(latencies, 95)),
        "peak_process_rss_mb": peak_rss[0] / MIB,
        "inference_memory_delta_mb": max(0.0, peak_rss[0] - baseline_rss) / MIB,
        "memory_measurement": "psutil process RSS sampled every 5 ms",
        "cpu_threads": 1,
        "batch_size": 1,
    }


# %% Notebook code cell 13
def save_search_progress(study, trial):
    progress_path = SEARCH_ROOT / "optuna_search_history.csv"
    study.trials_dataframe().to_csv(progress_path, index=False)
    print(f"Saved search progress through trial {trial.number + 1}.")


def objective(trial):
    set_seed(SEED + trial.number)
    architecture_config = sample_architecture(trial)
    learning_rate = trial.suggest_float(
        "learning_rate", 5e-4, 5e-3, log=True
    )
    train_loader, validation_loader = make_loaders(
        architecture_config["input_resolution"],
        search_train_idx,
        search_val_idx,
    )
    model = SearchableCNN(architecture_config)
    trial.set_user_attr("architecture_config", architecture_config)
    trial.set_user_attr("parameter_count", count_parameters(model))

    model, history, best_accuracy = train_architecture(
        model,
        train_loader,
        validation_loader,
        SEARCH_EPOCHS,
        learning_rate,
        f"trial_{trial.number:03d}",
    )
    resources = measure_cpu_resources(
        model, architecture_config["input_resolution"]
    )
    trial.set_user_attr("validation_accuracy", best_accuracy)
    trial.set_user_attr("resources", resources)
    trial.set_user_attr("training_history", history)

    violations = []
    if resources["cpu_latency_p95_ms"] > MAX_CPU_P95_LATENCY_MS:
        violations.append("cpu_latency_p95_ms")
    if resources["checkpoint_size_mb"] > MAX_CHECKPOINT_SIZE_MB:
        violations.append("checkpoint_size_mb")
    if resources["inference_memory_delta_mb"] > MAX_INFERENCE_MEMORY_DELTA_MB:
        violations.append("inference_memory_delta_mb")
    trial.set_user_attr("rejected_constraints", violations)

    del model, train_loader, validation_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if violations:
        trial.set_user_attr("feasible", False)
        print(
            f"Trial {trial.number} rejected ({', '.join(violations)}); "
            f"val_acc={best_accuracy:.4f}"
        )
        return -1.0
    trial.set_user_attr("feasible", True)
    print(
        f"Trial {trial.number} feasible; val_acc={best_accuracy:.4f}, "
        f"latency_p95={resources['cpu_latency_p95_ms']:.2f} ms, "
        f"size={resources['checkpoint_size_mb']:.2f} MB"
    )
    return best_accuracy


study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(
        seed=SEED, n_startup_trials=min(5, SEARCH_TRIALS)
    ),
    study_name="cifar10_hardware_aware_cnn_nas",
)
study.optimize(
    objective,
    n_trials=SEARCH_TRIALS,
    callbacks=[save_search_progress],
    show_progress_bar=True,
)


# %% Notebook code cell 14
history_rows = []
for trial in study.trials:
    resources = trial.user_attrs.get("resources", {})
    history_rows.append({
        "trial": trial.number,
        "state": trial.state.name,
        "objective_value": trial.value,
        "validation_accuracy": trial.user_attrs.get("validation_accuracy"),
        "feasible": trial.user_attrs.get("feasible", False),
        "rejected_constraints": json.dumps(
            trial.user_attrs.get("rejected_constraints", [])
        ),
        "parameter_count": trial.user_attrs.get("parameter_count"),
        **resources,
        "architecture_config": json.dumps(
            trial.user_attrs.get("architecture_config", {})
        ),
        "params": json.dumps(trial.params),
    })
search_history = pd.DataFrame(history_rows)
search_history.to_csv(SEARCH_ROOT / "optuna_search_history.csv", index=False)
search_history.to_json(
    SEARCH_ROOT / "optuna_search_history.json",
    orient="records",
    indent=2,
)

feasible_trials = [
    trial for trial in study.trials
    if trial.state == optuna.trial.TrialState.COMPLETE
    and trial.user_attrs.get("feasible", False)
]
if not feasible_trials:
    raise RuntimeError(
        "No feasible architecture was found. Adjust the deployment limits or "
        "expand the search space, then rerun the search."
    )
best_trial = max(feasible_trials, key=lambda trial: trial.value)
best_config = best_trial.user_attrs["architecture_config"]
best_search_summary = {
    "best_trial": best_trial.number,
    "search_validation_accuracy": best_trial.user_attrs["validation_accuracy"],
    "parameter_count": best_trial.user_attrs["parameter_count"],
    "learning_rate": best_trial.params["learning_rate"],
    "resources": best_trial.user_attrs["resources"],
    "architecture_config": best_config,
    "deployment_constraints": {
        "max_cpu_p95_latency_ms": MAX_CPU_P95_LATENCY_MS,
        "max_checkpoint_size_mb": MAX_CHECKPOINT_SIZE_MB,
        "max_inference_memory_delta_mb": MAX_INFERENCE_MEMORY_DELTA_MB,
    },
}
BEST_CONFIG_PATH = SEARCH_ROOT / "best_architecture.json"
with open(BEST_CONFIG_PATH, "w", encoding="utf-8") as file:
    json.dump(best_search_summary, file, indent=2)

print("Selected feasible trial:", best_trial.number)
print("Search validation accuracy:", f"{best_trial.value:.4f}")
print("Best architecture:", json.dumps(best_config, indent=2))


# %% Notebook code cell 16
best_resolution = best_config["input_resolution"]
final_train_loader, final_validation_loader = make_loaders(
    best_resolution, train_idx, val_idx
)
best_learning_rate = best_trial.params["learning_rate"]
final_model = SearchableCNN(best_config)
FINAL_CHECKPOINT_PATH = FINAL_MODEL_ROOT / "best_cifar10_nas_model.pt"
final_model, final_history, final_validation_accuracy = train_architecture(
    final_model,
    final_train_loader,
    final_validation_loader,
    FINAL_EPOCHS,
    best_learning_rate,
    "best_architecture_from_scratch",
    checkpoint_path=FINAL_CHECKPOINT_PATH,
)
pd.DataFrame(final_history).to_csv(
    BENCHMARK_ROOT / "best_architecture_training_history.csv",
    index=False,
)
print(
    "Best validation accuracy after full training:",
    f"{final_validation_accuracy:.4f}",
)
print("Saved checkpoint:", FINAL_CHECKPOINT_PATH)


# %% Notebook code cell 18
test_dataset = TransformedSubset(
    test_raw, np.arange(len(test_raw)), make_transforms(best_resolution, training=False)
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=NUM_WORKERS > 0,
)
test_metrics = evaluate_model(final_model.to(DEVICE), test_loader)
test_result = {
    "test_loss": test_metrics["loss"],
    "test_accuracy": test_metrics["accuracy"],
    "validation_accuracy": final_validation_accuracy,
    "parameter_count": count_parameters(final_model),
    "architecture_config": best_config,
    "checkpoint": str(FINAL_CHECKPOINT_PATH),
}
with open(BENCHMARK_ROOT / "best_architecture_test_metrics.json", "w", encoding="utf-8") as file:
    json.dump(test_result, file, indent=2)
print(
    f"Final test accuracy={test_metrics['accuracy']:.4f}; "
    f"test loss={test_metrics['loss']:.4f}"
)


# %% Notebook code cell 20
final_model = final_model.to("cpu").eval()
ONNX_PATH = ONNX_ROOT / "best_cifar10_nas_model.onnx"
export_input = torch.randn(1, 3, best_resolution, best_resolution)
torch.onnx.export(
    final_model,
    export_input,
    str(ONNX_PATH),
    input_names=["images"],
    output_names=["logits"],
    dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
    opset_version=17,
    dynamo=False,
)
onnx.checker.check_model(onnx.load(str(ONNX_PATH)))

session_options = ort.SessionOptions()
session_options.intra_op_num_threads = 1
session_options.inter_op_num_threads = 1
onnx_session = ort.InferenceSession(
    str(ONNX_PATH),
    sess_options=session_options,
    providers=["CPUExecutionProvider"],
)
assert "CPUExecutionProvider" in onnx_session.get_providers()

parity_inputs = torch.randn(4, 3, best_resolution, best_resolution)
with torch.inference_mode():
    pytorch_logits = final_model(parity_inputs).numpy()
onnx_logits = onnx_session.run(
    ["logits"], {"images": parity_inputs.numpy()}
)[0]
onnx_max_abs_error = float(np.max(np.abs(pytorch_logits - onnx_logits)))
assert np.allclose(pytorch_logits, onnx_logits, rtol=1e-3, atol=1e-4), (
    f"ONNX output mismatch; max_abs_error={onnx_max_abs_error}"
)


# %% Notebook code cell 21
def measure_onnx_cpu(session, resolution, repeats=100):
    inputs = np.random.default_rng(SEED).standard_normal(
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
    latencies = []
    try:
        for _ in range(10):
            session.run(["logits"], {"images": inputs})
        for _ in range(repeats):
            start = time.perf_counter()
            session.run(["logits"], {"images": inputs})
            latencies.append((time.perf_counter() - start) * 1000)
    finally:
        stop_monitor.set()
        monitor.join()
        peak_rss[0] = max(peak_rss[0], process.memory_info().rss)

    return {
        "cpu_latency_median_ms": float(np.median(latencies)),
        "cpu_latency_p95_ms": float(np.percentile(latencies, 95)),
        "peak_process_rss_mb": peak_rss[0] / MIB,
        "inference_memory_delta_mb": max(0.0, peak_rss[0] - baseline_rss) / MIB,
        "memory_measurement": "psutil process RSS sampled every 5 ms",
        "batch_size": 1,
        "input_resolution": resolution,
    }


onnx_resources = measure_onnx_cpu(onnx_session, best_resolution)
onnx_size_mb = ONNX_PATH.stat().st_size / MIB
onnx_benchmark = {
    "onnx_path": str(ONNX_PATH),
    "onnx_size_mb": onnx_size_mb,
    "onnx_max_abs_logit_error": onnx_max_abs_error,
    **onnx_resources,
}
with open(BENCHMARK_ROOT / "best_architecture_onnx_cpu_benchmark.json", "w", encoding="utf-8") as file:
    json.dump(onnx_benchmark, file, indent=2)
print("ONNX verified; max abs logit error:", onnx_max_abs_error)
print(json.dumps(onnx_benchmark, indent=2))


# %% Notebook code cell 23
inference_transform = make_transforms(best_resolution, training=False)

def predict_cifar10(image, top_k=5):
    input_tensor = inference_transform(image).unsqueeze(0).numpy().astype(np.float32)
    logits = onnx_session.run(["logits"], {"images": input_tensor})[0][0]
    logits = logits - logits.max()
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum()
    indices = np.argsort(probabilities)[-top_k:][::-1]
    return [
        (class_names[int(index)], float(probabilities[index]))
        for index in indices
    ]


example_image, _ = test_raw[0]
print("Example ONNX top-3:", predict_cifar10(example_image, top_k=3))


# %% Notebook code cell 24
checks = {
    "Optuna trials recorded": len(study.trials) == SEARCH_TRIALS,
    "Feasible best architecture": best_trial.user_attrs.get("feasible", False),
    "Best architecture JSON": BEST_CONFIG_PATH.exists(),
    "Search history CSV": (SEARCH_ROOT / "optuna_search_history.csv").exists(),
    "Final checkpoint": FINAL_CHECKPOINT_PATH.exists(),
    "Held-out test metrics": (BENCHMARK_ROOT / "best_architecture_test_metrics.json").exists(),
    "ONNX model": ONNX_PATH.exists(),
    "ONNX Runtime CPU provider": "CPUExecutionProvider" in onnx_session.get_providers(),
    "ONNX parity": np.allclose(pytorch_logits, onnx_logits, rtol=1e-3, atol=1e-4),
}
for check_name, passed in checks.items():
    print(f"[{'PASS' if passed else 'FAIL'}] {check_name}")
if not all(checks.values()):
    raise RuntimeError("Q2 is incomplete; review the failed checks above.")

manifest = {
    "assignment": "A4 Q2 - Hardware-Aware CIFAR-10 NAS",
    "seed": SEED,
    "dataset": "CIFAR-10",
    "train_size": len(train_idx),
    "validation_size": len(val_idx),
    "test_size": len(test_raw),
    "class_names": class_names,
    "device": str(DEVICE),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "python": platform.python_version(),
    "torch": torch.__version__,
    "search_trials": SEARCH_TRIALS,
    "search_epochs": SEARCH_EPOCHS,
    "final_epochs": FINAL_EPOCHS,
    "best_architecture": best_config,
    "test_accuracy": test_metrics["accuracy"],
    "onnx_cpu_benchmark": onnx_benchmark,
}
with open(MANIFEST_ROOT / "q2_experiment_manifest.json", "w", encoding="utf-8") as file:
    json.dump(manifest, file, indent=2)

print("\nQUESTION 2 COMPLETE")
print("Search history:", SEARCH_ROOT / "optuna_search_history.csv")
print("Best architecture:", BEST_CONFIG_PATH)
print("ONNX CPU benchmark:", BENCHMARK_ROOT / "best_architecture_onnx_cpu_benchmark.json")

