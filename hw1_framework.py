from __future__ import annotations

import copy
import math
import random
import time
from pathlib import Path
from typing import Any, Callable

try:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import psutil
    import seaborn as sns
    import torch
    import torch.nn as nn
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        precision_recall_fscore_support,
    )
    from torch.utils.data import DataLoader, Dataset
    from torchvision import datasets, transforms
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as TF
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Missing runtime dependencies. Install the packages from requirements.txt before "
        "running the notebook."
    ) from exc


DEFAULT_CLASS_NAMES = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "a",
    "b",
    "d",
    "e",
    "f",
    "g",
    "h",
    "n",
    "q",
    "r",
    "t",
]


def set_seed(seed: int = 42) -> None:
    """Keep dataset splits and training runs reproducible across teammates."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Prefer CUDA when available, otherwise fall back to CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_project_dirs(base_dir: str | Path) -> dict[str, Path]:
    """Centralize output folders so all teammates save results to the same places."""
    root = Path(base_dir).resolve()
    paths = {
        "root": root,
        "data": root / "data",
        "figures": root / "figures",
        "models": root / "models",
        "results": root / "results",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


class EmnistOrientationFix:
    """EMNIST samples are commonly shown rotated; this transform makes plots easier to read."""

    def __call__(self, image):
        image = TF.rotate(image, -90, interpolation=InterpolationMode.NEAREST)
        image = TF.hflip(image)
        return image


class AddGaussianNoise:
    """A lightweight augmentation that keeps the handwritten character recognizable."""

    def __init__(self, std: float = 0.0) -> None:
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.std <= 0:
            return tensor
        noise = torch.randn_like(tensor) * self.std
        return torch.clamp(tensor + noise, 0.0, 1.0)


class TransformSubset(Dataset):
    """Use different transforms for train/validation while keeping one fixed split."""

    def __init__(
        self,
        base_dataset: Dataset,
        indices: list[int],
        transform: Callable | None = None,
    ) -> None:
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        image, label = self.base_dataset[self.indices[idx]]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def build_transforms(
    train: bool = True,
    augment: bool = False,
    rotation_deg: float = 10.0,
    noise_std: float = 0.0,
    blur: bool = False,
) -> transforms.Compose:
    """Keep train/test preprocessing consistent and make augmentation easy to toggle."""
    transform_steps: list[Any] = [EmnistOrientationFix()]
    if train and augment:
        transform_steps.append(transforms.RandomRotation(rotation_deg))
        transform_steps.append(
            transforms.RandomAffine(
                degrees=0,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
                interpolation=InterpolationMode.BILINEAR,
            )
        )
        if blur:
            transform_steps.append(transforms.GaussianBlur(kernel_size=3))
    transform_steps.extend(
        [
            transforms.ToTensor(),
            AddGaussianNoise(std=noise_std if train and augment else 0.0),
            transforms.Normalize(mean=(0.5,), std=(0.5,)),
        ]
    )
    return transforms.Compose(transform_steps)


def get_class_names(dataset: Dataset) -> list[str]:
    classes = getattr(dataset, "classes", None)
    if classes is None:
        return DEFAULT_CLASS_NAMES
    return [str(label) for label in classes]


def load_emnist_balanced(
    data_dir: str | Path,
    batch_size: int = 128,
    valid_ratio: float = 0.1,
    num_workers: int = 0,
    subset_ratio: float = 1.0,
    augment: bool = False,
    rotation_deg: float = 10.0,
    noise_std: float = 0.0,
    blur: bool = False,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Build one reproducible split that every model can share.

    `subset_ratio` is only applied to the training portion so the validation/test sets stay fixed.
    """
    raw_train = datasets.EMNIST(
        root=str(data_dir),
        split="balanced",
        train=True,
        download=True,
        transform=None,
    )
    raw_test = datasets.EMNIST(
        root=str(data_dir),
        split="balanced",
        train=False,
        download=True,
        transform=None,
    )

    generator = torch.Generator().manual_seed(seed)
    full_indices = torch.randperm(len(raw_train), generator=generator).tolist()
    valid_size = int(len(raw_train) * valid_ratio)
    train_indices = full_indices[valid_size:]
    valid_indices = full_indices[:valid_size]

    subset_ratio = max(0.0, min(1.0, subset_ratio))
    if subset_ratio < 1.0:
        target_size = max(1, int(len(train_indices) * subset_ratio))
        train_indices = train_indices[:target_size]

    train_dataset = TransformSubset(
        raw_train,
        train_indices,
        transform=build_transforms(
            train=True,
            augment=augment,
            rotation_deg=rotation_deg,
            noise_std=noise_std,
            blur=blur,
        ),
    )
    valid_dataset = TransformSubset(
        raw_train,
        valid_indices,
        transform=build_transforms(train=False, augment=False),
    )
    test_indices = list(range(len(raw_test)))
    test_dataset = TransformSubset(
        raw_test,
        test_indices,
        transform=build_transforms(train=False, augment=False),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return {
        "train_dataset": train_dataset,
        "valid_dataset": valid_dataset,
        "test_dataset": test_dataset,
        "train_loader": train_loader,
        "valid_loader": valid_loader,
        "test_loader": test_loader,
        "class_names": get_class_names(raw_train),
    }


def show_dataset_stats(
    train_dataset: Dataset,
    valid_dataset: Dataset,
    test_dataset: Dataset,
    class_names: list[str],
) -> pd.DataFrame:
    stats = pd.DataFrame(
        [
            {"split": "train", "samples": len(train_dataset)},
            {"split": "valid", "samples": len(valid_dataset)},
            {"split": "test", "samples": len(test_dataset)},
            {"split": "classes", "samples": len(class_names)},
        ]
    )
    print(stats.to_string(index=False))
    return stats


def denormalize_image(image: torch.Tensor) -> torch.Tensor:
    return image * 0.5 + 0.5


def show_sample_images(
    dataset: Dataset,
    class_names: list[str],
    num_samples: int = 12,
    seed: int = 42,
):
    generator = random.Random(seed)
    chosen_indices = generator.sample(range(len(dataset)), k=min(num_samples, len(dataset)))
    rows = math.ceil(len(chosen_indices) / 4)
    fig, axes = plt.subplots(rows, 4, figsize=(12, 3 * rows))
    axes = np.array(axes).reshape(-1)
    for axis in axes:
        axis.axis("off")
    for plot_index, sample_index in enumerate(chosen_indices):
        image, label = dataset[sample_index]
        axes[plot_index].imshow(denormalize_image(image).squeeze(0), cmap="gray")
        axes[plot_index].set_title(f"Label: {class_names[label]}")
        axes[plot_index].axis("off")
    fig.suptitle("Sample EMNIST Balanced Images", fontsize=14)
    fig.tight_layout()
    return fig


def resolve_activation(name: str) -> nn.Module:
    activation_map = {
        "relu": nn.ReLU,
        "leaky_relu": nn.LeakyReLU,
        "elu": nn.ELU,
        "gelu": nn.GELU,
    }
    key = name.lower()
    if key not in activation_map:
        raise ValueError(f"Unsupported activation: {name}")
    return activation_map[key]()


def make_normalization(name: str | None, num_features: int, dims: str) -> nn.Module | None:
    if name is None or str(name).lower() == "none":
        return None
    normalized_name = str(name).lower()
    if dims == "1d":
        if normalized_name == "batchnorm":
            return nn.BatchNorm1d(num_features)
        if normalized_name == "layernorm":
            return nn.LayerNorm(num_features)
    if dims == "2d":
        if normalized_name == "batchnorm":
            return nn.BatchNorm2d(num_features)
        if normalized_name == "layernorm":
            return nn.GroupNorm(1, num_features)
    raise ValueError(f"Unsupported normalization '{name}' for dims='{dims}'")


class MLPClassifier(nn.Module):
    """Baseline MLP used for the part assigned to you."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        input_dim = config.get("input_dim", 28 * 28)
        hidden_dims = config.get("hidden_dims", [512, 256, 128])
        num_classes = config.get("num_classes", 47)
        activation_name = config.get("activation", "relu")
        normalization_name = config.get("normalization", "batchnorm")
        dropout = config.get("dropout", 0.3)

        layers: list[nn.Module] = [nn.Flatten()]
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            norm_layer = make_normalization(normalization_name, hidden_dim, dims="1d")
            if norm_layer is not None:
                layers.append(norm_layer)
            layers.append(resolve_activation(activation_name))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The final layer returns logits because CrossEntropyLoss handles softmax internally.
        return self.network(x)


class CNNScaffold(nn.Module):
    """Reusable CNN baseline that teammates can deepen or tune later."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        channels = config.get("channels", [32, 64, 128])
        activation = config.get("activation", "relu")
        normalization = config.get("normalization", "batchnorm")
        dropout = config.get("dropout", 0.3)
        num_classes = config.get("num_classes", 47)

        blocks: list[nn.Module] = []
        in_channels = 1
        for out_channels in channels:
            blocks.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            norm_layer = make_normalization(normalization, out_channels, dims="2d")
            if norm_layer is not None:
                blocks.append(norm_layer)
            blocks.append(resolve_activation(activation))
            blocks.append(nn.MaxPool2d(kernel_size=2))
            if dropout > 0:
                blocks.append(nn.Dropout2d(dropout))
            in_channels = out_channels

        self.features = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_channels, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        activation: str = "relu",
        normalization: str = "batchnorm",
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.norm1 = make_normalization(normalization, out_channels, dims="2d")
        self.act = resolve_activation(activation)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.norm2 = make_normalization(normalization, out_channels, dims="2d")
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            shortcut_layers: list[nn.Module] = [
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            ]
            shortcut_norm = make_normalization(normalization, out_channels, dims="2d")
            if shortcut_norm is not None:
                shortcut_layers.append(shortcut_norm)
            self.shortcut = nn.Sequential(*shortcut_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.conv1(x)
        if self.norm1 is not None:
            out = self.norm1(out)
        out = self.act(out)
        out = self.conv2(out)
        if self.norm2 is not None:
            out = self.norm2(out)
        out = out + residual
        out = self.act(out)
        return out


class ResNetScaffold(nn.Module):
    """A lightweight ResNet starter that already contains residual connections."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        base_channels = config.get("base_channels", 32)
        activation = config.get("activation", "relu")
        normalization = config.get("normalization", "batchnorm")
        num_classes = config.get("num_classes", 47)

        stem_layers: list[nn.Module] = [nn.Conv2d(1, base_channels, kernel_size=3, padding=1, bias=False)]
        stem_norm = make_normalization(normalization, base_channels, dims="2d")
        if stem_norm is not None:
            stem_layers.append(stem_norm)
        stem_layers.append(resolve_activation(activation))
        self.stem = nn.Sequential(*stem_layers)

        self.layer1 = ResidualBlock(
            base_channels,
            base_channels,
            stride=1,
            activation=activation,
            normalization=normalization,
        )
        self.layer2 = ResidualBlock(
            base_channels,
            base_channels * 2,
            stride=2,
            activation=activation,
            normalization=normalization,
        )
        self.layer3 = ResidualBlock(
            base_channels * 2,
            base_channels * 4,
            stride=2,
            activation=activation,
            normalization=normalization,
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(base_channels * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        return self.head(x)


class PatchEmbedding(nn.Module):
    def __init__(self, image_size: int, patch_size: int, embed_dim: int) -> None:
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.projection = nn.Conv2d(
            1,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class VisionTransformerScaffold(nn.Module):
    """A compact ViT scaffold designed for 28x28 grayscale images."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        image_size = config.get("image_size", 28)
        patch_size = config.get("patch_size", 4)
        embed_dim = config.get("embed_dim", 128)
        num_heads = config.get("num_heads", 4)
        depth = config.get("depth", 4)
        mlp_ratio = config.get("mlp_ratio", 2.0)
        dropout = config.get("dropout", 0.1)
        num_classes = config.get("num_classes", 47)

        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size for the ViT scaffold")

        self.patch_embed = PatchEmbedding(image_size=image_size, patch_size=patch_size, embed_dim=embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(x)
        batch_size = tokens.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)
        tokens = tokens + self.position_embedding[:, : tokens.size(1), :]
        tokens = self.encoder(tokens)
        cls_representation = self.norm(tokens[:, 0])
        return self.head(cls_representation)


def build_mlp(config: dict[str, Any]) -> nn.Module:
    return MLPClassifier(config)


def build_cnn(config: dict[str, Any]) -> nn.Module:
    return CNNScaffold(config)


def build_resnet(config: dict[str, Any]) -> nn.Module:
    return ResNetScaffold(config)


def build_vit(config: dict[str, Any]) -> nn.Module:
    return VisionTransformerScaffold(config)


MODEL_BUILDERS = {
    "mlp": build_mlp,
    "cnn": build_cnn,
    "resnet": build_resnet,
    "vit": build_vit,
}


def build_optimizer(model: nn.Module, optimizer_name: str, lr: float, weight_decay: float = 0.0):
    key = optimizer_name.lower()
    if key == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    if key == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if key == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def build_scheduler(optimizer, scheduler_name: str | None, scheduler_params: dict[str, Any] | None = None):
    if scheduler_name is None or str(scheduler_name).lower() == "none":
        return None
    scheduler_params = scheduler_params or {}
    key = scheduler_name.lower()
    if key == "steplr":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_params.get("step_size", 5),
            gamma=scheduler_params.get("gamma", 0.5),
        )
    if key == "cosineannealinglr":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_params.get("t_max", 10),
            eta_min=scheduler_params.get("eta_min", 1e-6),
        )
    if key == "reducelronplateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_params.get("mode", "min"),
            factor=scheduler_params.get("factor", 0.5),
            patience=scheduler_params.get("patience", 2),
        )
    raise ValueError(f"Unsupported scheduler: {scheduler_name}")


def build_loss_fn() -> nn.Module:
    return nn.CrossEntropyLoss()


def compute_l1_penalty(model: nn.Module) -> torch.Tensor:
    return sum(parameter.abs().sum() for parameter in model.parameters())


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer,
    device: torch.device,
    l1_lambda: float = 0.0,
) -> dict[str, float]:
    model.train()
    running_loss = 0.0
    running_correct = 0
    sample_count = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        if l1_lambda > 0:
            loss = loss + l1_lambda * compute_l1_penalty(model)
        loss.backward()
        optimizer.step()

        predictions = logits.argmax(dim=1)
        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        running_correct += (predictions == labels).sum().item()
        sample_count += batch_size

    return {
        "loss": running_loss / max(sample_count, 1),
        "accuracy": running_correct / max(sample_count, 1),
    }


@torch.no_grad()
def evaluate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    collect_predictions: bool = False,
) -> dict[str, Any]:
    model.eval()
    running_loss = 0.0
    running_correct = 0
    sample_count = 0
    all_labels: list[int] = []
    all_predictions: list[int] = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)
        predictions = logits.argmax(dim=1)

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        running_correct += (predictions == labels).sum().item()
        sample_count += batch_size

        if collect_predictions:
            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

    result = {
        "loss": running_loss / max(sample_count, 1),
        "accuracy": running_correct / max(sample_count, 1),
    }
    if collect_predictions:
        result["y_true"] = all_labels
        result["y_pred"] = all_predictions
    return result


def save_checkpoint(model: nn.Module, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    return path


def load_checkpoint(model: nn.Module, path: str | Path, device: torch.device) -> nn.Module:
    state_dict = torch.load(Path(path), map_location=device)
    model.load_state_dict(state_dict)
    return model


def run_training_experiment(
    model_name: str,
    model_builder: Callable[[dict[str, Any]], nn.Module],
    config: dict[str, Any],
    loaders: dict[str, Any],
    device: torch.device,
    output_dir: str | Path,
) -> dict[str, Any]:
    """
    Train one model with a shared loop so results remain comparable across teammates.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = model_builder(config).to(device)
    criterion = build_loss_fn()
    optimizer = build_optimizer(
        model=model,
        optimizer_name=config.get("optimizer", "adam"),
        lr=config.get("learning_rate", 1e-3),
        weight_decay=config.get("weight_decay", 0.0),
    )
    scheduler = build_scheduler(
        optimizer=optimizer,
        scheduler_name=config.get("scheduler"),
        scheduler_params=config.get("scheduler_params"),
    )

    epochs = config.get("epochs", 10)
    early_stopping_patience = config.get("early_stopping_patience", 5)
    l1_lambda = config.get("l1_lambda", 0.0)

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "valid_loss": [],
        "valid_accuracy": [],
        "learning_rate": [],
    }
    process = psutil.Process()
    peak_memory_mb = process.memory_info().rss / (1024**2)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_valid_accuracy = float("-inf")
    best_valid_loss = float("inf")
    patience_counter = 0
    start_time = time.perf_counter()

    for epoch in range(epochs):
        train_metrics = train_one_epoch(
            model=model,
            loader=loaders["train_loader"],
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            l1_lambda=l1_lambda,
        )
        valid_metrics = evaluate_one_epoch(
            model=model,
            loader=loaders["valid_loader"],
            criterion=criterion,
            device=device,
            collect_predictions=False,
        )

        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_metrics["loss"])
        history["train_accuracy"].append(train_metrics["accuracy"])
        history["valid_loss"].append(valid_metrics["loss"])
        history["valid_accuracy"].append(valid_metrics["accuracy"])
        history["learning_rate"].append(current_lr)

        if scheduler is not None:
            if scheduler.__class__.__name__.lower() == "reducelronplateau":
                scheduler.step(valid_metrics["loss"])
            else:
                scheduler.step()

        peak_memory_mb = max(peak_memory_mb, process.memory_info().rss / (1024**2))

        if valid_metrics["accuracy"] > best_valid_accuracy:
            best_valid_accuracy = valid_metrics["accuracy"]
            best_valid_loss = valid_metrics["loss"]
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            break

    training_time_sec = time.perf_counter() - start_time
    model.load_state_dict(best_state)
    checkpoint_path = save_checkpoint(model, output_dir / f"{model_name}_best.pt")

    gpu_peak_memory_mb = None
    if torch.cuda.is_available():
        gpu_peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024**2)

    summary = {
        "model_name": model_name,
        "best_epoch": best_epoch,
        "best_valid_accuracy": best_valid_accuracy,
        "best_valid_loss": best_valid_loss,
        "training_time_sec": training_time_sec,
        "peak_process_memory_mb": peak_memory_mb,
        "peak_gpu_memory_mb": gpu_peak_memory_mb,
        "checkpoint_path": checkpoint_path,
    }

    return {
        "model": model,
        "history": history,
        "summary": summary,
        "config": copy.deepcopy(config),
    }


def plot_training_curves(history: dict[str, list[float]], model_name: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["valid_loss"], label="Valid Loss")
    axes[0].set_title(f"{model_name} Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(history["train_accuracy"], label="Train Accuracy")
    axes[1].plot(history["valid_accuracy"], label="Valid Accuracy")
    axes[1].set_title(f"{model_name} Accuracy Curve")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    fig.tight_layout()
    return fig


def compute_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": accuracy,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
    }


def summarize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    summary_keys = [
        "loss",
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
    ]
    return {key: metrics[key] for key in summary_keys if key in metrics}


@torch.no_grad()
def evaluate_on_test(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    criterion = build_loss_fn()
    result = evaluate_one_epoch(
        model=model,
        loader=loader,
        criterion=criterion,
        device=device,
        collect_predictions=True,
    )
    result.update(compute_metrics(result["y_true"], result["y_pred"]))
    return result


def plot_confusion_matrix_from_preds(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str],
    model_name: str,
    figsize: tuple[int, int] = (12, 10),
):
    matrix = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        matrix,
        cmap="Blues",
        ax=ax,
        square=False,
        cbar=True,
        xticklabels=class_names,
        yticklabels=class_names,
    )
    ax.set_title(f"{model_name} Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    fig.tight_layout()
    return fig


@torch.no_grad()
def preview_predictions(
    model: nn.Module,
    loader: DataLoader,
    class_names: list[str],
    device: torch.device,
    num_samples: int = 6,
):
    model.eval()
    images, labels = next(iter(loader))
    images = images[:num_samples].to(device)
    labels = labels[:num_samples]
    logits = model(images)
    predictions = logits.argmax(dim=1).cpu()

    fig, axes = plt.subplots(1, num_samples, figsize=(3 * num_samples, 3))
    axes = np.array(axes).reshape(-1)
    for index in range(num_samples):
        axes[index].imshow(denormalize_image(images[index].cpu()).squeeze(0), cmap="gray")
        axes[index].set_title(
            f"Pred: {class_names[predictions[index]]}\nTrue: {class_names[labels[index]]}"
        )
        axes[index].axis("off")
    fig.tight_layout()
    return fig


def apply_perturbation(
    images: torch.Tensor,
    perturbation: str,
    severity: float = 0.1,
) -> torch.Tensor:
    if perturbation == "gaussian_noise":
        noise = torch.randn_like(images) * severity
        return torch.clamp(images + noise, -1.0, 1.0)
    if perturbation == "blur":
        blur_transform = transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))
        return blur_transform(images)
    if perturbation == "rotation":
        rotated = [TF.rotate(image, angle=float(severity), interpolation=InterpolationMode.BILINEAR) for image in images]
        return torch.stack(rotated)
    raise ValueError(f"Unsupported perturbation: {perturbation}")


@torch.no_grad()
def evaluate_robustness(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    perturbations: list[dict[str, Any]],
) -> pd.DataFrame:
    model.eval()
    rows: list[dict[str, Any]] = []

    for perturbation_cfg in perturbations:
        y_true: list[int] = []
        y_pred: list[int] = []
        for images, labels in loader:
            images = apply_perturbation(
                images,
                perturbation=perturbation_cfg["name"],
                severity=perturbation_cfg.get("severity", 0.1),
            ).to(device)
            labels = labels.to(device)
            predictions = model(images).argmax(dim=1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(predictions.cpu().tolist())

        metric_row = compute_metrics(y_true, y_pred)
        metric_row["perturbation"] = perturbation_cfg["name"]
        metric_row["severity"] = perturbation_cfg.get("severity", 0.1)
        rows.append(metric_row)

    return pd.DataFrame(rows)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def set_nested_config_value(config: dict[str, Any], dotted_key: str, value: Any) -> dict[str, Any]:
    updated = copy.deepcopy(config)
    cursor = updated
    keys = dotted_key.split(".")
    for key in keys[:-1]:
        cursor = cursor.setdefault(key, {})
    cursor[keys[-1]] = value
    return updated


def run_single_factor_search(
    model_name: str,
    model_builder: Callable[[dict[str, Any]], nn.Module],
    base_config: dict[str, Any],
    factor_name: str,
    candidate_values: list[Any],
    loaders: dict[str, Any],
    device: torch.device,
    output_dir: str | Path,
) -> dict[str, Any]:
    """
    Search one factor at a time so the experiment cost stays manageable for coursework.
    """
    rows: list[dict[str, Any]] = []
    best_config = copy.deepcopy(base_config)
    best_result: dict[str, Any] | None = None
    best_metric = float("-inf")

    for candidate in candidate_values:
        trial_config = set_nested_config_value(base_config, factor_name, candidate)
        result = run_training_experiment(
            model_name=f"{model_name}_{factor_name.replace('.', '_')}_{candidate}",
            model_builder=model_builder,
            config=trial_config,
            loaders=loaders,
            device=device,
            output_dir=output_dir,
        )
        metric = result["summary"]["best_valid_accuracy"]
        rows.append(
            {
                "factor": factor_name,
                "candidate": str(candidate),
                "best_valid_accuracy": metric,
                "best_epoch": result["summary"]["best_epoch"],
                "training_time_sec": result["summary"]["training_time_sec"],
            }
        )
        if metric > best_metric:
            best_metric = metric
            best_config = trial_config
            best_result = result

    return {
        "results": pd.DataFrame(rows),
        "best_config": best_config,
        "best_result": best_result,
    }


def run_small_sample_experiment(
    model_name: str,
    model_builder: Callable[[dict[str, Any]], nn.Module],
    base_config: dict[str, Any],
    runtime_config: dict[str, Any],
    sample_ratios: list[float],
    device: torch.device,
    output_dir: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Reuse the same train loop for the 30% / 50% / 100% comparison."""
    rows: list[dict[str, Any]] = []
    results: dict[str, Any] = {}

    for ratio in sample_ratios:
        loaders = load_emnist_balanced(
            data_dir=runtime_config["data_dir"],
            batch_size=runtime_config["batch_size"],
            valid_ratio=runtime_config["valid_ratio"],
            num_workers=runtime_config["num_workers"],
            subset_ratio=ratio,
            augment=runtime_config.get("augment", False),
            rotation_deg=runtime_config.get("rotation_deg", 10.0),
            noise_std=runtime_config.get("noise_std", 0.0),
            blur=runtime_config.get("blur", False),
            seed=runtime_config.get("seed", 42),
        )

        experiment = run_training_experiment(
            model_name=f"{model_name}_{int(ratio * 100)}pct",
            model_builder=model_builder,
            config=base_config,
            loaders=loaders,
            device=device,
            output_dir=Path(output_dir) / "small_sample",
        )
        test_metrics = evaluate_on_test(experiment["model"], loaders["test_loader"], device=device)
        rows.append(
            {
                "sample_ratio": ratio,
                "train_samples": len(loaders["train_dataset"]),
                "best_valid_accuracy": experiment["summary"]["best_valid_accuracy"],
                "test_accuracy": test_metrics["accuracy"],
                "test_f1_macro": test_metrics["f1_macro"],
            }
        )
        results[f"{int(ratio * 100)}pct"] = {
            "experiment": experiment,
            "test_metrics": test_metrics,
        }

    return pd.DataFrame(rows), results


def get_default_runtime_config(project_dir: str | Path) -> dict[str, Any]:
    project_dir = Path(project_dir).resolve()
    return {
        "project_dir": project_dir,
        "data_dir": project_dir / "data",
        "batch_size": 128,
        "valid_ratio": 0.1,
        "num_workers": 0,
        "seed": 42,
        "augment": True,
        "rotation_deg": 10.0,
        "noise_std": 0.02,
        "blur": False,
    }


def get_default_mlp_config() -> dict[str, Any]:
    return {
        "input_dim": 28 * 28,
        "hidden_dims": [512, 256, 128],
        "num_classes": 47,
        "activation": "relu",
        "normalization": "batchnorm",
        "dropout": 0.3,
        "optimizer": "adam",
        "learning_rate": 1e-3,
        "scheduler": "StepLR",
        "scheduler_params": {"step_size": 5, "gamma": 0.5},
        "weight_decay": 1e-4,
        "l1_lambda": 0.0,
        "epochs": 12,
        "early_stopping_patience": 4,
    }


def get_default_cnn_config() -> dict[str, Any]:
    return {
        "channels": [32, 64, 128],
        "num_classes": 47,
        "activation": "relu",
        "normalization": "batchnorm",
        "dropout": 0.2,
        "optimizer": "adam",
        "learning_rate": 1e-3,
        "scheduler": "StepLR",
        "scheduler_params": {"step_size": 5, "gamma": 0.5},
        "weight_decay": 1e-4,
        "epochs": 15,
        "early_stopping_patience": 5,
    }


def get_default_resnet_config() -> dict[str, Any]:
    return {
        "base_channels": 32,
        "num_classes": 47,
        "activation": "relu",
        "normalization": "batchnorm",
        "optimizer": "adam",
        "learning_rate": 1e-3,
        "scheduler": "CosineAnnealingLR",
        "scheduler_params": {"t_max": 12, "eta_min": 1e-5},
        "weight_decay": 1e-4,
        "epochs": 15,
        "early_stopping_patience": 5,
    }


def get_default_vit_config() -> dict[str, Any]:
    return {
        "image_size": 28,
        "patch_size": 4,
        "embed_dim": 128,
        "num_heads": 4,
        "depth": 4,
        "mlp_ratio": 2.0,
        "dropout": 0.1,
        "num_classes": 47,
        "optimizer": "adam",
        "learning_rate": 3e-4,
        "scheduler": "CosineAnnealingLR",
        "scheduler_params": {"t_max": 12, "eta_min": 1e-5},
        "weight_decay": 1e-4,
        "epochs": 15,
        "early_stopping_patience": 5,
    }
