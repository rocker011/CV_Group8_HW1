from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import hw1_framework as hw


PROJECT_DIR = Path(__file__).resolve().parent


def apply_config_updates(config: dict, updates: dict) -> dict:
    """Merge nested config patches without requiring the latest shared framework."""
    updated = json.loads(json.dumps(config))

    def _merge(target: dict, patch: dict) -> None:
        for key, value in patch.items():
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                _merge(target[key], value)
            else:
                target[key] = value

    _merge(updated, updates)
    return updated


def run_candidate_search(
    model_name: str,
    model_builder,
    base_config: dict,
    search_name: str,
    candidates: list[dict],
    loaders: dict,
    device,
    output_dir: Path,
) -> dict:
    """
    Fallback wrapper for older framework versions that do not expose run_candidate_search yet.
    """
    if hasattr(hw, "run_candidate_search"):
        return hw.run_candidate_search(
            model_name=model_name,
            model_builder=model_builder,
            base_config=base_config,
            search_name=search_name,
            candidates=candidates,
            loaders=loaders,
            device=device,
            output_dir=output_dir,
        )

    rows = []
    best_config = json.loads(json.dumps(base_config))
    best_result = None
    best_metric = float("-inf")

    for candidate in candidates:
        candidate_name = str(candidate["name"])
        trial_config = apply_config_updates(base_config, candidate.get("updates", {}))
        result = hw.run_training_experiment(
            model_name=f"{model_name}_{search_name}_{candidate_name}",
            model_builder=model_builder,
            config=trial_config,
            loaders=loaders,
            device=device,
            output_dir=output_dir,
        )
        metric = result["summary"]["best_valid_accuracy"]
        rows.append(
            {
                "search_name": search_name,
                "candidate": candidate_name,
                "best_valid_accuracy": metric,
                "best_valid_loss": result["summary"]["best_valid_loss"],
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


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)


def build_loaders(runtime_config: dict, subset_ratio: float = 1.0) -> dict:
    return hw.load_emnist_balanced(
        data_dir=runtime_config["data_dir"],
        batch_size=runtime_config["batch_size"],
        valid_ratio=runtime_config["valid_ratio"],
        num_workers=runtime_config["num_workers"],
        subset_ratio=subset_ratio,
        augment=runtime_config["augment"],
        rotation_deg=runtime_config["rotation_deg"],
        noise_std=runtime_config["noise_std"],
        blur=runtime_config["blur"],
        seed=runtime_config["seed"],
    )


def main() -> None:
    project_paths = hw.ensure_project_dirs(PROJECT_DIR)
    runtime_config = hw.get_default_runtime_config(PROJECT_DIR)
    hw.set_seed(runtime_config["seed"])
    device = hw.get_device()

    print(f"Project directory: {PROJECT_DIR}")
    print(f"Device: {device}")

    baseline_config = hw.get_default_mlp_config()
    full_loaders = build_loaders(runtime_config, subset_ratio=1.0)

    baseline_checkpoint = project_paths["models"] / "mlp_baseline_best.pt"
    if baseline_checkpoint.exists():
        print("Using existing baseline checkpoint.")
        baseline_model = hw.build_mlp(baseline_config).to(device)
        hw.load_checkpoint(baseline_model, baseline_checkpoint, device)
    else:
        print("Training baseline checkpoint because none exists yet.")
        baseline_run = hw.run_training_experiment(
            model_name="mlp_baseline",
            model_builder=hw.build_mlp,
            config=baseline_config,
            loaders=full_loaders,
            device=device,
            output_dir=project_paths["models"],
        )
        baseline_model = baseline_run["model"]

    baseline_valid_metrics = hw.evaluate_on_test(baseline_model, full_loaders["valid_loader"], device)
    baseline_test_metrics = hw.evaluate_on_test(baseline_model, full_loaders["test_loader"], device)

    search_runtime = dict(runtime_config)
    search_runtime["batch_size"] = 256
    search_loaders = build_loaders(search_runtime, subset_ratio=0.5)

    search_config = hw.get_default_mlp_config()
    search_config["epochs"] = 8
    search_config["early_stopping_patience"] = 3

    search_tables: list[pd.DataFrame] = []
    tuned_config = dict(search_config)

    search_spaces = [
        (
            "scheduler",
            [
                {
                    "name": "StepLR",
                    "updates": {
                        "scheduler": "StepLR",
                        "scheduler_params": {"step_size": 3, "gamma": 0.5},
                    },
                },
                {
                    "name": "CosineAnnealingLR",
                    "updates": {
                        "scheduler": "CosineAnnealingLR",
                        "scheduler_params": {"t_max": 8, "eta_min": 1e-5},
                    },
                },
                {
                    "name": "ReduceLROnPlateau",
                    "updates": {
                        "scheduler": "ReduceLROnPlateau",
                        "scheduler_params": {"mode": "min", "factor": 0.5, "patience": 1},
                    },
                },
            ],
        ),
        (
            "activation",
            [
                {"name": "relu", "updates": {"activation": "relu"}},
                {"name": "leaky_relu", "updates": {"activation": "leaky_relu"}},
                {"name": "gelu", "updates": {"activation": "gelu"}},
            ],
        ),
        (
            "optimizer",
            [
                {"name": "adam_1e-3", "updates": {"optimizer": "adam", "learning_rate": 1e-3}},
                {"name": "sgd_5e-2", "updates": {"optimizer": "sgd", "learning_rate": 5e-2}},
                {"name": "rmsprop_1e-3", "updates": {"optimizer": "rmsprop", "learning_rate": 1e-3}},
            ],
        ),
        (
            "normalization",
            [
                {"name": "batchnorm", "updates": {"normalization": "batchnorm"}},
                {"name": "layernorm", "updates": {"normalization": "layernorm"}},
                {"name": "none", "updates": {"normalization": "none"}},
            ],
        ),
        (
            "dropout",
            [
                {"name": "dropout_0.0", "updates": {"dropout": 0.0}},
                {"name": "dropout_0.3", "updates": {"dropout": 0.3}},
                {"name": "dropout_0.5", "updates": {"dropout": 0.5}},
            ],
        ),
        (
            "regularization",
            [
                {"name": "none", "updates": {"l1_lambda": 0.0, "weight_decay": 0.0}},
                {"name": "l1", "updates": {"l1_lambda": 1e-6, "weight_decay": 0.0}},
                {"name": "l2", "updates": {"l1_lambda": 0.0, "weight_decay": 1e-4}},
            ],
        ),
    ]

    print("Starting MLP factor search...")
    for search_name, candidates in search_spaces:
        result = run_candidate_search(
            model_name="mlp",
            model_builder=hw.build_mlp,
            base_config=tuned_config,
            search_name=search_name,
            candidates=candidates,
            loaders=search_loaders,
            device=device,
            output_dir=project_paths["models"] / "search",
        )
        result_df = result["results"]
        result_df["selected"] = result_df["best_valid_accuracy"] == result_df["best_valid_accuracy"].max()
        search_tables.append(result_df)
        tuned_config = result["best_config"]
        print(f"Finished search block: {search_name}")
        print(result_df.to_string(index=False))
        print()

    search_results = pd.concat(search_tables, ignore_index=True)
    search_results_path = project_paths["results"] / "mlp_search_results.csv"
    search_results.to_csv(search_results_path, index=False)
    save_json(project_paths["results"] / "mlp_best_config.json", tuned_config)

    final_config = dict(tuned_config)
    final_config["epochs"] = 15
    final_config["early_stopping_patience"] = 5

    print("Training final MLP on the full training split...")
    final_run = hw.run_training_experiment(
        model_name="mlp_final",
        model_builder=hw.build_mlp,
        config=final_config,
        loaders=full_loaders,
        device=device,
        output_dir=project_paths["models"],
    )

    final_valid_metrics = hw.evaluate_on_test(final_run["model"], full_loaders["valid_loader"], device)
    final_test_metrics = hw.evaluate_on_test(final_run["model"], full_loaders["test_loader"], device)

    history_df = pd.DataFrame(final_run["history"])
    history_df.to_csv(project_paths["results"] / "mlp_final_history.csv", index=False)

    curve_fig = hw.plot_training_curves(final_run["history"], "MLP Final")
    curve_fig.savefig(project_paths["figures"] / "mlp_final_curves.png", dpi=200, bbox_inches="tight")
    plt.close(curve_fig)

    preview_fig = hw.preview_predictions(
        model=final_run["model"],
        loader=full_loaders["test_loader"],
        class_names=full_loaders["class_names"],
        device=device,
        num_samples=6,
    )
    preview_fig.savefig(project_paths["figures"] / "mlp_final_predictions.png", dpi=200, bbox_inches="tight")
    plt.close(preview_fig)

    confusion_fig = hw.plot_confusion_matrix_from_preds(
        y_true=final_test_metrics["y_true"],
        y_pred=final_test_metrics["y_pred"],
        class_names=full_loaders["class_names"],
        model_name="MLP Final",
    )
    confusion_fig.savefig(project_paths["figures"] / "mlp_final_confusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.close(confusion_fig)

    print("Running small-sample experiment...")
    small_sample_df, _ = hw.run_small_sample_experiment(
        model_name="mlp_final",
        model_builder=hw.build_mlp,
        base_config=final_config,
        runtime_config=runtime_config,
        sample_ratios=[0.3, 0.5, 1.0],
        device=device,
        output_dir=project_paths["models"],
    )
    small_sample_df.to_csv(project_paths["results"] / "mlp_small_sample_results.csv", index=False)

    small_sample_fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(small_sample_df["sample_ratio"], small_sample_df["test_accuracy"], marker="o", label="Test Accuracy")
    ax.plot(small_sample_df["sample_ratio"], small_sample_df["test_f1_macro"], marker="s", label="Macro F1")
    ax.set_xlabel("Training Data Ratio")
    ax.set_ylabel("Score")
    ax.set_title("MLP Small-Sample Performance")
    ax.legend()
    ax.grid(alpha=0.3)
    small_sample_fig.tight_layout()
    small_sample_fig.savefig(project_paths["figures"] / "mlp_small_sample.png", dpi=200, bbox_inches="tight")
    plt.close(small_sample_fig)

    summary_payload = {
        "baseline_valid_metrics": hw.summarize_metrics(baseline_valid_metrics),
        "baseline_test_metrics": hw.summarize_metrics(baseline_test_metrics),
        "final_config": final_config,
        "final_summary": final_run["summary"],
        "final_valid_metrics": hw.summarize_metrics(final_valid_metrics),
        "final_test_metrics": hw.summarize_metrics(final_test_metrics),
    }
    save_json(project_paths["results"] / "mlp_experiment_summary.json", summary_payload)

    summary_rows = [
        {"stage": "baseline_valid", **hw.summarize_metrics(baseline_valid_metrics)},
        {"stage": "baseline_test", **hw.summarize_metrics(baseline_test_metrics)},
        {"stage": "final_valid", **hw.summarize_metrics(final_valid_metrics)},
        {"stage": "final_test", **hw.summarize_metrics(final_test_metrics)},
    ]
    pd.DataFrame(summary_rows).to_csv(project_paths["results"] / "mlp_metric_summary.csv", index=False)

    print("MLP pipeline completed.")
    print("Best config:")
    print(json.dumps(final_config, indent=2, ensure_ascii=False))
    print("Final test metrics:")
    print(hw.summarize_metrics(final_test_metrics))
    print("Small-sample results:")
    print(small_sample_df.to_string(index=False))


if __name__ == "__main__":
    main()
