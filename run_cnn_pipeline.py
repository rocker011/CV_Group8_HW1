from __future__ import annotations

import copy
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import hw1_framework as hw


PROJECT_DIR = Path(__file__).resolve().parent


def apply_config_updates(config: dict, updates: dict) -> dict:
    """Merge nested config patches while preserving the original config."""
    updated = copy.deepcopy(config)

    def _merge(target: dict, patch: dict) -> None:
        for key, value in patch.items():
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                _merge(target[key], value)
            else:
                target[key] = copy.deepcopy(value)

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
    """Fallback wrapper for framework versions without run_candidate_search."""
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
    best_config = copy.deepcopy(base_config)
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


def mark_selected_candidates(results_df: pd.DataFrame) -> pd.DataFrame:
    """Mark one selected candidate in a search block using validation metrics."""
    marked_df = results_df.copy()
    marked_df["selected"] = False

    ranked = marked_df.sort_values(
        ["best_valid_accuracy", "best_valid_loss", "training_time_sec"],
        ascending=[False, True, True],
    )
    marked_df.loc[ranked.index[0], "selected"] = True
    return marked_df


def save_json(path: Path, payload: dict) -> None:
    """Persist JSON outputs with UTF-8 encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)


def build_loaders(runtime_config: dict, subset_ratio: float = 1.0) -> dict:
    """Build train/valid/test loaders with the shared data split pipeline."""
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

    baseline_config = hw.get_default_cnn_config()
    full_loaders = build_loaders(runtime_config, subset_ratio=1.0)

    baseline_checkpoint = project_paths["models"] / "cnn_baseline_best.pt"
    if baseline_checkpoint.exists():
        print("Using existing CNN baseline checkpoint.")
        baseline_model = hw.build_cnn(baseline_config).to(device)
        hw.load_checkpoint(baseline_model, baseline_checkpoint, device)
        baseline_summary = {
            "model_name": "cnn_baseline",
            "checkpoint_path": baseline_checkpoint,
        }
    else:
        print("Training CNN baseline checkpoint because none exists yet.")
        baseline_run = hw.run_training_experiment(
            model_name="cnn_baseline",
            model_builder=hw.build_cnn,
            config=baseline_config,
            loaders=full_loaders,
            device=device,
            output_dir=project_paths["models"],
        )
        baseline_model = baseline_run["model"]
        baseline_summary = baseline_run["summary"]

    baseline_valid_metrics = hw.evaluate_on_test(
        baseline_model,
        full_loaders["valid_loader"],
        device,
    )
    baseline_test_metrics = hw.evaluate_on_test(
        baseline_model,
        full_loaders["test_loader"],
        device,
    )

    save_json(
        project_paths["results"] / "cnn_baseline.json",
        {
            "baseline_config": baseline_config,
            "baseline_summary": baseline_summary,
            "baseline_valid_metrics": hw.summarize_metrics(baseline_valid_metrics),
            "baseline_test_metrics": hw.summarize_metrics(baseline_test_metrics),
        },
    )

    search_runtime = dict(runtime_config)
    search_runtime["batch_size"] = 256
    search_loaders = build_loaders(search_runtime, subset_ratio=0.5)

    search_config = copy.deepcopy(baseline_config)
    search_config["epochs"] = 8
    search_config["early_stopping_patience"] = 3

    search_spaces = [
        (
            "channels",
            [
                {"name": "c32_64", "updates": {"channels": [32, 64]}},
                {"name": "c64_128", "updates": {"channels": [64, 128]}},
                {"name": "c32_64_128", "updates": {"channels": [32, 64, 128]}},
            ],
        ),
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
                        "scheduler_params": {
                            "mode": "min",
                            "factor": 0.5,
                            "patience": 1,
                        },
                    },
                },
            ],
        ),
        (
            "activation",
            [
                {"name": "relu", "updates": {"activation": "relu"}},
                {"name": "leaky_relu", "updates": {"activation": "leaky_relu"}},
                {"name": "elu", "updates": {"activation": "elu"}},
            ],
        ),
        (
            "optimizer",
            [
                {
                    "name": "adam_1e-3",
                    "updates": {"optimizer": "adam", "learning_rate": 1e-3},
                },
                {
                    "name": "adamw_1e-3",
                    "updates": {"optimizer": "adamw", "learning_rate": 1e-3},
                },
                {
                    "name": "sgd_5e-2",
                    "updates": {"optimizer": "sgd", "learning_rate": 5e-2},
                },
                {
                    "name": "rmsprop_1e-3",
                    "updates": {"optimizer": "rmsprop", "learning_rate": 1e-3},
                },
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
                {"name": "l1", "updates": {"l1_lambda": 1e-7, "weight_decay": 0.0}},
                {"name": "l2", "updates": {"l1_lambda": 0.0, "weight_decay": 1e-4}},
            ],
        ),
    ]

    search_tables: list[pd.DataFrame] = []
    tuned_config = copy.deepcopy(search_config)
    selected_candidates: dict[str, dict] = {}

    print("Starting CNN factor search...")
    for search_name, candidates in search_spaces:
        result = run_candidate_search(
            model_name="cnn",
            model_builder=hw.build_cnn,
            base_config=tuned_config,
            search_name=search_name,
            candidates=candidates,
            loaders=search_loaders,
            device=device,
            output_dir=project_paths["models"] / "search",
        )

        result_df = mark_selected_candidates(result["results"])
        search_tables.append(result_df)
        tuned_config = result["best_config"]

        selected_row = result_df.loc[result_df["selected"]].iloc[0].to_dict()
        selected_candidates[search_name] = selected_row

        print(f"Finished search block: {search_name}")
        print(result_df.to_string(index=False))
        print()

    search_results = pd.concat(search_tables, ignore_index=True)
    search_results_path = project_paths["results"] / "cnn_search_results.csv"
    search_results.to_csv(search_results_path, index=False)

    save_json(project_paths["results"] / "cnn_best_config.json", tuned_config)
    save_json(
        project_paths["results"] / "cnn_tuning_summary.json",
        {
            "best_config_after_search": tuned_config,
            "selected_candidates": selected_candidates,
        },
    )

    final_config = copy.deepcopy(tuned_config)
    final_config["epochs"] = 20
    final_config["early_stopping_patience"] = 5
    if final_config.get("scheduler") == "CosineAnnealingLR":
        final_config["scheduler_params"] = {"t_max": 20, "eta_min": 1e-5}

    print("Training final CNN on the full training split...")
    final_run = hw.run_training_experiment(
        model_name="cnn_final",
        model_builder=hw.build_cnn,
        config=final_config,
        loaders=full_loaders,
        device=device,
        output_dir=project_paths["models"],
    )

    final_valid_metrics = hw.evaluate_on_test(final_run["model"], full_loaders["valid_loader"], device)
    final_test_metrics = hw.evaluate_on_test(final_run["model"], full_loaders["test_loader"], device)

    history_df = pd.DataFrame(final_run["history"])
    history_df.to_csv(project_paths["results"] / "cnn_final_history.csv", index=False)

    curve_fig = hw.plot_training_curves(final_run["history"], "CNN Final")
    curve_fig.savefig(project_paths["figures"] / "cnn_final_curves.png", dpi=200, bbox_inches="tight")
    plt.close(curve_fig)

    preview_fig = hw.preview_predictions(
        model=final_run["model"],
        loader=full_loaders["test_loader"],
        class_names=full_loaders["class_names"],
        device=device,
        num_samples=6,
    )
    preview_fig.savefig(project_paths["figures"] / "cnn_final_predictions.png", dpi=200, bbox_inches="tight")
    plt.close(preview_fig)

    confusion_fig = hw.plot_confusion_matrix_from_preds(
        y_true=final_test_metrics["y_true"],
        y_pred=final_test_metrics["y_pred"],
        class_names=full_loaders["class_names"],
        model_name="CNN Final",
    )
    confusion_fig.savefig(
        project_paths["figures"] / "cnn_final_confusion_matrix.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(confusion_fig)

    print("Running CNN small-sample experiment...")
    small_sample_df, _ = hw.run_small_sample_experiment(
        model_name="cnn_final",
        model_builder=hw.build_cnn,
        base_config=final_config,
        runtime_config=runtime_config,
        sample_ratios=[0.3, 0.5, 1.0],
        device=device,
        output_dir=project_paths["models"],
    )
    small_sample_df.to_csv(project_paths["results"] / "cnn_small_sample_results.csv", index=False)

    small_sample_fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(small_sample_df["sample_ratio"], small_sample_df["test_accuracy"], marker="o", label="Test Accuracy")
    ax.plot(small_sample_df["sample_ratio"], small_sample_df["test_f1_macro"], marker="s", label="Macro F1")
    ax.set_xlabel("Training Data Ratio")
    ax.set_ylabel("Score")
    ax.set_title("CNN Small-Sample Performance")
    ax.legend()
    ax.grid(alpha=0.3)
    small_sample_fig.tight_layout()
    small_sample_fig.savefig(project_paths["figures"] / "cnn_small_sample.png", dpi=200, bbox_inches="tight")
    plt.close(small_sample_fig)

    summary_payload = {
        "baseline_valid_metrics": hw.summarize_metrics(baseline_valid_metrics),
        "baseline_test_metrics": hw.summarize_metrics(baseline_test_metrics),
        "final_config": final_config,
        "final_summary": final_run["summary"],
        "final_valid_metrics": hw.summarize_metrics(final_valid_metrics),
        "final_test_metrics": hw.summarize_metrics(final_test_metrics),
    }
    save_json(project_paths["results"] / "cnn_experiment_summary.json", summary_payload)

    metric_rows = [
        {"stage": "baseline_valid", **hw.summarize_metrics(baseline_valid_metrics)},
        {"stage": "baseline_test", **hw.summarize_metrics(baseline_test_metrics)},
        {"stage": "final_valid", **hw.summarize_metrics(final_valid_metrics)},
        {"stage": "final_test", **hw.summarize_metrics(final_test_metrics)},
    ]
    pd.DataFrame(metric_rows).to_csv(project_paths["results"] / "cnn_metric_summary.csv", index=False)

    save_json(
        project_paths["results"] / "cnn_final_metrics.json",
        {
            "final_summary": final_run["summary"],
            "final_valid_metrics": hw.summarize_metrics(final_valid_metrics),
            "final_test_metrics": hw.summarize_metrics(final_test_metrics),
        },
    )
    save_json(
        project_paths["results"] / "cnn_small_data.json",
        {"rows": small_sample_df.to_dict(orient="records")},
    )

    print("CNN pipeline completed.")
    print("Best config:")
    print(json.dumps(final_config, indent=2, ensure_ascii=False))
    print("Final test metrics:")
    print(hw.summarize_metrics(final_test_metrics))
    print("Small-sample results:")
    print(small_sample_df.to_string(index=False))


if __name__ == "__main__":
    main()
