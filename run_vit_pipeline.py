from __future__ import annotations

import copy
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import hw1_framework as hw


PROJECT_DIR = Path(__file__).resolve().parent
progress_iter = getattr(hw, "progress_iter", lambda iterable, **kwargs: iterable)


def apply_config_updates(config: dict, updates: dict) -> dict:
    """Merge nested config patches without requiring the latest shared framework."""
    updated = copy.deepcopy(config)

    def _merge(target: dict, patch: dict) -> None:
        for key, value in patch.items():
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                _merge(target[key], value)
            else:
                target[key] = copy.deepcopy(value)

    _merge(updated, updates)
    return updated


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
    best_config = copy.deepcopy(base_config)
    best_result = None
    best_metric = float("-inf")

    candidate_iterator = progress_iter(
        candidates,
        desc=f"{model_name} {search_name}",
        leave=False,
        dynamic_ncols=True,
    )

    for candidate in candidate_iterator:
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

        if hasattr(candidate_iterator, "set_postfix"):
            candidate_iterator.set_postfix(
                candidate=candidate_name,
                best_acc=f"{best_metric:.4f}",
            )

    if hasattr(candidate_iterator, "close"):
        candidate_iterator.close()

    return {
        "results": pd.DataFrame(rows),
        "best_config": best_config,
        "best_result": best_result,
    }


def evaluate_on_validation(model, loader, device) -> dict:
    result = hw.evaluate_one_epoch(
        model=model,
        loader=loader,
        criterion=hw.build_loss_fn(),
        device=device,
        collect_predictions=True,
    )
    result.update(hw.compute_metrics(result["y_true"], result["y_pred"]))
    return result


def run_small_sample_test_experiment(
    model_name: str,
    model_builder,
    base_config: dict,
    runtime_config: dict,
    sample_ratios: list[float],
    device,
    output_dir: Path,
) -> tuple[pd.DataFrame, dict]:
    """Reuse the shared train loop and compare the trained models on the fixed test split."""
    rows: list[dict] = []
    results: dict[str, dict] = {}

    ratio_iterator = progress_iter(
        sample_ratios,
        desc=f"{model_name} small-sample",
        leave=False,
        dynamic_ncols=True,
    )

    for ratio in ratio_iterator:
        loaders = build_loaders(runtime_config, subset_ratio=ratio)
        experiment = hw.run_training_experiment(
            model_name=f"{model_name}_{int(ratio * 100)}pct",
            model_builder=model_builder,
            config=copy.deepcopy(base_config),
            loaders=loaders,
            device=device,
            output_dir=Path(output_dir) / "small_sample",
        )
        test_metrics = hw.evaluate_on_test(experiment["model"], loaders["test_loader"], device)
        rows.append(
            {
                "sample_ratio": ratio,
                "train_samples": len(loaders["train_dataset"]),
                "best_valid_accuracy": experiment["summary"]["best_valid_accuracy"],
                "best_valid_loss": experiment["summary"]["best_valid_loss"],
                "test_accuracy": test_metrics["accuracy"],
                "test_f1_macro": test_metrics["f1_macro"],
            }
        )
        results[f"{int(ratio * 100)}pct"] = {
            "experiment": experiment,
            "test_metrics": test_metrics,
        }

        if hasattr(ratio_iterator, "set_postfix"):
            ratio_iterator.set_postfix(
                ratio=f"{int(ratio * 100)}%",
                test_acc=f"{test_metrics['accuracy']:.4f}",
            )

    if hasattr(ratio_iterator, "close"):
        ratio_iterator.close()

    return pd.DataFrame(rows), results


def plot_small_sample_test(summary_df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(summary_df["sample_ratio"], summary_df["test_accuracy"], marker="o")
    axes[0].set_xlabel("Training Data Ratio")
    axes[0].set_ylabel("Test Accuracy")
    axes[0].set_title("ViT Small-Sample Test Accuracy")
    axes[0].grid(alpha=0.3)

    axes[1].plot(summary_df["sample_ratio"], summary_df["best_valid_loss"], marker="s", color="tab:orange")
    axes[1].set_xlabel("Training Data Ratio")
    axes[1].set_ylabel("Best Validation Loss")
    axes[1].set_title("ViT Small-Sample Best Validation Loss")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    return fig


def make_runtime_variant(
    runtime_config: dict,
    *,
    augment: bool,
    rotation_deg: float | None = None,
    noise_std: float | None = None,
    blur: bool | None = None,
) -> dict:
    variant = dict(runtime_config)
    variant["augment"] = augment
    if rotation_deg is not None:
        variant["rotation_deg"] = rotation_deg
    if noise_std is not None:
        variant["noise_std"] = noise_std
    if blur is not None:
        variant["blur"] = blur
    return variant


def summarize_runtime_augmentation(runtime_config: dict) -> dict:
    return {
        "augment": runtime_config["augment"],
        "rotation_deg": runtime_config["rotation_deg"],
        "noise_std": runtime_config["noise_std"],
        "blur": runtime_config["blur"],
    }


def run_augmentation_comparison(
    model_builder,
    base_config: dict,
    runtime_config: dict,
    device,
    output_dir: Path,
) -> tuple[pd.DataFrame, dict[str, dict], str]:
    """Compare the searched-best ViT config with and without training-time augmentation."""
    rows: list[dict] = []
    results: dict[str, dict] = {}

    variants = [
        {
            "setting": "no_augmentation",
            "model_name": "vit_final_noaug",
            "runtime_config": make_runtime_variant(
                runtime_config,
                augment=False,
                rotation_deg=0.0,
                noise_std=0.0,
                blur=False,
            ),
        },
        {
            "setting": "with_augmentation",
            "model_name": "vit_final_aug",
            "runtime_config": make_runtime_variant(
                runtime_config,
                augment=True,
            ),
        },
    ]

    variant_iterator = progress_iter(
        variants,
        desc="vit augmentation compare",
        leave=True,
        dynamic_ncols=True,
    )

    best_setting = ""
    best_valid_accuracy = float("-inf")

    for variant in variant_iterator:
        variant_loaders = build_loaders(variant["runtime_config"], subset_ratio=1.0)
        experiment = hw.run_training_experiment(
            model_name=variant["model_name"],
            model_builder=model_builder,
            config=copy.deepcopy(base_config),
            loaders=variant_loaders,
            device=device,
            output_dir=output_dir,
        )
        valid_metrics = evaluate_on_validation(experiment["model"], variant_loaders["valid_loader"], device)
        test_metrics = hw.evaluate_on_test(experiment["model"], variant_loaders["test_loader"], device)

        row = {
            "setting": variant["setting"],
            **summarize_runtime_augmentation(variant["runtime_config"]),
            "best_epoch": experiment["summary"]["best_epoch"],
            "best_valid_accuracy": experiment["summary"]["best_valid_accuracy"],
            "best_valid_loss": experiment["summary"]["best_valid_loss"],
            "valid_accuracy": valid_metrics["accuracy"],
            "valid_f1_macro": valid_metrics["f1_macro"],
            "test_accuracy": test_metrics["accuracy"],
            "test_f1_macro": test_metrics["f1_macro"],
            "training_time_sec": experiment["summary"]["training_time_sec"],
            "peak_process_memory_mb": experiment["summary"]["peak_process_memory_mb"],
            "peak_gpu_memory_mb": experiment["summary"]["peak_gpu_memory_mb"],
        }
        rows.append(row)
        results[variant["setting"]] = {
            "experiment": experiment,
            "runtime_config": variant["runtime_config"],
            "valid_metrics": valid_metrics,
            "test_metrics": test_metrics,
        }

        if valid_metrics["accuracy"] > best_valid_accuracy:
            best_valid_accuracy = valid_metrics["accuracy"]
            best_setting = variant["setting"]

        if hasattr(variant_iterator, "set_postfix"):
            variant_iterator.set_postfix(
                setting=variant["setting"],
                valid_acc=f"{valid_metrics['accuracy']:.4f}",
                test_acc=f"{test_metrics['accuracy']:.4f}",
            )

    if hasattr(variant_iterator, "close"):
        variant_iterator.close()

    comparison_df = pd.DataFrame(rows)
    comparison_df["selected_final"] = comparison_df["setting"] == best_setting
    return comparison_df, results, best_setting


def plot_augmentation_comparison(comparison_df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    settings = comparison_df["setting"].tolist()
    x = range(len(settings))
    width = 0.35

    axes[0].bar([idx - width / 2 for idx in x], comparison_df["valid_accuracy"], width=width, label="Valid Accuracy")
    axes[0].bar([idx + width / 2 for idx in x], comparison_df["test_accuracy"], width=width, label="Test Accuracy")
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(settings, rotation=10)
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("ViT Augmentation Accuracy Comparison")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar([idx - width / 2 for idx in x], comparison_df["valid_f1_macro"], width=width, label="Valid Macro F1")
    axes[1].bar([idx + width / 2 for idx in x], comparison_df["test_f1_macro"], width=width, label="Test Macro F1")
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(settings, rotation=10)
    axes[1].set_ylabel("Macro F1")
    axes[1].set_title("ViT Augmentation F1 Comparison")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    return fig


def main() -> None:
    project_paths = hw.ensure_project_dirs(PROJECT_DIR)
    runtime_config = hw.get_default_runtime_config(PROJECT_DIR)
    hw.set_seed(runtime_config["seed"])
    device = hw.get_device()

    print(f"Project directory: {PROJECT_DIR}")
    print(f"Device: {device}")

    full_loaders = build_loaders(runtime_config, subset_ratio=1.0)
    baseline_config = hw.get_default_vit_config()
    baseline_parameter_count = hw.count_parameters(hw.build_vit(baseline_config))

    baseline_checkpoint = project_paths["models"] / "vit_baseline_best.pt"
    if baseline_checkpoint.exists():
        print("Using existing ViT baseline checkpoint.")
        baseline_model = hw.build_vit(baseline_config).to(device)
        hw.load_checkpoint(baseline_model, baseline_checkpoint, device)
    else:
        print("Training ViT baseline checkpoint because none exists yet.")
        baseline_run = hw.run_training_experiment(
            model_name="vit_baseline",
            model_builder=hw.build_vit,
            config=baseline_config,
            loaders=full_loaders,
            device=device,
            output_dir=project_paths["models"],
        )
        baseline_model = baseline_run["model"]

    baseline_valid_metrics = evaluate_on_validation(baseline_model, full_loaders["valid_loader"], device)
    baseline_test_metrics = hw.evaluate_on_test(baseline_model, full_loaders["test_loader"], device)

    search_runtime = dict(runtime_config)
    search_runtime["batch_size"] = 256
    search_loaders = build_loaders(search_runtime, subset_ratio=0.4)

    search_config = hw.get_default_vit_config()
    search_config["epochs"] = 8
    search_config["early_stopping_patience"] = 3
    search_config["scheduler_params"] = {"t_max": 8, "eta_min": 1e-5}

    search_tables: list[pd.DataFrame] = []
    tuned_config = copy.deepcopy(search_config)

    search_spaces = [
        (
            "architecture",
            [
                {
                    "name": "patch7_embed96_depth4_heads4",
                    "updates": {
                        "patch_size": 7,
                        "embed_dim": 96,
                        "num_heads": 4,
                        "depth": 4,
                        "mlp_ratio": 2.0,
                    },
                },
                {
                    "name": "patch4_embed128_depth4_heads4",
                    "updates": {
                        "patch_size": 4,
                        "embed_dim": 128,
                        "num_heads": 4,
                        "depth": 4,
                        "mlp_ratio": 2.0,
                    },
                },
                {
                    "name": "patch4_embed160_depth6_heads5",
                    "updates": {
                        "patch_size": 4,
                        "embed_dim": 160,
                        "num_heads": 5,
                        "depth": 6,
                        "mlp_ratio": 2.0,
                    },
                },
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
                        "scheduler_params": {"mode": "min", "factor": 0.5, "patience": 1},
                    },
                },
            ],
        ),
        (
            "activation",
            [
                {"name": "relu", "updates": {"activation": "relu"}},
                {"name": "gelu", "updates": {"activation": "gelu"}},
                {"name": "elu", "updates": {"activation": "elu"}},
            ],
        ),
        (
            "optimizer",
            [
                {"name": "adamw_3e-4", "updates": {"optimizer": "adamw", "learning_rate": 3e-4}},
                {"name": "adam_3e-4", "updates": {"optimizer": "adam", "learning_rate": 3e-4}},
                {"name": "rmsprop_1e-3", "updates": {"optimizer": "rmsprop", "learning_rate": 1e-3}},
            ],
        ),
        (
            "normalization",
            [
                {"name": "layernorm", "updates": {"normalization": "layernorm"}},
                {"name": "batchnorm", "updates": {"normalization": "batchnorm"}},
                {"name": "none", "updates": {"normalization": "none"}},
            ],
        ),
        (
            "dropout",
            [
                {
                    "name": "dropout_0.0",
                    "updates": {
                        "dropout": 0.0,
                        "embedding_dropout": 0.0,
                        "head_dropout": 0.0,
                        "attention_dropout": 0.0,
                    },
                },
                {
                    "name": "dropout_0.1",
                    "updates": {
                        "dropout": 0.1,
                        "embedding_dropout": 0.1,
                        "head_dropout": 0.1,
                        "attention_dropout": 0.0,
                    },
                },
                {
                    "name": "dropout_0.2",
                    "updates": {
                        "dropout": 0.2,
                        "embedding_dropout": 0.2,
                        "head_dropout": 0.2,
                        "attention_dropout": 0.1,
                    },
                },
            ],
        ),
        (
            "regularization",
            [
                {"name": "none", "updates": {"l1_lambda": 0.0, "weight_decay": 0.0}},
                {"name": "l1", "updates": {"l1_lambda": 1e-7, "weight_decay": 0.0}},
                {"name": "l2", "updates": {"l1_lambda": 0.0, "weight_decay": 5e-4}},
            ],
        ),
    ]

    print("Starting ViT factor search...")
    search_block_iterator = progress_iter(
        search_spaces,
        desc="vit search blocks",
        leave=True,
        dynamic_ncols=True,
    )

    for search_name, candidates in search_block_iterator:
        result = run_candidate_search(
            model_name="vit",
            model_builder=hw.build_vit,
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
        if hasattr(search_block_iterator, "set_postfix"):
            search_block_iterator.set_postfix(
                block=search_name,
                best_acc=f"{result_df['best_valid_accuracy'].max():.4f}",
            )
        print(f"Finished search block: {search_name}")
        print(result_df.to_string(index=False))
        print()

    if hasattr(search_block_iterator, "close"):
        search_block_iterator.close()

    search_results = pd.concat(search_tables, ignore_index=True)
    search_results.to_csv(project_paths["results"] / "vit_search_results.csv", index=False)
    save_json(project_paths["results"] / "vit_best_config.json", tuned_config)

    final_config = copy.deepcopy(tuned_config)
    final_config["epochs"] = 18
    final_config["early_stopping_patience"] = 5
    if final_config.get("scheduler") == "CosineAnnealingLR":
        final_config["scheduler_params"] = {"t_max": 18, "eta_min": 1e-5}

    final_parameter_count = hw.count_parameters(hw.build_vit(final_config))

    print("Running augmentation comparison on the best ViT hyperparameter combination...")
    augmentation_df, augmentation_runs, selected_training_setting = run_augmentation_comparison(
        model_builder=hw.build_vit,
        base_config=final_config,
        runtime_config=runtime_config,
        device=device,
        output_dir=project_paths["models"],
    )
    augmentation_df.to_csv(project_paths["results"] / "vit_augmentation_comparison.csv", index=False)

    augmentation_fig = plot_augmentation_comparison(augmentation_df)
    augmentation_fig.savefig(
        project_paths["figures"] / "vit_augmentation_comparison.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(augmentation_fig)

    selected_final_run = augmentation_runs[selected_training_setting]
    final_run = selected_final_run["experiment"]
    final_valid_metrics = selected_final_run["valid_metrics"]
    final_test_metrics = selected_final_run["test_metrics"]
    final_runtime_config = selected_final_run["runtime_config"]

    final_checkpoint_path = hw.save_checkpoint(final_run["model"], project_paths["models"] / "vit_final_best.pt")
    final_summary = dict(final_run["summary"])
    final_summary["checkpoint_path"] = final_checkpoint_path
    final_summary["training_setting"] = selected_training_setting
    final_summary.update(summarize_runtime_augmentation(final_runtime_config))

    history_df = pd.DataFrame(final_run["history"])
    history_df.to_csv(project_paths["results"] / "vit_final_history.csv", index=False)

    curve_fig = hw.plot_training_curves(final_run["history"], "ViT Final")
    curve_fig.savefig(project_paths["figures"] / "vit_final_curves.png", dpi=200, bbox_inches="tight")
    plt.close(curve_fig)

    print("Running ViT small-sample experiment with test-set comparison...")
    small_sample_df, _ = run_small_sample_test_experiment(
        model_name="vit_final",
        model_builder=hw.build_vit,
        base_config=final_config,
        runtime_config=final_runtime_config,
        sample_ratios=[0.3, 0.5, 1.0],
        device=device,
        output_dir=project_paths["models"],
    )
    small_sample_df.to_csv(project_paths["results"] / "vit_small_sample_results.csv", index=False)

    small_sample_fig = plot_small_sample_test(small_sample_df)
    small_sample_fig.savefig(project_paths["figures"] / "vit_small_sample.png", dpi=200, bbox_inches="tight")
    plt.close(small_sample_fig)

    summary_payload = {
        "baseline_parameter_count": baseline_parameter_count,
        "baseline_valid_metrics": hw.summarize_metrics(baseline_valid_metrics),
        "baseline_test_metrics": hw.summarize_metrics(baseline_test_metrics),
        "final_parameter_count": final_parameter_count,
        "final_config": final_config,
        "augmentation_comparison": augmentation_df.to_dict(orient="records"),
        "selected_training_setting": selected_training_setting,
        "final_runtime_config": summarize_runtime_augmentation(final_runtime_config),
        "final_summary": final_summary,
        "final_valid_metrics": hw.summarize_metrics(final_valid_metrics),
        "final_test_metrics": hw.summarize_metrics(final_test_metrics),
    }
    save_json(project_paths["results"] / "vit_experiment_summary.json", summary_payload)

    summary_rows = [
        {
            "stage": "baseline_valid",
            "parameter_count": baseline_parameter_count,
            **hw.summarize_metrics(baseline_valid_metrics),
        },
        {
            "stage": "baseline_test",
            "parameter_count": baseline_parameter_count,
            **hw.summarize_metrics(baseline_test_metrics),
        },
        {
            "stage": "final_valid",
            "parameter_count": final_parameter_count,
            **hw.summarize_metrics(final_valid_metrics),
        },
        {
            "stage": "final_test",
            "parameter_count": final_parameter_count,
            **hw.summarize_metrics(final_test_metrics),
        },
    ]
    pd.DataFrame(summary_rows).to_csv(project_paths["results"] / "vit_metric_summary.csv", index=False)

    print("ViT pipeline completed.")
    print("Best config:")
    print(json.dumps(final_config, indent=2, ensure_ascii=False))
    print("Selected augmentation setting:")
    print(selected_training_setting)
    print("Final validation metrics:")
    print(hw.summarize_metrics(final_valid_metrics))
    print("Final test metrics:")
    print(hw.summarize_metrics(final_test_metrics))
    print("Augmentation comparison:")
    print(augmentation_df.to_string(index=False))
    print("Small-sample test results:")
    print(small_sample_df.to_string(index=False))


if __name__ == "__main__":
    main()
