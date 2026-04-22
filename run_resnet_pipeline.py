import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import hw1_framework as hw


def main():
    PROJECT_DIR = Path(__file__).resolve().parent
    project_paths = hw.ensure_project_dirs(PROJECT_DIR)
    device = hw.get_device()
    runtime_config = hw.get_default_runtime_config(PROJECT_DIR)

    print("Loading datasets...")
    loaders = hw.load_emnist_balanced(
        data_dir=runtime_config["data_dir"],
        batch_size=runtime_config["batch_size"],
        valid_ratio=runtime_config["valid_ratio"],
        num_workers=runtime_config["num_workers"],
        seed=runtime_config["seed"],
        augment=runtime_config["augment"],
        rotation_deg=runtime_config["rotation_deg"],
        noise_std=runtime_config["noise_std"],
    )
    class_names = loaders["class_names"]

    #
    # 1. Baseline Model Training
    #
    print("Training baseline ResNet...")
    resnet_config = hw.get_default_resnet_config()

    baseline_run = hw.run_training_experiment(
        model_name="resnet_baseline",
        model_builder=hw.build_resnet,
        config=resnet_config,
        loaders=loaders,
        device=device,
        output_dir=project_paths["models"],
    )

    curve_fig = hw.plot_training_curves(baseline_run["history"], "ResNet Baseline")
    curve_fig.savefig(project_paths["figures"] / "resnet_baseline_curves.png", dpi=200, bbox_inches="tight")
    plt.close(curve_fig)

    #
    # 2. Hyperparameter Search
    #
    print("Searching activation functions...")
    search_candidates = [
        {"name": "ReLU", "updates": {"activation": "relu"}},
        {"name": "LeakyReLU", "updates": {"activation": "leaky_relu"}},
        {"name": "GELU", "updates": {"activation": "gelu"}},
    ]

    search_results = hw.run_candidate_search(
        model_name="resnet_search",
        model_builder=hw.build_resnet,
        base_config=resnet_config,
        search_name="activation",
        candidates=search_candidates,
        loaders=loaders,
        device=device,
        output_dir=project_paths["models"],
    )

    search_df = search_results["results"]
    search_df.to_csv(project_paths["results"] / "resnet_search_results.csv", index=False)

    #
    # 3. Final Model Training
    #
    print("Training final ResNet...")
    best_config = search_results["best_config"]

    with open(project_paths["results"] / "resnet_best_config.json", "w", encoding="utf-8") as f:
        json.dump(best_config, f, indent=2, ensure_ascii=False, default=str)

    final_run = hw.run_training_experiment(
        model_name="resnet_final",
        model_builder=hw.build_resnet,
        config=best_config,
        loaders=loaders,
        device=device,
        output_dir=project_paths["models"],
    )

    pd.DataFrame(final_run["history"]).to_csv(project_paths["results"] / "resnet_final_history.csv", index=False)

    curve_fig = hw.plot_training_curves(final_run["history"], "ResNet Final")
    curve_fig.savefig(project_paths["figures"] / "resnet_final_curves.png", dpi=200, bbox_inches="tight")
    plt.close(curve_fig)

    #
    # 4. Final Model Evaluation
    #
    print("Evaluating final ResNet...")
    final_test_metrics = hw.evaluate_on_test(final_run["model"], loaders["test_loader"], device)

    cm_fig = hw.plot_confusion_matrix_from_preds(
        y_true=final_test_metrics["y_true"],
        y_pred=final_test_metrics["y_pred"],
        class_names=class_names,
        model_name="ResNet Final",
    )
    cm_fig.savefig(project_paths["figures"] / "resnet_final_confusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.close(cm_fig)

    preview_fig = hw.preview_predictions(final_run["model"], loaders["test_loader"], class_names, device)
    preview_fig.savefig(project_paths["figures"] / "resnet_final_predictions.png", dpi=200, bbox_inches="tight")
    plt.close(preview_fig)

    #
    # 5. Small-Sample Experiment
    #
    print("Running small-sample experiment...")
    small_sample_df, _ = hw.run_small_sample_experiment(
        model_name="resnet",
        model_builder=hw.build_resnet,
        base_config=best_config,
        runtime_config=runtime_config,
        sample_ratios=[0.3, 0.5, 1.0],
        device=device,
        output_dir=project_paths["models"],
    )

    small_sample_df.to_csv(project_paths["results"] / "resnet_small_sample_results.csv", index=False)

    small_sample_fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(small_sample_df["sample_ratio"], small_sample_df["test_accuracy"], marker="o", label="Test Accuracy",
            color="tab:blue")
    ax.plot(small_sample_df["sample_ratio"], small_sample_df["test_f1_macro"], marker="s", label="Macro F1",
            color="tab:orange")
    ax.set_xlabel("Training Data Ratio")
    ax.set_ylabel("Score")
    ax.set_title("ResNet Small-Sample Performance")
    ax.set_xticks([0.3, 0.5, 1.0])
    ax.legend()
    ax.grid(alpha=0.3)
    small_sample_fig.tight_layout()
    small_sample_fig.savefig(project_paths["figures"] / "resnet_small_sample.png", dpi=200, bbox_inches="tight")
    plt.close(small_sample_fig)

    #
    # 6. Experiment Summaries
    #
    print("Saving experiment summaries...")
    summary_payload = {
        "final_config": best_config,
        "final_summary": final_run["summary"],
        "final_test_metrics": hw.summarize_metrics(final_test_metrics),
    }
    with open(project_paths["results"] / "resnet_experiment_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, ensure_ascii=False, default=str)

    baseline_test_metrics = hw.evaluate_on_test(baseline_run["model"], loaders["test_loader"], device)
    metric_rows = [
        {"stage": "baseline", **hw.summarize_metrics(baseline_test_metrics)},
        {"stage": "final_test", **hw.summarize_metrics(final_test_metrics)},
    ]
    pd.DataFrame(metric_rows).to_csv(project_paths["results"] / "resnet_metric_summary.csv", index=False)

    print("ResNet pipeline completed.")


if __name__ == "__main__":
    main()