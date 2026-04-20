# CV HW1 Collaboration Handoff

## 1. What is already implemented

- `Step 1-4` notebook scaffold is ready:
  - imports and environment setup
  - seed/device configuration
  - EMNIST Balanced loading
  - dataset statistics
  - sample visualization
- `MLP` has a reusable implementation and the shared training pipeline needed for:
  - baseline training
  - single-factor hyperparameter exploration
  - best-model retraining
  - training curve plotting
  - test evaluation
  - small-sample experiments using 30% / 50% / 100% of the training split
- `CNN / ResNet / ViT` all have working scaffold classes with a unified interface.
- Shared utilities for later `Step 6` are included:
  - confusion matrix
  - precision / recall / F1
  - first-6 prediction preview
  - perturbation-based robustness evaluation

## 1.1 Current progress status

Current state of the work is:

- `Step 1-4`: functionally complete.
- Shared framework for all four models: complete.
- `MLP` baseline run: complete.
- `MLP` tuning, final training, and small-sample experiment: complete.

What is already done for the MLP owner:

- dependencies installed locally
- baseline notebook execution completed once on CPU
- baseline checkpoint saved to `models/mlp_baseline_best.pt`
- baseline data pipeline, training loop, evaluation loop, and plotting hooks verified
- factor-by-factor search completed
- final best MLP trained and evaluated
- required `30% / 50% / 100%` small-sample experiment completed
- final figures, metric tables, config files, and checkpoints saved under `figures/`, `results/`, and `models/`

What still remains for submission:

1. write the final report section that explains the chosen MLP design
2. optionally mirror the saved result tables inside the notebook narrative if the team wants a more presentation-ready notebook

Short judgment:

- your assigned implementation work is essentially complete
- the remaining work is mainly documentation and report writing

## 1.2 Current MLP baseline result

Baseline run summary:

- device: `CPU`
- trainable parameters: `573,999`
- checkpoint: `models/mlp_baseline_best.pt`
- validation accuracy: `0.8621`
- validation macro F1: `0.8602`
- test accuracy: `0.8581`
- test macro F1: `0.8549`

Interpretation:

- this is a solid baseline for an MLP on `EMNIST Balanced`
- the model is learning correctly and the pipeline is usable
- however, this should be treated as the starting point, not the final submitted MLP result

Is it good enough?

- yes, as a baseline
- no, as the final answer for your assigned MLP work

Reason:

- the homework explicitly asks for exploration of schedulers, activations, optimizers, normalization, regularization, and dropout
- the final model should be the best configuration found after that exploration
- for the report, it is much stronger if the final MLP is better than the baseline and the improvement can be explained clearly

Practical target:

- if tuning moves test accuracy from about `85.8%` to the high `86%` range or better, that is already a meaningful improvement for the report
- even if the gain is small, the important part is to show a clear search process and explain why the final configuration was chosen

## 1.3 Final MLP result

Best config selected after sequential search:

- hidden layers: `512 -> 256 -> 128`
- activation: `GELU`
- normalization: `BatchNorm`
- dropout: `0.0`
- optimizer: `SGD`
- learning rate: `0.05`
- scheduler: `StepLR(step_size=3, gamma=0.5)`
- regularization: `L1`, `l1_lambda = 1e-6`
- final training epochs setting: `15`

Final run summary:

- best epoch: `15`
- training time: about `523.7s`
- peak process memory: about `664.8 MB`
- final checkpoint: `models/mlp_final_best.pt`

Final performance:

- validation accuracy: `0.8816`
- validation macro F1: `0.8822`
- test accuracy: `0.8787`
- test macro F1: `0.8777`

Improvement over the baseline:

- test accuracy: `0.8581 -> 0.8787`, about `+2.05` percentage points
- test macro F1: `0.8549 -> 0.8777`, about `+2.28` percentage points

Interpretation:

- this is a clearly better result than the baseline
- the tuned MLP is strong enough to serve as your final MLP contribution
- for a non-convolutional model on `EMNIST Balanced`, this is a respectable result and gives you enough material to discuss why the chosen settings worked better

## 2. File map

- `Group8.ipynb`
  - main collaboration notebook
  - intended as the final submission entry point
- `hw1_framework.py`
  - reusable framework code
  - teammates should prefer editing here instead of duplicating logic inside the notebook
- `run_mlp_pipeline.py`
  - reproducible script used to finish the full MLP workflow
  - runs search, final training, evaluation, and the small-sample experiment
- `requirements.txt`
  - runtime dependencies to install before running

## 3. Ownership boundary for collaboration

- Shared code should stay centralized in `hw1_framework.py`.
- The notebook should mostly orchestrate experiments and show results.
- To reduce merge conflicts:
  - do not duplicate the training loop in multiple notebook cells
  - do not create separate data split logic per model
  - do not change the metric definitions independently

Recommended ownership split:

- Student A: `MLP`
- Student B: `CNN`
- Student C: `ResNet`
- Student D: `ViT`

Each teammate should mainly touch:

- their model config cell in `Group8.ipynb`
- their model builder or model class in `hw1_framework.py`
- their own markdown analysis cells in the notebook/report

## 4. Shared API summary

### 4.1 Data loading

Use:

```python
runtime_config = hw.get_default_runtime_config(PROJECT_DIR)
loaders = hw.load_emnist_balanced(
    data_dir=runtime_config["data_dir"],
    batch_size=runtime_config["batch_size"],
    valid_ratio=runtime_config["valid_ratio"],
    num_workers=runtime_config["num_workers"],
    subset_ratio=1.0,
    augment=runtime_config["augment"],
    rotation_deg=runtime_config["rotation_deg"],
    noise_std=runtime_config["noise_std"],
    blur=runtime_config["blur"],
    seed=runtime_config["seed"],
)
```

Returned keys:

- `train_dataset`
- `valid_dataset`
- `test_dataset`
- `train_loader`
- `valid_loader`
- `test_loader`
- `class_names`

### 4.2 Model builders

All models must follow the same input/output convention:

- input shape: `[B, 1, 28, 28]`
- output shape: `[B, 47]`
- output must be raw logits
- do not add `softmax` inside the model

Builder functions:

- `hw.build_mlp(config)`
- `hw.build_cnn(config)`
- `hw.build_resnet(config)`
- `hw.build_vit(config)`

### 4.3 Training entry point

Use the same training wrapper for every model:

```python
result = hw.run_training_experiment(
    model_name="mlp_baseline",
    model_builder=hw.build_mlp,
    config=mlp_config,
    loaders=loaders,
    device=device,
    output_dir=project_paths["models"],
)
```

Returned content:

- `result["model"]`
- `result["history"]`
- `result["summary"]`
- `result["config"]`

### 4.4 Evaluation entry points

- `hw.evaluate_on_test(model, loader, device)`
- `hw.preview_predictions(model, loader, class_names, device, num_samples=6)`
- `hw.plot_confusion_matrix_from_preds(y_true, y_pred, class_names, model_name)`
- `hw.evaluate_robustness(model, loader, device, perturbations)`

## 5. What the MLP part already supports

Default MLP config:

- 3 hidden layers: `512 -> 256 -> 128`
- configurable activation
- configurable normalization
- configurable dropout
- configurable optimizer
- configurable scheduler
- configurable L1 / L2 regularization

Suggested exploration order for the report:

1. Baseline MLP
2. Learning-rate scheduler search
3. Activation search
4. Optimizer search
5. Normalization search
6. Regularization search
7. Dropout search
8. Best-config retraining
9. 30% / 50% / 100% small-sample comparison

Recommended execution order inside the notebook:

1. run `RUN_MLP_BASELINE = True` and verify the baseline checkpoint exists
2. switch `RUN_MLP_SEARCH = True` to perform the single-factor searches
3. keep the selected `tuned_mlp_config`
4. switch `RUN_MLP_FINAL = True` to train the best MLP model
5. switch `RUN_MLP_SMALL_SAMPLE = True` to finish the required small-sample analysis

Current status of the notebook flags:

- `RUN_MLP_BASELINE`: enabled and already executed once
- `RUN_MLP_SEARCH`: still disabled
- `RUN_MLP_FINAL`: still disabled
- `RUN_MLP_SMALL_SAMPLE`: still disabled

Current status of the MLP workflow overall:

- the full MLP workflow has already been completed through `run_mlp_pipeline.py`
- final tables are saved in `results/`
- final figures are saved in `figures/`
- final model checkpoints are saved in `models/`

Important output files:

- `results/mlp_best_config.json`
- `results/mlp_search_results.csv`
- `results/mlp_metric_summary.csv`
- `results/mlp_small_sample_results.csv`
- `results/mlp_experiment_summary.json`
- `figures/mlp_final_curves.png`
- `figures/mlp_small_sample.png`
- `models/mlp_final_best.pt`

## 6. What each teammate should do next

### 6.1 CNN teammate

Start from:

- `hw.get_default_cnn_config()`
- `hw.build_cnn(config)`

Likely edits:

- adjust convolution block depth
- tune channel counts
- try kernel size / pooling / dropout variants
- add stronger but still safe regularization if needed

### 6.2 ResNet teammate

Start from:

- `hw.get_default_resnet_config()`
- `hw.build_resnet(config)`
- `ResidualBlock`

Likely edits:

- deepen the residual stack
- compare residual vs non-residual CNN behavior
- tune base channel width and training schedule

### 6.3 ViT teammate

Start from:

- `hw.get_default_vit_config()`
- `hw.build_vit(config)`

Likely edits:

- patch size
- embedding dimension
- number of heads
- encoder depth
- dropout and learning-rate tuning

## 7. Notebook execution order

Run the notebook top to bottom in this order:

1. Environment and imports
2. Shared config setup
3. Data loading and visualization
4. MLP baseline and search
5. MLP best-model training
6. MLP small-sample experiment
7. CNN / ResNet / ViT teammate sections
8. Shared Step 6 evaluation

## 8. Output convention

The framework creates these folders automatically:

- `data/`
- `figures/`
- `models/`
- `results/`

Recommended saving practice:

- model checkpoints: `models/<model_name>_best.pt`
- figures: `figures/<model_name>_<topic>.png`
- tables: `results/<model_name>_<topic>.csv`

## 9. Report-writing mapping

Use the notebook artifacts to populate the report:

- dataset intro and samples: `Step 1-4` cells
- model structure description: model config + model class
- tuning logic: search plan cells
- training/validation curves: `plot_training_curves`
- final metrics: `evaluate_on_test`
- qualitative examples: `preview_predictions`
- confusion matrix: `plot_confusion_matrix_from_preds`
- robustness analysis: `evaluate_robustness`

## 10. Important cautions

- Keep the train/valid/test split fixed across all models.
- Do not compare models trained with different random splits.
- Do not place `softmax` in the final layer if you use `CrossEntropyLoss`.
- If augmentation settings change, note that clearly in the report.
- For fairness, use the same evaluation metrics and the same test set for all models.
- Before merging team contributions, make sure all four best-model checkpoints can be loaded by the shared evaluation cells.
- The EMNIST download is large and torchvision stores multiple raw subsets under `data/EMNIST/raw`; keep that folder ignored by git.
- The saved `mlp_baseline_best.pt` checkpoint is a local artifact for reruns and is intentionally excluded from version control.

## 11. Suggested merge checklist

Before final submission, verify:

1. All four models can train with the shared wrapper.
2. The notebook can be run from top to bottom without manual edits.
3. All figures used in the report are generated from the notebook.
4. Each model has:
   - best config
   - training curves
   - test metrics
   - confusion matrix
   - first 6 predictions
   - robustness results
5. Small-sample results are presented in the same table format for all models.
