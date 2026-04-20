# CV HW1 协作文档

## 1. 当前完成情况

- `Step 1-4` 已完成：
  - 环境与依赖导入
  - 随机种子与设备配置
  - `EMNIST Balanced` 数据集加载
  - 固定训练/验证/测试划分
  - 数据集统计信息输出
  - 样本图像可视化
- `MLP` 部分已完成：
  - baseline 训练
  - 单因素调参搜索
  - 最终最优模型训练
  - 测试集评估
  - 训练曲线保存
  - 前 6 个样本预测图保存
  - 混淆矩阵保存
  - `30% / 50% / 100%` 小样本实验
- `CNN / ResNet / ViT` 已完成统一框架与模型脚手架，可直接由其他同学继续补充。
- `Step 6` 需要的共享评估接口也已经预留，包括：
  - `accuracy / precision / recall / F1`
  - 混淆矩阵
  - 前 6 个预测样本展示
  - 扰动鲁棒性评估

## 1.1 分工完成度判断

你负责的实现部分现在可以视为基本完成：

- `Step 1-4`：已完成
- 四类模型共享框架：已完成
- `MLP baseline`：已完成
- `MLP` 调参、最终训练、小样本实验：已完成

当前剩余工作主要是：

1. 将现有结果整理进最终报告
2. 如有需要，把 notebook 中的中文分析文字再补充得更完整一些
3. 等其他同学完成 `CNN / ResNet / ViT` 后，统一汇总四种模型的最终对比

简短结论：

- 你的代码实现任务已经基本收尾
- 后续重点是报告写作和组内合并

## 1.2 MLP Baseline 结果

Baseline 运行情况：

- 设备：`CPU`
- 可训练参数量：`573,999`
- checkpoint：`models/mlp_baseline_best.pt`
- 验证集准确率：`0.8621`
- 验证集宏平均 F1：`0.8602`
- 测试集准确率：`0.8581`
- 测试集宏平均 F1：`0.8549`

说明：

- 这个结果已经证明数据流程、训练流程、评估流程都没有问题
- 它适合作为 `MLP` 的起点
- 但它不是最终提交版结果，因为作业要求展示调参过程和最优模型

## 1.3 MLP 最终结果

顺序调参后选出的最优配置：

- 隐藏层：`512 -> 256 -> 128`
- 激活函数：`GELU`
- 归一化：`BatchNorm`
- Dropout：`0.0`
- 优化器：`SGD`
- 学习率：`0.05`
- 学习率调度器：`StepLR(step_size=3, gamma=0.5)`
- 正则化：`L1`
- `l1_lambda = 1e-6`
- 最终训练轮数设置：`15`

最终训练结果：

- 最优 epoch：`15`
- 训练时间：约 `523.7s`
- 峰值进程内存：约 `664.8 MB`
- 最终 checkpoint：`models/mlp_final_best.pt`

最终性能：

- 验证集准确率：`0.8816`
- 验证集宏平均 F1：`0.8822`
- 测试集准确率：`0.8787`
- 测试集宏平均 F1：`0.8777`

相对 baseline 的提升：

- 测试集准确率：`0.8581 -> 0.8787`，提升约 `2.05` 个百分点
- 测试集宏平均 F1：`0.8549 -> 0.8777`，提升约 `2.28` 个百分点

结论：

- 这个 MLP 结果已经明显优于 baseline
- 完全可以作为你负责部分的最终结果
- 报告里已经有足够材料说明“调参确实带来了有效提升”

## 1.4 小样本实验结果

使用最终 MLP 配置，在不同训练数据比例下的表现如下：

- `30%` 训练数据：
  - 测试集准确率：`0.8601`
  - 测试集宏平均 F1：`0.8583`
- `50%` 训练数据：
  - 测试集准确率：`0.8709`
  - 测试集宏平均 F1：`0.8699`
- `100%` 训练数据：
  - 测试集准确率：`0.8795`
  - 测试集宏平均 F1：`0.8783`

这部分已经可以直接支撑报告中的“小样本学习能力分析”。

## 2. 文件说明

- `Group8.ipynb`
  - 主 notebook
  - 已保存执行结果
  - 其他同学打开后即可直接看到你已完成部分的结果快照
- `hw1_framework.py`
  - 共享框架代码
  - 包含数据加载、模型构建、训练、评估、绘图等公共函数
- `run_mlp_pipeline.py`
  - 可复现实验脚本
  - 用于完整执行 `MLP` 的搜索、最终训练、评估、画图和小样本实验
- `requirements.txt`
  - 依赖列表

## 3. 协作边界

- 公共逻辑尽量只维护在 `hw1_framework.py`
- notebook 主要负责组织流程、展示结果、写分析文字
- 为减少冲突，不建议每个人在 notebook 里各写一套训练循环
- 各模型同学尽量只改自己对应的模型配置和模型定义

推荐分工：

- 同学 A：`MLP`
- 同学 B：`CNN`
- 同学 C：`ResNet`
- 同学 D：`ViT`

每位同学优先修改的位置：

- `Group8.ipynb` 中自己模型对应的配置区块
- `hw1_framework.py` 中自己模型对应的 builder / model class
- 自己负责的结果分析文字

## 4. 共享接口说明

### 4.1 数据加载

统一使用：

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

返回字段：

- `train_dataset`
- `valid_dataset`
- `test_dataset`
- `train_loader`
- `valid_loader`
- `test_loader`
- `class_names`

### 4.2 模型构建接口

所有模型都必须遵守同一输入输出规范：

- 输入形状：`[B, 1, 28, 28]`
- 输出形状：`[B, 47]`
- 输出必须是 `logits`
- 最后一层不要手动加 `softmax`

统一 builder：

- `hw.build_mlp(config)`
- `hw.build_cnn(config)`
- `hw.build_resnet(config)`
- `hw.build_vit(config)`

### 4.3 训练入口

所有模型统一走同一个训练包装：

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

返回内容：

- `result["model"]`
- `result["history"]`
- `result["summary"]`
- `result["config"]`

### 4.4 评估接口

- `hw.evaluate_on_test(model, loader, device)`
- `hw.preview_predictions(model, loader, class_names, device, num_samples=6)`
- `hw.plot_confusion_matrix_from_preds(y_true, y_pred, class_names, model_name)`
- `hw.evaluate_robustness(model, loader, device, perturbations)`

## 5. MLP 部分已经支持的功能

默认 `MLP` 配置：

- 3 个隐藏层：`512 -> 256 -> 128`
- 可切换激活函数
- 可切换归一化方式
- 可切换 Dropout
- 可切换优化器
- 可切换学习率调度器
- 支持 `L1 / L2 / 无正则`

推荐在报告里按下面顺序描述实验过程：

1. Baseline MLP
2. 学习率调度器搜索
3. 激活函数搜索
4. 优化器搜索
5. 归一化搜索
6. 正则化搜索
7. Dropout 搜索
8. 最优配置重训
9. `30% / 50% / 100%` 小样本比较

当前实际状态：

- 全部流程已经通过 `run_mlp_pipeline.py` 跑完
- 结果表保存在 `results/`
- 图像保存在 `figures/`
- 最终模型保存在 `models/`

重要输出文件：

- `results/mlp_best_config.json`
- `results/mlp_search_results.csv`
- `results/mlp_metric_summary.csv`
- `results/mlp_small_sample_results.csv`
- `results/mlp_experiment_summary.json`
- `figures/mlp_final_curves.png`
- `figures/mlp_final_predictions.png`
- `figures/mlp_final_confusion_matrix.png`
- `figures/mlp_small_sample.png`
- `models/mlp_final_best.pt`

## 6. 其他三位同学接手建议

### 6.1 CNN

从这里开始：

- `hw.get_default_cnn_config()`
- `hw.build_cnn(config)`

建议重点：

- 卷积层深度
- 通道数
- kernel size / pooling / dropout 组合
- 正则化策略

### 6.2 ResNet

从这里开始：

- `hw.get_default_resnet_config()`
- `hw.build_resnet(config)`
- `ResidualBlock`

建议重点：

- 残差块层数
- 是否比普通 CNN 更稳定
- 通道宽度与训练策略

### 6.3 ViT

从这里开始：

- `hw.get_default_vit_config()`
- `hw.build_vit(config)`

建议重点：

- patch size
- embedding 维度
- head 数量
- encoder 深度
- dropout 与学习率

## 7. Notebook 使用说明

当前 notebook 已经保存了执行结果，推荐按下面方式使用：

1. 直接打开 `Group8.ipynb`
2. 查看上半部分的：
   - 数据划分统计
   - 样本图像
   - MLP 结构信息
3. 查看 `已完成结果快照` 部分：
   - 已完成步骤状态
   - baseline 与最终结果对比
   - 最终配置
   - 调参结果表
   - 小样本实验结果
   - 曲线图 / 预测图 / 混淆矩阵 / 小样本图

如果只是查看结果，不需要重新训练。

如果确实需要重新跑：

- 将 notebook 中对应 `RUN_*` 开关改成 `True`
- 或直接执行 `run_mlp_pipeline.py`

## 8. 输出规范

框架会自动创建这些目录：

- `data/`
- `figures/`
- `models/`
- `results/`

建议统一保存规则：

- checkpoint：`models/<model_name>_best.pt`
- 图像：`figures/<model_name>_<topic>.png`
- 表格：`results/<model_name>_<topic>.csv`

## 9. 报告写作对应关系

可以直接按下面映射去写报告：

- 数据集介绍与样本展示：`Step 1-4`
- 模型结构说明：配置 + 模型定义
- 调参过程：搜索结果表
- 训练过程：loss / accuracy 曲线
- 最终性能：测试指标表
- 定性分析：前 6 个预测样本
- 类别分析：混淆矩阵
- 小样本分析：`30% / 50% / 100%` 实验结果

## 10. 注意事项

- 四个模型必须共用同一个固定 train/valid/test 划分
- 不要给不同模型随意换随机划分，否则结果不可公平比较
- 使用 `CrossEntropyLoss` 时，模型最后一层不要手动加 `softmax`
- 如果某位同学改了增强策略，报告里必须明确说明
- 所有模型要使用同样的测试集和同样的指标
- 合并前要确认四个最佳模型都能被共享评估单元正确加载
- `data/EMNIST/raw` 很大，且已被 `.gitignore` 排除，不要手动提交
- `models/*.pt` 属于本地产物，默认不提交版本库

## 11. 最终合并检查清单

提交前建议逐项确认：

1. 四个模型都能用共享训练器正常训练
2. notebook 可以从上到下顺序运行
3. 报告中的图和表都能在 notebook 中找到对应来源
4. 每个模型至少都有：
   - 最优配置
   - 训练曲线
   - 测试指标
   - 混淆矩阵
   - 前 6 个预测样本
   - 鲁棒性结果
5. 四个模型的小样本实验表格格式保持一致
