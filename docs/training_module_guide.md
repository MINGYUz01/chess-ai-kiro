# 训练模块使用指南

本文档提供了棋盘识别系统训练模块的使用指南，包括模型训练、超参数优化、模型评估和导出等功能。

## 目录

1. [基本训练](#基本训练)
2. [超参数优化](#超参数优化)
3. [模型评估](#模型评估)
4. [模型导出](#模型导出)
5. [命令行工具](#命令行工具)
6. [常见问题](#常见问题)

## 基本训练

### 准备数据

在开始训练之前，您需要准备好数据集并创建数据配置文件。数据配置文件应包含训练集、验证集和测试集的路径，以及类别信息。

您可以使用 `DataConfigGenerator` 类来生成数据配置文件：

```python
from chess_ai_project.src.chess_board_recognition.training.config_validator import DataConfigGenerator

# 初始化数据配置生成器
generator = DataConfigGenerator()

# 生成数据配置文件
generator.generate_data_yaml(
    train_path="./data/train",
    val_path="./data/val",
    test_path="./data/test",
    class_names=["board", "grid_lines", "red_king", "red_advisor", "red_bishop", "red_knight", "red_rook", "red_cannon", "red_pawn", "black_king", "black_advisor", "black_bishop", "black_knight", "black_rook", "black_cannon", "black_pawn", "selected_piece"],
    output_path="./data/chess_board.yaml"
)
```

### 训练模型

使用 `YOLO11Trainer` 类来训练模型：

```python
from chess_ai_project.src.chess_board_recognition.training.trainer import YOLO11Trainer

# 初始化训练器
trainer = YOLO11Trainer()

# 加载预训练模型（可选）
trainer.load_model("yolo11n.pt")

# 训练模型
results = trainer.train(
    data_yaml_path="./data/chess_board.yaml",
    epochs=100,
    batch_size=16,
    image_size=640,
    device="auto"
)

# 验证模型
val_results = trainer.validate(data_yaml_path="./data/chess_board.yaml")
```

### 监控训练进度

使用 `TrainingMonitor` 类来监控训练进度：

```python
from chess_ai_project.src.chess_board_recognition.training.monitor import TrainingMonitor

# 初始化训练监控器
monitor = TrainingMonitor(metrics_file="./logs/training_metrics.json")

# 开始监控
monitor.start_monitoring(trainer)

# 训练模型
results = trainer.train(data_yaml_path="./data/chess_board.yaml")

# 停止监控
monitor.stop_monitoring()

# 绘制训练指标图表
monitor.plot_metrics(save_path="./logs/metrics_plot.png")

# 生成训练报告
report = monitor.generate_training_report(save_path="./logs/training_report.json")
```

## 超参数优化

使用 `HyperparameterOptimizer` 类来优化模型超参数：

```python
from chess_ai_project.src.chess_board_recognition.training.hyperparameter_optimizer import HyperparameterOptimizer

# 初始化训练器
trainer = YOLO11Trainer()

# 初始化超参数优化器
optimizer = HyperparameterOptimizer(
    trainer=trainer,
    data_yaml_path="./data/chess_board.yaml",
    output_dir="./hyperparameter_search"
)

# 随机搜索
param_space = {
    'batch': (8, 32),
    'lr0': (0.0001, 0.01),
    'weight_decay': (0.0001, 0.01),
    'momentum': (0.8, 0.99),
}
best_params = optimizer.random_search(param_space, num_trials=10, epochs=5)

# 网格搜索
param_grid = {
    'batch': [8, 16, 32],
    'lr0': [0.0005, 0.001, 0.01],
}
best_params = optimizer.grid_search(param_grid, epochs=5)

# 贝叶斯优化（需要安装scikit-optimize库）
param_space = {
    'batch': (8, 32),
    'lr0': (0.0001, 0.01),
}
best_params = optimizer.bayesian_optimization(param_space, num_trials=10, epochs=5)

# 绘制搜索历史
optimizer.plot_search_history(save_path="./hyperparameter_search/search_history.png")

# 生成优化报告
optimizer.generate_optimization_report(save_path="./hyperparameter_search/optimization_report.json")
```

## 模型评估

使用 `ModelEvaluator` 类来评估模型性能：

```python
from chess_ai_project.src.chess_board_recognition.training.evaluator import ModelEvaluator

# 初始化模型评估器
evaluator = ModelEvaluator(
    model_path="./models/best.pt",
    output_dir="./evaluation_results"
)

# 在数据集上评估模型
results = evaluator.evaluate_on_dataset(
    data_yaml_path="./data/chess_board.yaml",
    batch_size=16,
    image_size=640,
    device="auto"
)

# 在图像目录上评估模型
results = evaluator.evaluate_on_images(
    image_dir="./data/test_images",
    ground_truth_dir="./data/test_labels"
)

# 绘制评估结果
evaluator.plot_evaluation_results(save_path="./evaluation_results/evaluation_plot.png")

# 生成评估报告
evaluator.generate_evaluation_report(save_path="./evaluation_results/evaluation_report.json")

# 比较多个模型
comparison = evaluator.compare_models(
    model_paths=["./models/model1.pt", "./models/model2.pt"],
    data_yaml_path="./data/chess_board.yaml"
)

# 绘制模型比较图表
evaluator.plot_model_comparison(
    comparison_results=comparison,
    save_path="./evaluation_results/model_comparison.png"
)
```

## 模型导出

使用 `ModelExporter` 类来导出模型为不同格式：

```python
from chess_ai_project.src.chess_board_recognition.training.model_exporter import ModelExporter

# 初始化模型导出器
exporter = ModelExporter(
    model_path="./models/best.pt",
    output_dir="./exported_models"
)

# 导出为ONNX格式
onnx_path = exporter.export_model(
    format="onnx",
    image_size=640,
    batch_size=1,
    dynamic=True,
    simplify=True
)

# 导出为多种格式
results = exporter.export_to_multiple_formats(
    formats=["onnx", "torchscript", "openvino"],
    image_size=640
)

# 优化ONNX模型
optimized_path = exporter.optimize_model(
    model_path=onnx_path,
    format="onnx",
    quantize=True
)

# 比较模型大小
size_comparison = exporter.compare_model_sizes(results)
```

## 命令行工具

我们提供了两个命令行工具来简化训练和优化过程：

### 训练脚本

```bash
python -m chess_ai_project.src.chess_board_recognition.training.train_script \
    --config ./configs/training_config.yaml \
    --data ./data/chess_board.yaml \
    --model yolo11n.pt \
    --epochs 100 \
    --batch-size 16 \
    --img-size 640 \
    --device auto
```

### 优化脚本

```bash
python -m chess_ai_project.src.chess_board_recognition.training.optimization_script \
    --config ./configs/training_config.yaml \
    --data ./data/chess_board.yaml \
    --model ./models/best.pt \
    --mode all \
    --output-dir ./results \
    --search-method random \
    --trials 10 \
    --epochs 5 \
    --export-formats onnx,torchscript
```

## 常见问题

### 1. 如何选择合适的超参数搜索方法？

- **网格搜索**：适用于参数空间较小，且您对参数范围有较好了解的情况。
- **随机搜索**：适用于参数空间较大，且您不确定参数范围的情况。
- **贝叶斯优化**：适用于计算资源有限，需要高效搜索的情况。

### 2. 如何处理训练过程中的过拟合？

- 增加数据增强
- 使用早停策略（设置patience参数）
- 调整正则化参数（weight_decay）
- 减小模型复杂度

### 3. 如何选择合适的导出格式？

- **ONNX**：跨平台通用格式，适用于多种推理框架
- **TorchScript**：适用于PyTorch环境
- **OpenVINO**：适用于Intel硬件加速
- **TensorRT**：适用于NVIDIA GPU加速

### 4. 如何提高训练速度？

- 使用GPU训练
- 增大batch_size（如果内存允许）
- 使用混合精度训练（设置amp=True）
- 使用数据缓存（设置cache=True）

### 5. 如何提高模型准确率？

- 增加训练数据量和多样性
- 调整学习率和训练轮次
- 使用超参数优化找到最佳参数
- 尝试不同的模型架构和预训练权重