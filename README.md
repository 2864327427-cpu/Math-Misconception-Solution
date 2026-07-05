# 🏆 Kaggle | MAP 学生数学误解识别

本仓库包含 **Kaggle MAP@3 学生数学误解识别竞赛** 的核心训练与推理流程。我们使用基于 LLM 的端到端多分类方案，取得了 Private LB **0.946** 的成绩。

## 🚀 核心贡献

作为核心算法开发者，我的主要贡献集中在 **工程优化、LLM 微调与集成策略**：

1. **端到端统一空间建模**：将传统的级联分类替换为统一的 `Category × Misconception` 映射空间，使用 **Qwen3 (4B/8B/14B) + LoRA**。
2. **显存高效微调**：利用 `BitsAndBytes` 的 **4-bit 量化 (nf4)**，结合 `bfloat16` 混合精度与梯度检查点，在有限显存硬件上成功训练 14B 模型。
3. **结构先验（提示工程）**：基于正确答案统计分布设计 `True_/False_family` 前缀逻辑，显著提升了 LLM 的零样本泛化能力，并在后处理中过滤非法预测。
4. **一致性加权集成**：实现鲁棒的多模型集成算法，组合总概率 (0.34) + 一致性比率 (0.33) + 最大置信度 (0.33)，将 LB 分数从 0.945 提升至 0.946。

## 📂 仓库结构

```text
├── docs/
│   └── solution.md          # 完整方案报告（中文）
├── src/
│   ├── train.py             # Qwen3 LoRA 微调脚本
│   ├── inference.py         # 单模型推理（Top-25 logits）
│   ├── inference_script.ipynb # 集成笔记本（家族过滤 + 加权集成 + 提交）
│   └── train_run.sh           # 训练启动脚本
├── requirements.txt         # 依赖
├── .gitignore
└── README.md
```

## ⚙️ 运行方式

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 数据准备

从 [Kaggle MAP](https://kaggle.com/competitions/map-charting-student-math-misunderstandings) 下载竞赛数据，将 CSV 文件放入 `data/` 目录。

### 3. 模型训练（以 Qwen3-8B 为例）

```bash
python src/train.py --model_name "Qwen/Qwen3-8B" --quantization "4bit" --batch_size 8
```

### 4. 单模型推理

```bash
python src/inference.py --model_path ./output --test_data ./data/test.csv
```

### 5. 集成与提交

在 Jupyter/Kaggle Notebook 中运行 `src/inference_script.ipynb`，集成 4B+8B+14B 预测结果并生成提交文件。

或使用命令行集成：
```bash
python src/inference_ensemble.py --weights "4b,8b,14b"
```

## 🛠 技术栈

Python | PyTorch | Transformers | PEFT/LoRA | BitsAndBytes | Pandas
