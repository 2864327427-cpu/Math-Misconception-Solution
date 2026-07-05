# MAP2025 学生数学误解识别｜解决方案报告



## 1) 比赛理解

### 1.1 背景与目标

面向学生开放性解释文本，预测其是否存在数学误解及其具体类型，并以 **MAP@3** 评估。

学生的书面解释能揭示其推理与误解模式，但人工标注成本高、一致性差，且误解随时间演化。本竞赛希望参赛者构建 **NLP 模型**，在不同题目间 **泛化识别** 常见的数学误解，从而帮助教师更快定位并纠正学生错误思维。

早期直接用通用预训练语言模型的尝试效果有限（数学内容复杂、表述多样）。因此，竞赛提供了题目文本、学生选择及解释等多源信息，鼓励参赛者设计更高效一致的自动标注方法，提升误解标注质量与可扩展性。

---

### 1.2 竞赛数据

* 学生在 Eedi 平台完成诊断题（四选一：1 正确 + 3 干扰项），并要求给出 **书面解释** 为什么会选择该项。本任务用这些解释来识别潜在误解。
* 目标包含三步：

  1. 判断选择是否正确（`True/False` 前缀，例如 `True_*`）。
  2. 判断解释是否包含误解（`Correct/Misconception/Neither`，例如 `True_Misconception`）。
  3. 若有，则识别具体 **Misconception** 标签。

**文件**

* **[train/test].csv**

  * `QuestionId`：题目唯一 ID
  * `QuestionText`：题目文本
  * `MC_Answer`：学生所选选项
  * `StudentExplanation`：学生解释
  * `Category`（仅训练）：如 `True_Misconception`
  * `Misconception`（仅训练）：具体误解名；若无则 `NA`
* **sample_submission.csv**
* 每行最多 3 个预测，格式：`Category:Misconception`，以空格分隔

**简单 EDA**

* `Category` 分布：

  ```
  Category
  True_Correct           14802
  False_Misconception     9457
  False_Neither           6542
  True_Neither            5265
  True_Misconception       403
  False_Correct            227
  Name: count, dtype: int64
  ```
  
* `Misconception` 分布（Top 若干）：

  ```
  Misconception
  Incomplete                                1454
  Additive                                   929
  Duplication                                704
  Subtraction                                620
  Positive                                   566
  Wrong_term                                 558
  Irrelevant                                 497
  Wrong_fraction                             418
  Inversion                                  414
  Mult                                       353
  Denominator-only_change                    336
  Whole_numbers_larger                       329
  Adding_across                              307
  WNB                                        299
  Tacking                                    290
  Unknowable                                 282
  Wrong_Fraction                             273
  SwapDividend                               206
  Scale                                      179
  Not_variable                               154
  Firstterm                                  107
  Adding_terms                                97
  Multiplying_by_4                            96
  FlipChange                                  78
  Division                                    63
  Definition                                  54
  Interior                                    50
  Longer_is_bigger                            24
  Ignores_zeroes                              23
  Base_rate                                   23
  Shorter_is_bigger                           23
  Inverse_operation                           21
  Certainty                                   18
  Incorrect_equivalent_fraction_addition       9
  Wrong_Operation                              6
  Name: count, dtype: int64
  ```

---

### 1.3 评估指标

以 **平均精度@3（MAP@3）** 评估：

$$
\mathrm{MAP@3}=\frac{1}{U}\sum_{u=1}^{U}\sum_{k=1}^{\min(n,3)} P(k)\times \mathrm{rel}(k)
$$

* $U$：样本数
* $P(k)$：截止第 $k$ 个预测的精度
* $n$：该样本提交的预测数量
* $\mathrm{rel}(k)$：指示函数，若第 $k$ 位预测为正确标签则为 1，否则为 0
* 同一观察的正确标签一旦命中，后续重复命中不再记分（示例：`[A,B,C]`、`[A,A,A]`、`[A,B,A]` 均得分 1.0）。
* 每个样本只有 **一个** 正确标签。

---

### 1.4 领域知识入门

* **诊断题（DQ）**：用精心设计的错误选项暴露学生常见误解。
* **Category**：由两部分组成，前缀 `True/False`（所选是否正确），后缀 `Correct/Misconception/Neither`（解释层面）。
* **Misconception**：细粒度误解标签，示例：`Additive`（把乘法当加法）、`Whole_numbers_larger`（“整数更大”迁移到小数比较）等。
* **约束一致性**：某些推断（如所选项是否正确）可由题目统计提供强约束。
* **MAP@K**：排名相关评估，关注前 3 位的命中质量。

---



## 2) 解决方案解析

### 2.1 方案流程概览

* **数据预处理**

  * 统一目标为 **单一分类空间**：`Category:Misconception`（`NA` 用于无误解）。
  * 通过训练集中 `True_*` 样本，统计每个 `QuestionId` 最常见的“正确 `MC_Answer`”，并标记测试集中每行 `is_correct∈{0,1}`，用于输入提示（`Is Correct Answer: Yes/No`）及后续 **family 过滤**。
  * 文本拼接为输入：题干 + 选项 + 正误提示 + 学生解释；`max_length=256`。

* **模型设计与训练**

  * 基座：**Qwen3** 4B/8B/14B（HF Transformers `AutoModelForSequenceClassification`）。
  * 训练：**LoRA** 微调（部分实验含 4-bit 量化），`Trainer` 全量训练（无 CV），`cosine` 调度 + `warmup_ratio=0.1`，`bf16/float16`。

* **推理与后处理**

  * 单模输出 **Top-25** 类别概率表（`top_classes + prob_0...prob_24`）。
  * 构建 **family 前缀** 映射（`True_` / `False_`），在融合时仅保留一致前缀的类别。
  * **多模型集成**：综合“加权总概率 + 一致性占比 + 最大置信”三信号，并做前缀过滤与 **缺位补齐**（如 `Neither:NA`），生成每行最多 3 个提交标签。

---

### 2.2 关键技术点

#### 模型选择与原因

* **LLM 作为分类器（Qwen3 4B/8B/14B + LoRA）**

  * 解释文本含自然语言与数学关系，LLM 对语言理解、语境对齐更强。
  * 使用分类头将任务 **化为单一多类分类**，覆盖 `Category×Misconception` 的笛卡尔空间（含 `NA`），直接对应评测的“单真值”。

#### 特征工程 / 数据增强

* **输入模版加入强约束提示**：

  * `Is Correct Answer: Yes/No` 源于题目级统计，有助于模型更好地区分 `True_*` 与 `False_*`。
* **目标拼接（`Category:Misconception`）**：

  * 将二级决策变为单头分类，避免级联误差与不一致。
* **Top-25 候选保留**：

  * 为后续集成保留更多可用信息，降低截断损失。

#### 训练策略

* **损失**：标准多类交叉熵（分类头）。
* **优化/调度**：`Trainer` 默认优化器 + `cosine` 学习率调度，`warmup_ratio=0.1`。
* **精度/显存**：`bf16/float16`，可选 **bnb 4-bit**（`nf4`，double quant）。
* **LoRA**：针对 `q_proj/v_proj/o_proj/gate_proj/up_proj/down_proj`，示例超参：

  * 4B：`r=512, α=32, dropout=0.05`（不量化）
  * 8B：`r=64,  α=32, dropout=0.05`（4-bit）
  * 14B：`r=16,  α=32, dropout=0.05`（4-bit）
* **批次与轮数**（示例）：`epochs=2~3`，`per_device_train_batch_size=8~16`，`grad_accum=1~2`。
* **其他**：梯度检查点、动态 padding。

#### 推理与集成

* **Family 前缀过滤（创新）**
  * 机制：由训练集中每题最常见的“正确 `MC_Answer`”确定测试行的 `True_/False_` 前缀，仅在该前缀下排序候选。
  * 作用：把“是否答对”的 **结构性先验** 显式注入，显著减少错误前缀下的误报空间。
* **一致性加权集成（创新）**
  * 机制：综合 **加权总概率（0.34） + 模型一致性占比（0.33） + 最大加权置信（0.33）**，在同一候选空间聚合 4B/8B/14B 的信息。
  * 作用：既考虑整体支持度，又关注“多模型都看到”的稳定信号，并保留“尖峰置信”以抓住强预测。
* **缺位补齐策略**
* 若候选不足 3 个，优先补 `Neither:NA`，在 `True_` 情况下再补 `Correct:NA`，确保提交合法且稳健。

---

### 2.3 代码解析

#### A) 训练（`train.py`）

* **数据与标签**

  * 读 `train.csv`，`Misconception` 缺失填 `NA`；`target = Category:Misconception`；`LabelEncoder` → `label`。
  * 统计 **每题最常见正确选项**，生成 `is_correct∈{0,1}`。

* **输入构造**

  ```python
  def format_input_v2(row):
      x = "Yes" if row['is_correct'] else "No"
      return (
          f"Question: {row['QuestionText']}\n"
          f"Answer: {row['MC_Answer']}\n"
          f"Is Correct Answer: {x}\n"
          f"Student Explanation: {row['StudentExplanation']}"
      )
  ```

  * 输出字段：`text`（后续 `tokenizer(..., truncation=True, max_length=256)`）。

* **模型与训练**

  * `AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=K, ...)`
  * 可选 `BitsAndBytesConfig(load_in_4bit=True, quant_type="nf4", compute_dtype=bfloat16)`
  * **LoRA** 到投影/门控/上下投层；`prepare_model_for_kbit_training`；`bf16`。
  * `TrainingArguments`：`cosine`、`warmup_ratio=0.1`、`gradient_checkpointing=True`、`save_strategy="no"`。
  * `Trainer(...).train()`；保存到 `./output/`。

#### B) 离线推理（`inference.py`）

* **标签空间复原**：用训练集重新 `LabelEncoder.fit`，保证类别顺序一致。
* **同样构建 `is_correct` 与输入文本**（`format_input` 与训练一致）。
* **模型载入**：基座（4/8/14B）+ Peft LoRA；4-bit 量化、`fp16` 推理。
* **预测产出**：

  * 对每行输出 logits → softmax 概率；**保留 Top-25**（`top_classes` + `prob_0..prob_24`）到 CSV，供集成使用。

#### C) 集成与提交（`inference_script.ipynb`）

* **Family 映射**

  * 用训练集构建每题“最常见正确选项”；在测试集上得到 `fam_map[row_id]∈{"True_","False_"}`。

* **集成核心**

  ```python
  final_scores[class_name] = (
      class_total_prob[class_name] * 0.34 +
      (class_votes[class_name] / n_models) * 0.33 +
      class_max_prob[class_name] * 0.33
  )
  # 仅保留与 pref ('True_'/'False_') 一致的类别，再排序取前K
  ```

  * **过滤**：只保留匹配 `pref` 的类别。
  * **补齐**：不足 3 个时，优先添加 `pref+"Neither:NA"`，在 `True_` 场景补一个 `pref+"Correct:NA"`。
  * 生成 `submission.csv`（两列：`row_id, Category:Misconception`）。

---

### 2.4 结果与总结

**Private LB**

| 方法 | 模型          | 量化/LoRA    | 训练轮次 | 提交                   | Private LB |
| ---- | ------------- | ------------ | -------- | ---------------------- | ---------- |
| 单模 | Qwen3-14B     | 4-bit + LoRA | 3        | 单模                   | **0.945**  |
| 单模 | Qwen3-8B      | 4-bit + LoRA | 2        | 单模                   | 0.944      |
| 单模 | Qwen3-4B      | FP16 + LoRA  | 3        | 单模                   | 0.944      |
| 集成 | 4B + 8B + 14B |              |          | 三模融合（含前缀过滤） | **0.946**  |

---



## 3) 简历项目模板

**Kaggle｜MAP — Charting Student Math Misunderstandings（2025.07–2025.10）— 最终 LB 0.946（三模集成）**

* **项目背景/挑战**：开放式解释文本中定位数学误解；标签长尾、类别不均衡；需在跨题目场景泛化；评测为 **MAP@3**（排名相关）。
* **方案亮点**：
  * 用 **LLM 分类头 + LoRA** 端到端建模 `Category×Misconception` 单空间，避免级联误差；
  * 设计 **结构先验**：由题目级统计构造 `True_/False_` **family 前缀**并用于提示与后处理过滤；
  * 多模型 **一致性加权集成**（总概率 + 一致性占比 + 最大置信），并做合法性补齐；
  * 显存友好：**4-bit 量化 + bf16/fp16**，在 4B/8B/14B 上复现实验。
* **量化结果**：单模 LB 0.944–0.945；三模集成 **0.946**（+0.001 相比最佳单模）。
* **技术栈**：Python、PyTorch、Transformers、PEFT/LoRA、BitsAndBytes、HuggingFace Datasets、Pandas、Kaggle Notebook。
