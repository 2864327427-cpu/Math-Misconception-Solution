import argparse
import os
import pandas as pd
import numpy as np

# ----------------------------
# Argparse
# ----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--cuda", type=str, default="0,1",
                    help='CUDA devices, e.g. "0" or "0,1" or "0,1,2,3"')

# Model name
parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B", help='Model name')

# Enable / disable bnb_config (4-bit quantization)
group = parser.add_mutually_exclusive_group()
group.add_argument("--use_bnb", dest="use_bnb", action="store_true",
                   help="Enable 4-bit BitsAndBytes quantization.")
group.add_argument("--no_bnb", dest="use_bnb", action="store_false",
                   help="Disable 4-bit BitsAndBytes quantization.")
parser.set_defaults(use_bnb=True)

# LoRA hyperparams
parser.add_argument("--lora_r", type=int, default=64, help="LoRA rank r.")
parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha.")
parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout.")

# Training hyperparams
parser.add_argument("--num_train_epochs", type=int, default=2, help="Number of training epochs.")
parser.add_argument("--per_device_train_batch_size", type=int, default=16, help="Per-device train batch size.")
parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate.")
parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
TEMP_DIR = "./output"
os.makedirs(TEMP_DIR, exist_ok=True)

# ----------------------------
# 数据加载与预处理
# ----------------------------
train = pd.read_csv('./input/map-charting-student-math-misunderstandings/train.csv')

# 将 Misconception 的缺失值填充为 'NA'（与提交格式一致）
train.Misconception = train.Misconception.fillna('NA')

# 组合目标标签：Category:Misconception（与提交要求一致）
train['target'] = train.Category + ":" + train.Misconception

# 将字符串标签编码为整数类别 id（用于监督训练）
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train['label'] = le.fit_transform(train['target'])
n_classes = len(le.classes_) # 唯一标签类别数（num_labels）
print(f"Train shape: {train.shape} with {n_classes} target classes")
print("Train head:")
print(train.head())

# ----------------------------
# 清洗数据， 有些题目有多个选项是正确的， 只保留最常见的正确选项作为“正确答案”
# ----------------------------
# 对 Category 以 '_' 切分，前缀为 'True' 视为正确
idx = train.apply(lambda row: row.Category.split('_')[0], axis=1) == 'True'
# 从 True 子集内，按 (QuestionId, MC_Answer) 计数，挑选每题最常见的正确 MC_Answer
correct = train.loc[idx].copy() # 仅保留“正确”子集
correct['c'] = correct.groupby(['QuestionId', 'MC_Answer']).MC_Answer.transform('count') # 统计每个问题的MC_Answer出现次数
correct = correct.sort_values('c', ascending=False)
correct = correct.drop_duplicates(['QuestionId'])
correct = correct[['QuestionId', 'MC_Answer']] 
correct['is_correct'] = 1 # 标记这些 (QuestionId, MC_Answer) 为正确选项

# 将 is_correct 合并回训练集，其他样本缺失视为 0（错误）
train = train.merge(correct, on=['QuestionId', 'MC_Answer'], how='left')
train.is_correct = train.is_correct.fillna(0)


# ----------------------------
# 模型加载和配置
# ----------------------------
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

model_name = args.model_name

# BitsAndBytes 4-bit 量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

from_pretrained_kwargs = dict(
    num_labels=n_classes, # 分类头类别数
    device_map="auto",
    trust_remote_code=True
)
# 如开启 4-bit，则注入量化配置
if args.use_bnb:
    from_pretrained_kwargs["quantization_config"] = bnb_config

# 加载分类模型（会在最后一层添加分类 head）
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    **from_pretrained_kwargs
)

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# 配置 LoRA
lora_config = LoraConfig(
    r=args.lora_r,
    lora_alpha=args.lora_alpha,
    target_modules=["q_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=args.lora_dropout,
    bias="none",
    task_type="SEQ_CLS",
    modules_to_save=["score"]  # 保留额外模块（若模型内部使用）
)

# 将模型准备为 k-bit 友好的可训练状态
model = prepare_model_for_kbit_training(model)
# 注入 LoRA 适配器
model = get_peft_model(model, lora_config)

model = model.to(dtype=torch.bfloat16)

# 显式添加一个 [PAD] 记号
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
# 由于分词器词表变更，需要同步调整模型的嵌入矩阵大小
model.resize_token_embeddings(len(tokenizer))
# 设置模型的 pad_token_id，确保批量 padding 正确
model.config.pad_token_id = tokenizer.pad_token_id

print(next(model.parameters()).dtype)


# ----------------------------
# 构造训练输入文本
# ----------------------------
def format_input_v2(row):
    """
    构造单条训练输入文本：
    包含题目文本、学生选择、选择是否正确以及学生解释。
    该文本将被 tokenizer 截断到 max_length。
    """
    x = "Yes"
    if not row['is_correct']:
        x = "No"
    return (
        f"Question: {row['QuestionText']}\n"
        f"Answer: {row['MC_Answer']}\n"
        f"Is Correct Answer: {x}\n"
        f"Student Explanation: {row['StudentExplanation']}"
    )

# 基于上述格式化函数构造训练文本列
train['text'] = train.apply(format_input_v2, axis=1)
print("\nExample prompt for our LLM (after refinement):")
print(train.text.values[0])

from datasets import Dataset


# 构建干净的 DataFrame（防止多余列影响）
train_df_clean = train[['text', 'label']].copy()
train_df_clean['label'] = train_df_clean['label'].astype(np.int64)
train_df_clean = train_df_clean.reset_index(drop=True)

# 全量训练集
train_ds = Dataset.from_pandas(train_df_clean, preserve_index=False)

def tokenize(batch):
    """对文本批量分词并截断到 256"""
    return tokenizer(batch["text"], truncation=True, max_length=256)

# 进行分词映射；移除原始 'text' 列，仅保留 token 化后的特征与 'label'
train_ds = train_ds.map(tokenize, batched=True, remove_columns=['text'])


# ----------------------------
# 训练
# ----------------------------
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding

# 确保日志与输出目录存在
os.makedirs(f"{TEMP_DIR}/training_output/", exist_ok=True)
os.makedirs(f"{TEMP_DIR}/logs/", exist_ok=True)

training_args = TrainingArguments(
    output_dir=f"{TEMP_DIR}/training_output/",  # 输出目录
    do_train=True,                               # 仅训练，不做评估
    do_eval=False,
    save_strategy="no",                          # 不保存中间检查点
    num_train_epochs=args.num_train_epochs,      # 训练轮次
    per_device_train_batch_size=args.per_device_train_batch_size,  # batch 大小
    learning_rate=args.learning_rate,            # 学习率
    logging_dir=f"{TEMP_DIR}/logs/",             # 日志目录
    logging_steps=100,                           # 日志频率（步）
    gradient_accumulation_steps=args.gradient_accumulation_steps,  # 梯度累积
    remove_unused_columns=False,                 # 保留数据集中所有列（配合自定义处理）
    bf16=True,                                   # 使用 bfloat16 计算
    fp16=False,                                  # 不使用 float16
    report_to="none",                            # 不上报到外部可视化工具
    warmup_ratio=0.1,                            # 预热比例
    lr_scheduler_type="cosine",                  # 余弦学习率调度
    dataloader_drop_last=True,                   # 丢弃最后一个不完整 batch
    dataloader_pin_memory=False,                 # 禁用 pin_memory（按需调整）
    gradient_checkpointing=True,                 # 启用梯度检查点以省显存
)

# 使用动态 padding 的数据整理器，确保批间对齐
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    processing_class=tokenizer,
    data_collator=data_collator,
)

trainer.train()

print(f"Save model")
trainer.save_model(TEMP_DIR)


'''
#  https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507
torchrun --nproc_per_node=2 train.py \
  --cuda "0,1" \
  --model_name "Qwen/Qwen3-4B-Instruct-2507" \
  --no_bnb \
  --lora_r 512 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 8 \
  --learning_rate 2e-4 \
  --gradient_accumulation_steps 2


# https://huggingface.co/Qwen/Qwen3-8B
torchrun --nproc_per_node=2 train.py \
  --cuda "0,1" \
  --model_name "Qwen/Qwen3-8B" \
  --use_bnb \
  --lora_r 64 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --num_train_epochs 2 \
  --per_device_train_batch_size 16 \
  --learning_rate 2e-4 \
  --gradient_accumulation_steps 1



# https://huggingface.co/Qwen/Qwen3-14B
torchrun --nproc_per_node=4 train.py \
  --cuda "0,1,2,3" \
  --model_name "Qwen/Qwen3-14B" \
  --use_bnb \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 8 \
  --learning_rate 2e-4 \
  --gradient_accumulation_steps 2
'''