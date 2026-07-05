# 🏆 Kaggle | MAP - Charting Student Math Misunderstandings

This repository contains the core training and inference pipeline for the **Kaggle MAP@3 Student Math Misunderstanding** competition. Our team achieved a Private LB score of **0.946** using an LLM-based end-to-end multi-class classification approach.

## 🚀 Key Highlights & Contributions

As a core algorithm developer in this project, my main contributions focused on **engineering optimization, LLM fine-tuning, and ensemble strategies**:

1. **End-to-End Single Space Modeling:** Replaced the traditional cascaded classification with a unified `Category × Misconception` mapping space using **Qwen3 (4B/8B/14B) + LoRA**.
2. **Memory-Efficient Fine-Tuning:** Successfully trained 14B models on limited VRAM hardware utilizing **4-bit Quantization (nf4)** via `BitsAndBytes`, combined with `bfloat16` mixed precision and gradient checkpointing.
3. **Structural Prior (Prompt Engineering):** Designed a `True_/False_family` prefix logic based on statistical distributions of correct MC_Answers, significantly improving the LLM's zero-shot generalization and filtering out illegal predictions during post-processing.
4. **Consistency-Weighted Ensemble:** Implemented a robust multi-model ensemble algorithm combining Total Probability (0.34) + Consistency Ratio (0.33) + Max Confidence (0.33), boosting the LB score from 0.945 to 0.946.

## 📂 Repository Structure

```text
├── docs/
│   └── solution.md          # Full solution report (Chinese)
├── src/
│   ├── train.py             # Qwen3 LoRA fine-tuning script
│   ├── inference.py         # Single model inference (Top-25 logits)
│   ├── inference_ensemble.py # 3-model weighted ensemble (4B+8B+14B)
│   ├── inference_script.ipynb # Ensemble notebook (family filter + submission)
│   └── data_preprocess.py   # True/False family prompt construction
├── requirements.txt         # Dependencies
├── .gitignore
└── README.md
```

## ⚙️ How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Data Preparation

Download competition data from [Kaggle MAP](https://kaggle.com/competitions/map-charting-student-math-misunderstandings) and place CSVs in `data/` directory.

### 3. Model Training (Example for Qwen3-8B)

```bash
python src/train.py --model_name "Qwen/Qwen3-8B" --quantization "4bit" --batch_size 8
```

### 4. Single Model Inference

```bash
python src/inference.py --model_path ./output --test_data ./data/test.csv
```

### 5. Ensemble & Submit

Run `src/inference_script.ipynb` in Jupyter/Kaggle Notebook to ensemble 4B+8B+14B predictions and generate submission file.

Or use the CLI ensemble:
```bash
python src/inference_ensemble.py --weights "4b,8b,14b"
```

## 🛠 Tech Stack

Python | PyTorch | Transformers | PEFT/LoRA | BitsAndBytes | Pandas
