import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd


def extract_class_probabilities(row: pd.Series, suffix: str, top_k: int = 25) -> dict[str, float]:
    top_classes_col = f"top_classes{suffix}"
    class_names = str(row[top_classes_col]).split()

    class_probs: dict[str, float] = {}
    for i, class_name in enumerate(class_names[:top_k]):
        prob_col = f"prob_{i}{suffix}"
        if prob_col in row.index:
            class_probs[class_name] = float(row[prob_col])
    return class_probs


def ensemble_with_disagreement_handling(
    prob_files: list[Path],
    family_df: pd.DataFrame,
    model_weights: list[float],
    top_k: int = 3,
) -> pd.DataFrame:
    if len(prob_files) != len(model_weights):
        raise ValueError("Length mismatch: prob_files and model_weights must be equal.")

    prob_dfs = [pd.read_csv(p) for p in prob_files]

    merged_df = prob_dfs[0]
    for i, df in enumerate(prob_dfs[1:], 1):
        merged_df = merged_df.merge(df, on="row_id", suffixes=("", f"_model{i+1}"))

    fam_map = dict(zip(family_df["row_id"], family_df["family_prefix"]))
    n_models = len(prob_dfs)
    final_predictions = []

    for _, row in merged_df.iterrows():
        pref = fam_map.get(row["row_id"], "False_")

        all_class_probs = []
        for i in range(n_models):
            suffix = f"_model{i+1}" if i > 0 else ""
            all_class_probs.append(extract_class_probabilities(row, suffix=suffix, top_k=25))

        all_classes = set()
        for class_probs in all_class_probs:
            all_classes.update(class_probs.keys())

        class_votes = defaultdict(int)
        class_total_prob = defaultdict(float)
        class_max_prob = defaultdict(float)

        for i, class_probs in enumerate(all_class_probs):
            weight = model_weights[i]
            for class_name, prob in class_probs.items():
                class_votes[class_name] += 1
                class_total_prob[class_name] += prob * weight
                class_max_prob[class_name] = max(class_max_prob[class_name], prob * weight)

        final_scores: dict[str, float] = {}
        for class_name in all_classes:
            final_scores[class_name] = (
                class_total_prob[class_name] * 0.34
                + (class_votes[class_name] / n_models) * 0.33
                + class_max_prob[class_name] * 0.33
            )

        final_scores = {k: v for k, v in final_scores.items() if k.startswith(pref)}
        sorted_classes = sorted(final_scores.items(), key=lambda x: -x[1])
        top_classes = [class_name for class_name, _ in sorted_classes[:top_k]]

        fillers = [f"{pref}Neither:NA"] + ([f"{pref}Correct:NA"] if pref == "True_" else [])
        for filler in fillers:
            if len(top_classes) >= 3:
                break
            if filler not in top_classes:
                top_classes.append(filler)

        while len(top_classes) < 3:
            top_classes.append(fillers[0])

        final_predictions.append(" ".join(top_classes[:3]))

    return pd.DataFrame({"row_id": merged_df["row_id"].values, "Category:Misconception": final_predictions})


def main() -> None:
    parser = argparse.ArgumentParser(description="Ensemble Top-K class probabilities into final submission")
    parser.add_argument("--family_csv", type=Path, required=True, help="CSV with columns: row_id,family_prefix")
    parser.add_argument("--prob_files", type=Path, nargs="+", required=True)
    parser.add_argument("--weights", type=float, nargs="+", required=True)
    parser.add_argument("--output_csv", type=Path, default=Path("submission.csv"))
    parser.add_argument("--top_k", type=int, default=3)
    args = parser.parse_args()

    family_df = pd.read_csv(args.family_csv)
    submission = ensemble_with_disagreement_handling(
        prob_files=args.prob_files,
        family_df=family_df,
        model_weights=args.weights,
        top_k=args.top_k,
    )
    submission.to_csv(args.output_csv, index=False)
    print(f"Saved submission to: {args.output_csv}")


if __name__ == "__main__":
    main()
