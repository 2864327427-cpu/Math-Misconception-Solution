import argparse
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder


def build_correct_answer_map(train_df: pd.DataFrame) -> pd.DataFrame:
    """Build per-question most frequent correct answer map from True_* rows."""
    is_true = train_df["Category"].str.split("_").str[0] == "True"
    correct = train_df.loc[is_true, ["QuestionId", "MC_Answer"]].copy()
    correct["count"] = correct.groupby(["QuestionId", "MC_Answer"])["MC_Answer"].transform("count")
    correct = correct.sort_values("count", ascending=False)
    correct = correct.drop_duplicates(["QuestionId"])
    correct = correct[["QuestionId", "MC_Answer"]]
    correct["is_correct"] = 1
    return correct


def add_is_correct_flag(df: pd.DataFrame, correct_map: pd.DataFrame) -> pd.DataFrame:
    """Attach is_correct flag by merging with question-level correct answer map."""
    out = df.merge(correct_map, on=["QuestionId", "MC_Answer"], how="left")
    out["is_correct"] = out["is_correct"].fillna(0).astype(int)
    return out


def build_family_prefix_map(test_df: pd.DataFrame) -> dict:
    """Return row_id -> True_/False_ mapping from is_correct."""
    fam_series = test_df["is_correct"].map(lambda x: "True_" if int(x) == 1 else "False_")
    return dict(zip(test_df["row_id"], fam_series))


def prepare_train_targets(train_df: pd.DataFrame) -> tuple[pd.DataFrame, LabelEncoder]:
    """Create target label Category:Misconception and integer class id."""
    train_df = train_df.copy()
    train_df["Misconception"] = train_df["Misconception"].fillna("NA")
    train_df["target"] = train_df["Category"] + ":" + train_df["Misconception"]
    encoder = LabelEncoder()
    train_df["label"] = encoder.fit_transform(train_df["target"])
    return train_df, encoder


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess MAP data and build family prefix features.")
    parser.add_argument("--train_csv", type=Path, required=True)
    parser.add_argument("--test_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("./artifacts"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)

    train_df, _ = prepare_train_targets(train_df)
    correct_map = build_correct_answer_map(train_df)

    train_processed = add_is_correct_flag(train_df, correct_map)
    test_processed = add_is_correct_flag(test_df, correct_map)

    family_map = build_family_prefix_map(test_processed)
    family_df = pd.DataFrame({"row_id": list(family_map.keys()), "family_prefix": list(family_map.values())})

    train_processed.to_csv(args.output_dir / "train_processed.csv", index=False)
    test_processed.to_csv(args.output_dir / "test_processed.csv", index=False)
    family_df.to_csv(args.output_dir / "family_prefix_map.csv", index=False)

    print("Saved:")
    print(args.output_dir / "train_processed.csv")
    print(args.output_dir / "test_processed.csv")
    print(args.output_dir / "family_prefix_map.csv")


if __name__ == "__main__":
    main()
