import pandas as pd
import tiktoken
from sklearn.model_selection import train_test_split


def count_tokens(str_):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return len(encoding.encode(str_))


if __name__ == "__main__":
    df = pd.read_csv("data/raw.tsv", sep="\t", names=["label", "text"]).assign(
        text=lambda df_: df_["text"].str.strip()
    )

    print("Stats for Words per SMS")
    print(df["text"].str.split().str.len().describe())

    print("Stats for Tokens per SMS")
    print(df["text"].apply(count_tokens).describe())

    print("Class Balances")
    print(df["label"].value_counts())

    # discard all but 400 observations randomly, but with stratification
    sample_df, _ = train_test_split(df, stratify=df["label"], train_size=400)
    sample_df.to_csv("data/sample.csv", index=False)

    # split into eval and holdout
    eval_df, holdout_df = train_test_split(
        sample_df, stratify=sample_df["label"], train_size=0.5
    )
    eval_df.to_csv("data/eval.csv", index=False)
    holdout_df.to_csv("data/holdout.csv", index=False)
