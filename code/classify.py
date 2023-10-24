import pandas as pd
from prompt_classifier import classify_text_message
from sklearn.metrics import classification_report
from tqdm import tqdm
import openai
# this is so we can use progress_apply
tqdm.pandas(desc="progress")


def load_data(path):
    """Loads a CSV, expecting columns "label" and "text".
    Returns a pd.DataFrame with series named "label" and "text".
    """
    df = pd.read_csv(path)
    if "label" in df.columns and "text" in df.columns:
        return df[["label", "text"]]
    else:
        raise ValueError("CSV must have 'label' and 'text' columns.")    

def classify_all(df, api_key):
    """Takes a DataFrame with series named "label" and "text".
    Returns a pd.DataFrame with series named "label" and "text"
    and also "predicted_label" which is our zero-shot classifier's prediction.
    """
    openai.api_key = api_key
    df["predicted_label"] = df["text"].progress_apply(lambda x: classify_text_message(x, api_key))
    return df

    # in order to be able to use a progress bar with epd.DataFram
    # tqdm can modify pd.DataFrame's methods to include `progress_apply`
    # which works like apply, but adds a progress bar


if __name__ == "__main__":
    import os
    import sys

    import dotenv

    dotenv.load_dotenv()

    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

    IN_PATH = sys.argv[1]
    OUT_PATH = sys.argv[2]

    out_dir = os.path.dirname(OUT_PATH)
    os.makedirs(out_dir, exist_ok=True)

    in_df = load_data(IN_PATH)

    out_df = classify_all(in_df, api_key=OPENAI_API_KEY)

    out_df.to_csv(OUT_PATH, index=False)

    print(
        classification_report(y_true=out_df["label"], y_pred=out_df["predicted_label"])
    )
