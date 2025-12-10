import pandas as pd
import os

def load_and_clean(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Read the text file with pipe separator
    input_file = "data/MachineLearningRating_v3.txt"
    df = pd.read_csv(input_file, sep="|", engine="python", encoding="utf-8")

    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Ensure numeric columns are properly typed
    df['totalclaims'] = pd.to_numeric(df['totalclaims'], errors='coerce').fillna(0)
    df['totalpremium'] = pd.to_numeric(df['totalpremium'], errors='coerce').fillna(df['totalpremium'].median())

    return df

if __name__ == "__main__":
    input_file = "data/MachineLearningRating_v3.txt"
    output_file = "data/MachineLearningRating_clean.csv"

    df = load_and_clean(input_file)
    df.to_csv(output_file, index=False)
    print(f"File created: {output_file}")

