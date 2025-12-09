import pandas as pd

def load_and_clean(file_path):
    df = pd.read_csv("../data/MachineLearningRating_v3.txt", sep="|", engine="python", encoding="utf-8")
    # Example: convert columns to lowercase
    df.columns = [c.lower() for c in df.columns]
    # Fill missing values (example)
    df['totalclaims'] = df['totalclaims'].fillna(0)
    df['totalpremium'] = df['totalpremium'].fillna(df['totalpremium'].median())
    return df

if __name__ == "__main__":
    df = load_and_clean("data/MachineLearningRating_v3.txt")
    df.to_csv("data/MachineLearningRating_clean.csv", index=False)
