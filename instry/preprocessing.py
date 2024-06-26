import pandas as pd

def load_dataset(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    # Combine hard_skills, soft_skills, and extra_skills into a single 'skills' column
    df['skills'] = df.apply(lambda row: ' '.join([str(row['hard_skills']), str(row['soft_skills']), str(row['extra_skills'])]), axis=1)
    return df
