import pandas as pd

def get_feature_names():
    # Read only the first row from the Excel file to get the column names.
    df = pd.read_excel("./dataset/Rice_MSC_Dataset.xlsx", nrows=1)
    # Exclude the target column "CLASS" (case-insensitive)
    return [col for col in df.columns if col.upper() != "CLASS"]

# Optionally, cache the list for subsequent use
FEATURE_NAMES = get_feature_names()