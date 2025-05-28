import pandas as pd

# Load dataset
df = pd.read_excel("./dataset/Rice_MSC_Dataset.xlsx")

# Select a random sample of 100 rows from the dataset
sample_df = df.sample(n=100, random_state=42)

# Save the sample dataset to a new Excel file in /dataset as sample_rice_dataset.xlsx
sample_df.to_excel("./dataset/sample_rice_dataset.xlsx", index=False)