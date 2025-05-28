import pandas as pd
import os

# Load the dataset
df = pd.read_excel('dataset/Rice_MSC_Dataset.xlsx')

# Drop the 'CLASS' column if it exists
columns_to_process = df.drop(columns=['CLASS'], errors='ignore')

# Calculate Q1 and Q3
q1 = columns_to_process.quantile(0.25)
q3 = columns_to_process.quantile(0.75)

# Combine into a DataFrame
quartiles_df = pd.DataFrame({
    'FEATURE': q1.index,
    'Q1': q1.values,
    'Q3': q3.values
})

# Create output directory if it doesn't exist
os.makedirs('dataset', exist_ok=True)

# Save to Excel
quartiles_df.to_excel('dataset/iqr_results.xlsx', index=False)

print("Saved Q1 and Q3 values to 'dataset/iqr_result_results.xlsx'")