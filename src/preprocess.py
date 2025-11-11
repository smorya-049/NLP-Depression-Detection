import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Load dataset
df = pd.read_csv(r"data/raw/depression_dataset.csv")

# Rename columns to standard names for ML training
df = df.rename(columns={'clean_text': 'text', 'is_depression': 'label'})

# Remove null/empty rows just in case
df = df.dropna(subset=['text'])

# Remove duplicates
df = df.drop_duplicates(subset=['text'])

# Train-test split (80% train, 20% test)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

# Create output directory if not exists
os.makedirs("data/processed", exist_ok=True)

# Save the processed datasets
train_df.to_csv("data/processed/train.csv", index=False)
test_df.to_csv("data/processed/test.csv", index=False)

print("Preprocessing complete!")
print("Training samples:", len(train_df))
print("Testing samples:", len(test_df))
