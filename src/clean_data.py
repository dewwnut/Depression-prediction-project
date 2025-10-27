import pandas as pd

# Load dataset
df = pd.read_csv("C:/Users/Owner/depression-prediction/data/reddit_depression_dataset.csv")  # replace with your file path
print(f"Original dataset shape: {df.shape}")

# Drop unnecessary columns
columns_to_drop = ['Unnamed: 0', 'subreddit', 'created_utc']  # adjust if needed
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
print(f"After dropping unnecessary columns: {df.shape}")

# Handle missing values
# Drop rows where title or body is missing 
df = df.dropna(subset=['title', 'body'])

# Fill missing numeric values with 0
numeric_cols = ['upvotes', 'num_comments']
for col in numeric_cols:
    if col in df.columns:
        df[col] = df[col].fillna(0)

print(f"After handling missing values: {df.shape}")

# Combine title and body into one text column for NLP
df['text'] = df['title'] + ' ' + df['body']

# Remove duplicates based on text
df = df.drop_duplicates(subset=['text'])
df = df.reset_index(drop=True)
print(f"After removing duplicates: {df.shape}")

# Save cleaned CSV
cleaned_file_path = 'data/reddit_depression_cleaned.csv'
df.to_csv(cleaned_file_path, index=False)
print(f"Cleaned dataset saved to: {cleaned_file_path}")
