import pandas as pd

df = pd.read_csv("C:/Users/Owner/depression-prediction/data/reddit_depression_dataset.csv")
df_sample = df.sample(n=10000, random_state=42)  # adjust size as needed
df_sample.to_csv("C:/Users/Owner/depression-prediction/data/cleaned/reddit_depression_small.csv", index=False)
print("Saved small dataset for testing:", df_sample.shape)
