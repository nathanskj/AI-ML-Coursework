import pandas as pd

# Load dataset
df = pd.read_csv("hour.csv")

# Create demand class
df['demand_class'] = pd.qcut(df['cnt'], q=3, labels=['Low', 'Medium', 'High'])

# Drop columns
df = df.drop(columns=['instant', 'dteday', 'casual', 'registered', 'cnt'])

# Convert categories
categories = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit']
df[categories] = df[categories].astype('category')

# One-hot encode
df_preprocessed = pd.get_dummies(df, drop_first=True)

# Save versions
df.to_csv("original_data.csv", index=False)
df_preprocessed.to_csv("preprocessed_data.csv", index=False)
