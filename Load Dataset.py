import pandas as pd
file_path = "news.csv"
df = pd.read_csv(file_path)
print("Shape of the dataset:", df.shape)
print("\nData types:\n", df.dtypes)
print("\nMissing values in each column:\n", df.isnull().sum())
print(df.head())
