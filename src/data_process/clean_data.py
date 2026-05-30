import numpy as np
import pandas as pd
import os

dfs = []
all_columns = set()

# =========================
# READ + CLEAN + ENCODE
# =========================
for i in range(1, 11):

    df = pd.read_csv(f'data/period_{i}.csv')
    df = df.drop_duplicates()   # Remove duplicates

    # Check null
    print(f'period_{i}')
    print(df.isnull().sum())
    print('-' * 50)

    df = df.dropna()            # Drop null

    int_columns = ['Age', 'Tenure', 'Support Calls', 'Last Interaction']
    str_columns = ['Gender', 'Subscription Type', 'Contract Length']
    dummy_cols = df.select_dtypes(bool).columns

    df['CustomerID'] = df['CustomerID'].astype('int64')
    df[int_columns] = df[int_columns].astype(int)
    df = pd.get_dummies(df, columns=str_columns, drop_first=True)   # One-hot encoding
    df[dummy_cols] = df[dummy_cols].astype(int)                     # Bool -> int

    # Save dataframe temporarily
    dfs.append(df)

    # Collect all columns
    all_columns.update(df.columns)


# =========================
# ALIGN ALL DATAFRAMES
# =========================
all_columns = sorted(all_columns)

for i, df in enumerate(dfs):

    # Add missing columns
    df = df.reindex(
        columns=all_columns,
        fill_value=0
    )

    # Save cleaned file
    os.makedirs('Data/data_after_cleaning', exist_ok=True)
    df.to_csv(
        f'Data/data_after_cleaning/period_{i+1}.csv',
        index=False
    )

    print(
        f'period_{i+1} saved - shape: {df.shape}'
    )