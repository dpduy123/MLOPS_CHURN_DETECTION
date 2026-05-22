import numpy as np
import pandas as pd
import os

dfs = []
all_columns = set()

# =========================
# READ + CLEAN + ENCODE
# =========================
for i in range(1, 11):

    df = pd.read_csv(
        f'Data/period_{i}.csv'
    )

    # Remove duplicates
    df = df.drop_duplicates()

    # Check null
    print(f'period_{i}')
    print(df.isnull().sum())
    print('-' * 50)

    # Drop null
    df = df.dropna()

    # Convert datatype
    df['CustomerID'] = df['CustomerID'].astype('int64')

    int_columns = [
        'Age',
        'Tenure',
        'Support Calls',
        'Last Interaction'
    ]

    df[int_columns] = df[int_columns].astype(int)

    # One-hot encoding
    str_columns = [
        'Gender',
        'Subscription Type',
        'Contract Length'
    ]

    df = pd.get_dummies(
        df,
        columns=str_columns,
        drop_first=True
    )

    # Bool -> int
    dummy_cols = df.select_dtypes(bool).columns
    df[dummy_cols] = df[dummy_cols].astype(int)

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