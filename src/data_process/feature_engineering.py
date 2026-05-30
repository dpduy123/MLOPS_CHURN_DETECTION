import numpy as np
import pandas as pd
import os

for i in range(1, 11):
    df = pd.read_csv(f'Data/data_after_cleaning/period_{i}.csv')

    df['Tenure_Age_Ratio'] = df['Tenure'] / (df['Age'] + 1)
    df['Spend_per_Usage'] = df['Total Spend'] / (df['Usage Frequency'] + 1)
    df['Support_Calls_per_Tenure'] = df['Support Calls'] / (df['Tenure'] + 1)


    os.makedirs('Data/Data_v1', exist_ok=True)
    df.to_csv(f'Data/Data_v1/period_{i}.csv', index = False)

    df['Spending_Group'] = pd.qcut(
        df['Total Spend'],
        q=4,
        labels=False
    )

    # Create tenure groups
    df['Tenure_Group'] = pd.cut(
        df['Tenure'],
        bins=[0, 12, 24, 36, 100],
        labels=False
    )
    os.makedirs('Data/Data_v2', exist_ok=True)
    df.to_csv(f'Data/Data_v2/period_{i}.csv', index = False)