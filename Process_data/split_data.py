import pandas as pd
import numpy as np



df_train = pd.read_csv('..\\Data\\customer_churn_dataset-training-master.csv')
df_test = pd.read_csv('..\\Data\\customer_churn_dataset-testing-master.csv')

df = pd.concat([df_train, df_test], ignore_index=True)


splits = np.array_split(df, 10)

for i, split in enumerate(splits):
    split.to_csv(f'..\\Data\\period_{i+1}.csv', index=False)