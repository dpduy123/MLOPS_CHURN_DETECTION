import pandas as pd
import numpy as np



df_train = pd.read_csv('..\\Data\\customer_churn_dataset-training-master.csv')



splits = np.array_split(df_train, 10)

for i, split in enumerate(splits):
    split.to_csv(f'..\\Data\\split_data_raw\\period_{i+1}.csv', index=False)