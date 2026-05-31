import pandas as pd
import numpy as np
import os  

df_train = pd.read_csv('data/org/customer_churn_dataset-training-master.csv')

output_dir = 'data/raw'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

splits = np.array_split(df_train, 10)
for i, split in enumerate(splits):
    split.to_csv(f'{output_dir}/period_{i+1}.csv', index=False)
    print(f"Đã lưu: {output_dir}/period_{i+1}.csv")