import numpy as np
import pandas as pd




for i in range(1, 11):
    df = pd.read_csv(f'C:\\work\\Churn\\Data\\Split_data_raw\\period_{i}.csv')
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    print(df.isnull().sum())
    
    # drop null values
    df = df.dropna()

    # convert datatypes

    df['CustomerID'] = df['CustomerID'].astype("int64")
    int_columns = ['Age', 'Tenure', 'Support Calls', 'Last Interaction']
    df[int_columns] = df[int_columns].astype(int)
    
    # Convert categorical variables to numeric using one-hot encoding
    str_columns = ['Gender', 'Subscription Type', 'Contract Length']

    df = pd.get_dummies(df, columns=str_columns, drop_first=True)
    dummy_cols = df.select_dtypes(bool).columns
    df[dummy_cols] = df[dummy_cols].astype(int)

    df.to_csv(f'C:\\work\\Churn\\Data\\data_after_cleaning\\period_{i}.csv', index=False)