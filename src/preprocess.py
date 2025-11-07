import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_and_clean(path):
    df = pd.read_csv(path)
    df.dropna(inplace=True)
    
    # Encode categorical features
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col])
    
    # Scale numerical features
    num_cols = df.select_dtypes(include='number').columns
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    
    return df
