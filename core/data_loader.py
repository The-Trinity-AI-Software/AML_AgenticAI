
import pandas as pd

def load_data(train_path, test_path, feature_cols):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    train_encoded = pd.get_dummies(train_df[feature_cols], drop_first=True)
    test_encoded = pd.get_dummies(test_df[feature_cols], drop_first=True)
    train_encoded, test_encoded = train_encoded.align(test_encoded, join="left", axis=1, fill_value=0)
    return train_df, test_df, train_encoded, test_encoded
