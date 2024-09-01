import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import yaml
import logging

# logging configure

logger = logging.getLogger("feature_engineering")
logger.setLevel('DEBUG')

# console handler
console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)

# file handler
file_handler = logging.FileHandler('errors.log')
file_handler.setLevel('ERROR')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def load_params(params_path: str) -> int:
    try:
        max_features = yaml.safe_load(open(params_path, 'r'))['feature_engineering']['max_features']
        logger.debug("Max features retrieved from params file")
        return max_features
    except Exception as e:
        logger.error(f"Params file not found or invalid: {e}")
        raise

def read_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logger.debug(f"Data read successfully from {file_path}")
        return df
    except Exception as e:
        logger.error(f"Failed to read data from {file_path}: {e}")
        raise

def process_data(train_data: pd.DataFrame, test_data: pd.DataFrame, max_features: int) -> (pd.DataFrame, pd.DataFrame):
    try:
        train_data.fillna('', inplace=True)
        test_data.fillna('', inplace=True)
        
        vectorizer = TfidfVectorizer(max_features=max_features)

        X_train_bow = vectorizer.fit_transform(train_data['content'].values)
        X_test_bow = vectorizer.transform(test_data['content'].values)

        train_df = pd.DataFrame(X_train_bow.toarray())
        train_df['label'] = train_data['sentiment'].values

        test_df = pd.DataFrame(X_test_bow.toarray())
        test_df['label'] = test_data['sentiment'].values
        
        logger.debug("Data processing with CountVectorizer completed")
        return train_df, test_df
    except Exception as e:
        logger.error(f"Failed to process data: {e}")
        raise

def save_data(data_path: str, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    try:
        os.makedirs(data_path, exist_ok=True)
        train_df.to_csv(os.path.join(data_path, "train_tfidf.csv"), index=False)
        test_df.to_csv(os.path.join(data_path, "test_tfidf.csv"), index=False)
        logger.debug(f"Processed data saved successfully in {data_path}")
    except Exception as e:
        logger.error(f"Failed to save processed data: {e}")
        raise

def main():
    max_features = load_params('params.yaml')

    train_data = read_data('./data/interim/train_processed.csv')
    test_data = read_data('./data/interim/test_processed.csv')

    train_df, test_df = process_data(train_data, test_data, max_features)

    data_path = os.path.join("data", "processed")
    save_data(data_path, train_df, test_df)

if __name__ == "__main__":
    main()
