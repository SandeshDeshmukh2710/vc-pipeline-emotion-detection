import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
import os
import logging

# logging configure

logger = logging.getLogger("data_ingestion")
logger.setLevel('DEBUG')

# console handler
console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)

#file handler
file_handler = logging.FileHandler('errors.log')
file_handler.setLevel('ERROR')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def load_params(params_path: str) -> float: 
    try:
        test_size = yaml.safe_load(open(params_path, 'r'))['data_ingestion']['test_size']
        logger.debug("test size retrieved")
        return test_size
    except Exception as e:
        logger.error("Parmas File not found")
        print(e)
        raise
       

def read_data(url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(url)
        return df
    except Exception as e:
        print(f"Failed to read data from {url}: {e}")
        

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df.drop(columns=['tweet_id'], inplace=True)
        final_df = df[df['sentiment'].isin(['neutral', 'sadness'])]
        final_df['sentiment'].replace({'neutral': 1, 'sadness': 0}, inplace=True)
        return final_df
    except Exception as e:
        print(f"Failed to process data: {e}")
        

def save_data(data_path: str, train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
    try:
        os.makedirs(data_path, exist_ok=True)
        train_data.to_csv(os.path.join(data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(data_path, "test.csv"), index=False)
    except Exception as e:
        print(f"Failed to save data: {e}")

def main():
    test_size = load_params('params.yaml')
    df = read_data('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')
    
    if not df.empty:
        final_df = process_data(df)

        if not final_df.empty:
            train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)
            data_path = os.path.join("data", "raw")
            save_data(data_path, train_data, test_data)
        else:
            print("No data to split after processing.")
    else:
        print("No data loaded to process.")

if __name__ == "__main__":
    main()
