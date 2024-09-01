import numpy as np
import pandas as pd
import pickle
import yaml
from sklearn.ensemble import GradientBoostingClassifier
import logging

# logging configure

logger = logging.getLogger("model_building")
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

def load_params(params_path: str) -> dict:
    try:
        params = yaml.safe_load(open(params_path, 'r'))['model_building']
        logger.debug("Model building parameters retrieved from params file")
        return params
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

def train_model(X_train: np.ndarray, y_train: np.ndarray, params: dict) -> GradientBoostingClassifier:
    try:
        clf = GradientBoostingClassifier(n_estimators=params['n_estimators'], learning_rate=params['learning_rate'])
        clf.fit(X_train, y_train)
        logger.debug("Model training completed")
        return clf
    except Exception as e:
        logger.error(f"Failed to train model: {e}")
        raise

def save_model(model: GradientBoostingClassifier, model_path: str) -> None:
    try:
        with open(model_path, 'wb') as model_file:
            pickle.dump(model, model_file)
        logger.debug(f"Model saved successfully at {model_path}")
    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        raise

def main():
    params = load_params('params.yaml')

    train_data = read_data('./data/processed/train_bow.csv')

    X_train = train_data.iloc[:, 0:-1].values
    y_train = train_data.iloc[:, -1].values

    clf = train_model(X_train, y_train, params)

    save_model(clf, 'models/model.pkl')

if __name__ == "__main__":
    main()
