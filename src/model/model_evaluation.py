import pandas as pd
import numpy as np
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import logging

# logging configure

logger = logging.getLogger("model_evaluation")
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

def load_model(model_path: str):
    try:
        clf = pickle.load(open(model_path, 'rb'))
        logger.debug(f"Model loaded successfully from {model_path}")
        return clf
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}")
        raise

def read_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logger.debug(f"Data read successfully from {file_path}")
        return df
    except Exception as e:
        logger.error(f"Failed to read data from {file_path}: {e}")
        raise

def evaluate_model(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    try:
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        logger.debug("Model evaluation metrics calculated")
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc
        }
    except Exception as e:
        logger.error(f"Failed to evaluate model: {e}")
        raise

def save_metrics(metrics: dict, metrics_path: str) -> None:
    try:
        with open(metrics_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logger.debug(f"Metrics saved successfully to {metrics_path}")
    except Exception as e:
        logger.error(f"Failed to save metrics to {metrics_path}: {e}")
        raise

def main():
    clf = load_model('models/model.pkl')

    test_data = read_data('./data/processed/test_bow.csv')

    X_test = test_data.iloc[:, 0:-1].values
    y_test = test_data.iloc[:, -1].values

    metrics = evaluate_model(clf, X_test, y_test)

    save_metrics(metrics, 'reports/metrics.json')

if __name__ == "__main__":
    main()
