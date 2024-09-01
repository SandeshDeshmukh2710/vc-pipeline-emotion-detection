import numpy as np
import pandas as pd
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
import logging

# logging configure

logger = logging.getLogger("data_preprocessing")
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

def read_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logger.debug(f"Data read successfully from {file_path}")
        return df
    except Exception as e:
        logger.error(f"Failed to read data from {file_path}: {e}")
        raise

def lemmatization(text: str) -> str:
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = [lemmatizer.lemmatize(y) for y in text]
    return " ".join(text)

def remove_stop_words(text: str) -> str:
    stop_words = set(stopwords.words("english"))
    text = [i for i in str(text).split() if i not in stop_words]
    return " ".join(text)

def removing_numbers(text: str) -> str:
    text = ''.join([i for i in text if not i.isdigit()])
    return text

def lower_case(text: str) -> str:
    text = text.split()
    text = [y.lower() for y in text]
    return " ".join(text)

def removing_punctuations(text: str) -> str:
    try:
        text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
        text = text.replace('؛', "", )
        text = re.sub('\s+', ' ', text)
        text = " ".join(text.split())
        return text.strip()
    except Exception as e:
        logger.error(f"Failed to remove punctuations: {e}")
        raise

def removing_urls(text: str) -> str:
    try:
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub(r'', text)
    except Exception as e:
        logger.error(f"Failed to remove URLs: {e}")
        raise

def remove_small_sentences(df: pd.DataFrame) -> None:
    try:
        for i in range(len(df)):
            if len(df.content.iloc[i].split()) < 3:
                df.content.iloc[i] = np.nan
        logger.debug("Removed small sentences")
    except Exception as e:
        logger.error(f"Failed to remove small sentences: {e}")
        raise

def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df['content'] = df['content'].apply(lower_case)
        df['content'] = df['content'].apply(remove_stop_words)
        df['content'] = df['content'].apply(removing_numbers)
        df['content'] = df['content'].apply(removing_punctuations)
        df['content'] = df['content'].apply(removing_urls)
        df['content'] = df['content'].apply(lemmatization)
        logger.debug("Text normalization completed")
        return df
    except Exception as e:
        logger.error(f"Failed to normalize text: {e}")
        raise

def save_data(data_path: str, train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
    try:
        os.makedirs(data_path, exist_ok=True)
        train_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
        test_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)
        logger.debug(f"Data saved successfully in {data_path}")
    except Exception as e:
        logger.error(f"Failed to save data: {e}")
        raise

def main():
    nltk.download('wordnet')
    nltk.download('stopwords')

    train_data = read_data('./data/raw/train.csv')
    test_data = read_data('./data/raw/test.csv')

    train_processed_data = normalize_text(train_data)
    test_processed_data = normalize_text(test_data)

    train_processed_data.fillna('', inplace=True)
    test_processed_data.fillna('', inplace=True)

    data_path = os.path.join("data", "interim")
    save_data(data_path, train_processed_data, test_processed_data)

if __name__ == "__main__":
    main()
