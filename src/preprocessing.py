import re
import emoji
import pandas as pd
from datasets import Dataset
from langdetect import detect, LangDetectException
import logging
from multiprocessing import Pool
import contractions
import ftfy

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def detect_language(text: str) -> str:
    try:
        if not isinstance(text, str) or pd.isna(text):
            return 'unknown'
        lang = detect(text)
        return lang if lang == 'en' else 'unknown'
    except LangDetectException:
        return 'unknown'

def parallel_detect_language(texts):
    with Pool() as pool:
        return pool.map(detect_language, texts)

def preprocess_text(text: str) -> str:
    try:
        if not isinstance(text, str) or pd.isna(text) or len(text.strip()) < 5:
            return ""
        
        # Corriger les encodages
        text = ftfy.fix_text(text)
        
        # Convertir les emojis en texte
        text = emoji.demojize(text)
        
        # Étendre les contractions (ex: "I'm" -> "I am")
        text = contractions.fix(text)
        
        # Supprimer les URLs, mentions, hashtags
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+|\#\w+', '', text)
        
        # Conserver certains caractères spéciaux et convertir en minuscules
        text = re.sub(r'[^\w\s!?]', '', text).lower()
        
        # Supprimer les espaces multiples
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    except Exception as e:
        logger.error(f"Erreur lors du prétraitement du texte: {str(e)}")
        return ""

def preprocess_dataframe(df: pd.DataFrame, text_column: str = 'comment') -> pd.DataFrame:
    try:
        df['language'] = parallel_detect_language(df[text_column].tolist())
        df['cleaned_comment'] = df[text_column].apply(preprocess_text)
        df = df[df['cleaned_comment'].str.len() > 10]  # Filtrer les commentaires trop courts
        logger.info("Prétraitement terminé pour le DataFrame")
        return df
    except Exception as e:
        logger.error(f"Erreur lors du prétraitement du DataFrame: {str(e)}")
        return df

def prepare_dataset(df: pd.DataFrame, text_column: str = 'cleaned_comment', label_column: str = 'initial_label') -> Dataset:
    try:
        # Exclure les commentaires neutres
        df = df[df[label_column].isin(['POSITIVE', 'NEGATIVE'])]
        df['label'] = df[label_column].map({'POSITIVE': 1, 'NEGATIVE': 0})
        dataset = Dataset.from_pandas(df[[text_column, 'label', 'language']].rename(columns={text_column: 'text'}))
        logger.info(f"Dataset préparé pour l'entraînement : {len(dataset)} exemples")
        return dataset
    except Exception as e:
        logger.error(f"Erreur lors de la préparation du dataset: {str(e)}")
        raise

def save_processed_data(df: pd.DataFrame, output_path: str) -> None:
    try:
        df.to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f"Données prétraitées sauvegardées dans {output_path}")
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde des données prétraitées: {str(e)}")