import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import pandas as pd
from preprocessing import preprocess_dataframe, prepare_dataset
from nlp import SentimentClassifier, split_dataset
from evaluation import evaluate_predictions, save_evaluation_results
import logging
import wandb
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info("Script démarré")

def main():
    # Initialiser W&B avec la clé API
    wandb.login(key=os.getenv('WANDB_API_KEY'))
    
    logger.info("Début du processus d'entraînement")
    logger.info("Chargement des données depuis data/youtube_comments.csv")
    df = pd.read_csv('data/youtube_comments.csv')
    logger.info(f"Données chargées : {len(df)} lignes")
    
    logger.info("Prétraitement des données")
    df = preprocess_dataframe(df)
    logger.info("Prétraitement terminé")
    
    languages = ['en']
    for lang in languages:
        logger.info(f"Traitement de la langue : {lang}")
        lang_df = df[df['language'] == lang].copy()
        if len(lang_df) < 10:
            logger.warning(f"Pas assez de données pour la langue {lang}, saut...")
            continue
        
        logger.info("Préparation du dataset")
        dataset = prepare_dataset(lang_df)
        logger.info("Division du dataset en train/validation/test")
        dataset = split_dataset(dataset)
        logger.info("Dataset prêt pour l'entraînement")
        
        model_path = {
            'en': 'distilroberta-base',
        }.get(lang)
        
        logger.info(f"Initialisation du modèle pour {lang} avec {model_path}")
        classifier = SentimentClassifier(model_path=model_path, language=lang)
        output_dir = f'models/distilroberta_finetuned_{lang}'
        logger.info(f"Entraînement du modèle, sauvegarde dans {output_dir}")
        classifier.train_model(dataset, output_dir=output_dir)
        
        logger.info("Évaluation sur le split test")
        test_texts = dataset['test']['text']
        test_labels = ['POSITIVE' if label == 1 else 'NEGATIVE' for label in dataset['test']['label']]
        predictions = classifier.predict(test_texts)
        pred_labels = [pred['label'] for pred in predictions]
        pred_scores = [pred['score'] for pred in predictions]
        
        logger.info("Sauvegarde des résultats d'évaluation")
        evaluation = evaluate_predictions(test_labels, pred_labels, pred_scores)
        save_evaluation_results(evaluation, output_path=f'data/evaluation_results_{lang}.csv')
        
        # Logger les métriques d'évaluation dans W&B
        wandb.log({
            "test_accuracy": evaluation['classification_report']['accuracy'],
            "test_positive_f1": evaluation['classification_report']['POSITIVE']['f1-score'],
            "test_negative_f1": evaluation['classification_report']['NEGATIVE']['f1-score'],
            "test_roc_auc": evaluation.get('roc_auc', 0.0)
        })
        
        logger.info(f"Processus terminé pour la langue {lang}")

if __name__ == "__main__":
    main()