from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import pandas as pd
import logging
from typing import List, Dict

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_predictions(y_true: List[str], y_pred: List[str], y_scores: List[float] = None) -> Dict:
    """
    Évalue les prédictions du modèle avec des métriques.

    Args:
        y_true (List[str]): Liste des vraies étiquettes.
        y_pred (List[str]): Liste des étiquettes prédites.
        y_scores (List[float]): Scores de probabilité pour la classe POSITIVE (optionnel).

    Returns:
        Dict: Rapport de classification, matrice de confusion et ROC-AUC.
    """
    try:
        report = classification_report(y_true, y_pred, output_dict=True)
        cm = confusion_matrix(y_true, y_pred, labels=['NEGATIVE', 'POSITIVE'])
        roc_auc = roc_auc_score([1 if y == 'POSITIVE' else 0 for y in y_true], y_scores) if y_scores else None
        logger.info("Évaluation des prédictions terminée")
        return {
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'roc_auc': roc_auc
        }
    except Exception as e:
        logger.error(f"Erreur lors de l'évaluation: {str(e)}")
        return {}

def save_evaluation_results(evaluation: Dict, output_path: str = 'data/evaluation_results.csv') -> None:
    """
    Sauvegarde les résultats d'évaluation dans un fichier CSV.

    Args:
        evaluation (Dict): Résultats d'évaluation (rapport et matrice).
        output_path (str): Chemin du fichier CSV.
    """
    try:
        report = evaluation['classification_report']
        cm = evaluation['confusion_matrix']
        roc_auc = evaluation.get('roc_auc')
        
        # Créer un DataFrame avec les métriques
        metrics = {
            'Accuracy': report['accuracy'],
            'Positive Precision': report['POSITIVE']['precision'],
            'Positive Recall': report['POSITIVE']['recall'],
            'Positive F1': report['POSITIVE']['f1-score'],
            'Negative Precision': report['NEGATIVE']['precision'],
            'Negative Recall': report['NEGATIVE']['recall'],
            'Negative F1': report['NEGATIVE']['f1-score'],
            'ROC-AUC': roc_auc if roc_auc else 'N/A',
            'Confusion Matrix': str(cm)
        }
        
        df = pd.DataFrame([metrics])
        df.to_csv(output_path, index=False)
        logger.info(f"Résultats d'évaluation sauvegardés dans {output_path}")
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde des résultats: {str(e)}")

def evaluate_from_dataframe(df: pd.DataFrame, true_column: str = 'true_label', pred_column: str = 'label', score_column: str = 'score') -> Dict:
    """
    Évalue les prédictions à partir d'un DataFrame.

    Args:
        df (pd.DataFrame): DataFrame contenant les vraies étiquettes, prédictions et scores.
        true_column (str): Nom de la colonne des vraies étiquettes.
        pred_column (str): Nom de la colonne des prédictions.
        score_column (str): Nom de la colonne des scores.

    Returns:
        Dict: Rapport de classification, matrice de confusion et ROC-AUC.
    """
    try:
        y_true = df[true_column].tolist()
        y_pred = df[pred_column].tolist()
        y_scores = df[score_column].tolist() if score_column in df.columns else None
        return evaluate_predictions(y_true, y_pred, y_scores)
    except Exception as e:
        logger.error(f"Erreur lors de l'évaluation du DataFrame: {str(e)}")
        return {}