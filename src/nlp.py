import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, get_cosine_schedule_with_warmup
from datasets import DatasetDict
import evaluate
from typing import List, Dict
import logging
import numpy as np
import wandb

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def split_dataset(dataset) -> DatasetDict:
    """
    Divise le dataset en ensembles d'entraînement, de validation et de test.

    Args:
        dataset: Dataset Hugging Face à diviser.

    Returns:
        DatasetDict: Contient les splits 'train', 'validation' et 'test'.
    """
    try:
        # Diviser en train (80%) et temp (20%)
        train_test_split = dataset.train_test_split(test_size=0.2, seed=42)
        # Diviser temp en validation (10%) et test (10%)
        val_test_split = train_test_split['test'].train_test_split(test_size=0.5, seed=42)
        
        # Créer un DatasetDict
        dataset_dict = DatasetDict({
            'train': train_test_split['train'],
            'validation': val_test_split['train'],
            'test': val_test_split['test']
        })
        
        logger.info(f"Dataset divisé : {len(dataset_dict['train'])} train, "
                    f"{len(dataset_dict['validation'])} validation, "
                    f"{len(dataset_dict['test'])} test")
        return dataset_dict
    except Exception as e:
        logger.error(f"Erreur lors de la division du dataset: {str(e)}")
        raise

class CustomTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs):
        loss = super().training_step(model, inputs)
        wandb.log({"train_loss": loss})
        return loss

    def evaluation_step(self, inputs):
        outputs = super().evaluation_step(inputs)
        eval_loss = outputs.get("eval_loss")
        if eval_loss is not None:
            wandb.log({"eval_loss": eval_loss})
        return outputs

class SentimentClassifier:
    def __init__(self, model_path: str = 'distilroberta-base', language: str = 'en'):
        try:
            self.language = language
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            logger.info(f"Modèle chargé: {model_path} pour {language} sur {self.device}")
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle: {str(e)}")
            raise

    def tokenize_dataset(self, dataset):
        return dataset.map(
            lambda x: self.tokenizer(x['text'], truncation=True, padding='max_length', max_length=128),
            batched=True,
            remove_columns=["text"]
        ).with_format("torch")

    def train_model(self, dataset: DatasetDict, output_dir: str = 'models/distilroberta_finetuned') -> None:
        try:
            # Initialiser W&B
            wandb.init(project="sentiment_analysis_youtube", config={
                "model": "distilroberta-base",
                "language": self.language,
                "epochs": 8,
                "batch_size": 32,
                "learning_rate": 3e-5,
            })

            os.makedirs(output_dir, exist_ok=True)
            tokenized_dataset = self.tokenize_dataset(dataset)
            tokenized_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

            # Calcul des poids des classes
            num_positive = sum(1 for x in dataset['train']['label'] if x == 1)
            num_negative = sum(1 for x in dataset['train']['label'] if x == 0)
            total = num_positive + num_negative
            weight_positive = (1 / num_positive) * (total / 2.0)
            weight_negative = (1 / num_negative) * (total / 2.0)
            class_weights = torch.tensor([weight_negative, weight_positive]).to(self.device)

            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=8,
                per_device_train_batch_size=32,
                per_device_eval_batch_size=32,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="f1",
                logging_dir=f"{output_dir}/logs",
                logging_steps=50,
                learning_rate=3e-5,
                warmup_steps=1000,
                weight_decay=0.1,
                lr_scheduler_type="cosine",
                report_to="wandb",
            )

            metric = evaluate.load("f1")
            def compute_metrics(eval_pred):
                logits, labels = eval_pred
                predictions = np.argmax(logits, axis=-1)
                f1 = metric.compute(predictions=predictions, references=labels, average='weighted')
                wandb.log({"eval_f1": f1['f1']})
                return f1

            trainer = CustomTrainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_dataset['train'],
                eval_dataset=tokenized_dataset['validation'],
                compute_metrics=compute_metrics,
                class_weights=class_weights,
            )

            trainer.train()
            wandb.finish()
            self.save_model(output_dir)
            logger.info(f"Modèle entraîné et sauvegardé dans {output_dir}")
        except Exception as e:
            logger.error(f"Erreur lors de l'entraînement: {str(e)}")
            wandb.finish()
            raise

    def predict(self, texts: List[str], threshold: float = 0.5) -> List[Dict]:
        try:
            self.model.eval()
            predictions = []
            for text in texts:
                if not isinstance(text, str) or len(text.strip()) < 5:
                    predictions.append({'text': text, 'label': 'NEUTRAL', 'score': 0.5})
                    continue
                inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                    score = probs[:, 1].item()
                    label = 'POSITIVE' if score > threshold else 'NEGATIVE'
                predictions.append({'text': text, 'label': label, 'score': score})
            logger.info(f"Prédictions terminées pour {len(texts)} textes")
            return predictions
        except Exception as e:
            logger.error(f"Erreur lors de la prédiction: {str(e)}")
            return []

    def save_model(self, output_path: str) -> None:
        try:
            self.model.save_pretrained(output_path)
            self.tokenizer.save_pretrained(output_path)
            logger.info(f"Modèle sauvegardé dans {output_path}")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde du modèle: {str(e)}")