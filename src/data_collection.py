import os
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import pandas as pd
import logging
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from langdetect import detect, LangDetectException
from dotenv import load_dotenv

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Charger les variables d'environnement
load_dotenv()

def detect_language(text: str) -> str:
    try:
        if not isinstance(text, str) or pd.isna(text) or len(text.strip()) < 5:
            return 'unknown'
        lang = detect(text)
        return lang if lang == 'en' else 'unknown'
    except LangDetectException:
        return 'unknown'

def simple_sentiment_label(text: str) -> str:
    """Étiquetage initial des sentiments basé sur des mots-clés."""
    positive_keywords = ['great', 'awesome', 'love', 'amazing', 'fantastic', 'good', 'excellent']
    negative_keywords = ['bad', 'terrible', 'hate', 'awful', 'disappointing', 'poor']
    text_lower = text.lower()
    if any(keyword in text_lower for keyword in positive_keywords):
        return 'POSITIVE'
    elif any(keyword in text_lower for keyword in negative_keywords):
        return 'NEGATIVE'
    return 'NEUTRAL'

def collect_youtube_comments(search_queries: list, max_videos: int = 100, max_comments_per_video: int = 500) -> pd.DataFrame:
    """Collecte les commentaires YouTube via l'API pour plusieurs requêtes."""
    try:
        youtube = build('youtube', 'v3', developerKey=os.getenv('YOUTUBE_API_KEY'))
        all_comments = []
        
        for search_query in search_queries:
            logger.info(f"Recherche pour la requête : {search_query}")
            # Rechercher des vidéos
            search_response = youtube.search().list(
                q=search_query,
                part='id,snippet',
                maxResults=max_videos,
                type='video'
            ).execute()

            video_ids = [item['id']['videoId'] for item in search_response['items']]
            logger.info(f"{len(video_ids)} vidéos trouvées pour la requête : {search_query}")

            for video_id in video_ids:
                try:
                    # Vérifier si les commentaires sont activés
                    video_response = youtube.videos().list(
                        part='snippet,statistics',
                        id=video_id
                    ).execute()
                    if 'commentCount' not in video_response['items'][0]['statistics'] or int(video_response['items'][0]['statistics']['commentCount']) == 0:
                        logger.warning(f"Commentaires désactivés ou absents pour la vidéo {video_id}")
                        continue

                    # Récupérer les commentaires avec pagination
                    next_page_token = None
                    comments_fetched = 0
                    while comments_fetched < max_comments_per_video:
                        comment_response = youtube.commentThreads().list(
                            part='snippet',
                            videoId=video_id,
                            maxResults=100,  # Maximum par page
                            textFormat='plainText',
                            pageToken=next_page_token
                        ).execute()

                        for item in comment_response['items']:
                            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                            all_comments.append({
                                'comment': comment,
                                'source': f'YouTube_{video_id}',
                                'language': detect_language(comment),
                                'initial_label': simple_sentiment_label(comment)
                            })
                            comments_fetched += 1
                            if comments_fetched >= max_comments_per_video:
                                break

                        next_page_token = comment_response.get('nextPageToken')
                        if not next_page_token:
                            break

                    logger.info(f"{comments_fetched} commentaires collectés pour la vidéo {video_id}")

                except HttpError as e:
                    if 'commentsDisabled' in str(e):
                        logger.warning(f"Commentaires désactivés pour la vidéo {video_id}")
                    else:
                        logger.warning(f"Erreur lors de la récupération des commentaires pour la vidéo {video_id}: {str(e)}")
                    continue

        df = pd.DataFrame(all_comments)
        df = df[df['language'] == 'en'].reset_index(drop=True)
        logger.info(f"{len(df)} commentaires collectés après filtrage par langue")
        return df

    except Exception as e:
        logger.error(f"Erreur lors de la collecte des commentaires: {str(e)}")
        return pd.DataFrame()

def save_to_csv(df: pd.DataFrame, filename: str) -> None:
    try:
        df.to_csv(filename, index=False, encoding='utf-8')
        logger.info(f"Données sauvegardées dans {filename}")
        language_counts = df['language'].value_counts().to_dict()
        logger.info("Nombre de lignes par langue :")
        for lang, count in language_counts.items():
            logger.info(f"  - {lang}: {count}")
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde du fichier CSV: {str(e)}")

if __name__ == "__main__":
    SEARCH_QUERIES = [
        "Stranger Things review",
        "Stranger Things season 4 reaction",
        "Stranger Things analysis",
        "Stranger Things trailer reaction",
        "Stranger Things best moments"
    ]
    df = collect_youtube_comments(SEARCH_QUERIES, max_videos=100, max_comments_per_video=500)
    if not df.empty:
        save_to_csv(df, 'data/youtube_comments.csv')