import streamlit as st
import pandas as pd
from preprocessing import preprocess_text, preprocess_dataframe
from nlp import SentimentClassifier
from data_collection import collect_youtube_comments
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import logging
import os
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from datetime import datetime
from langdetect import detect, LangDetectException

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Clé API YouTube (doit être dans .env)
from dotenv import load_dotenv
load_dotenv()
YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')

# Vérifier la clé API
if not YOUTUBE_API_KEY:
    st.error("Clé API YouTube manquante. Ajoutez YOUTUBE_API_KEY dans le fichier .env.")
    st.stop()

# Initialisation de l'API YouTube
try:
    youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
except Exception as e:
    st.error(f"Erreur lors de l'initialisation de l'API YouTube : {str(e)}. Vérifiez votre clé API dans Google Cloud Console : https://console.cloud.google.com/apis/credentials")
    logger.error(f"Erreur API YouTube : {str(e)}")
    st.stop()

@st.cache_resource
def load_classifier():
    """Charge le modèle de classification fine-tuné."""
    model_path = 'models/distilroberta_finetuned_en'
    if not os.path.exists(model_path):
        logger.warning("Modèle fine-tuné non trouvé, chargement du modèle de base")
        model_path = 'distilroberta-base'
    return SentimentClassifier(model_path)

def filter_english_comments(df):
    """Filtre les commentaires en anglais uniquement."""
    def is_english(text):
        try:
            return detect(text) == 'en' if len(text) > 5 else False
        except LangDetectException:
            return False
    
    # Ajouter une colonne pour la langue détectée
    df['language'] = df['comment'].apply(lambda x: 'en' if is_english(x) else 'non_en')
    english_df = df[df['language'] == 'en'].copy()
    logger.info(f"Filtrage des commentaires : {len(english_df)} commentaires en anglais sur {len(df)}")
    return english_df

@st.cache_data(ttl=86400)  # Cache les résultats pendant 24 heures
def search_series_videos(series_name, max_results=15):
    """Recherche des vidéos YouTube pour une série spécifique."""
    try:
        # Définir des requêtes spécifiques pour chaque série
        series_queries = {
            "Game of Thrones": "Game of Thrones review trailer reaction discussion analysis series",
            "La Casa de Papel": "La Casa de Papel Money Heist review trailer reaction discussion analysis series",
            "Stranger Things": "Stranger Things review trailer reaction discussion analysis series"
        }
        query = series_queries.get(series_name, f"{series_name} review trailer reaction discussion analysis series")
        
        request = youtube.search().list(
            part="snippet",
            q=query,
            type="video",
            maxResults=max_results,
            order="relevance",  # Changé à "relevance" pour de meilleurs résultats
            relevanceLanguage="en"
        )
        response = request.execute()
        
        series_list = []
        for item in response['items']:
            title = item['snippet']['title'].lower()
            description = item['snippet']['description'].lower()
            # Filtre souple pour inclure les vidéos pertinentes
            if any(term in title or term in description for term in ["series", "review", "trailer", "reaction", "discussion", "analysis"]):
                series_list.append({
                    'video_id': item['id']['videoId'],
                    'title': item['snippet']['title'],
                    'thumbnail': item['snippet']['thumbnails']['medium']['url'],
                    'channel': item['snippet']['channelTitle']
                })
        
        # Supprimer les doublons (basé sur video_id)
        seen_ids = set()
        unique_series = []
        for item in series_list:
            if item['video_id'] not in seen_ids:
                unique_series.append(item)
                seen_ids.add(item['video_id'])
        
        logger.info(f"{len(unique_series)} vidéos trouvées pour la série : {series_name}")
        return unique_series
    except HttpError as e:
        error_message = f"Erreur API YouTube : {str(e)}"
        if "quota" in str(e).lower():
            error_message = "Quota API YouTube dépassé. Essayez demain ou créez une nouvelle clé API dans Google Cloud Console : https://console.cloud.google.com/apis/credentials"
        st.error(error_message)
        logger.error(f"Erreur API : {str(e)}")
        return []
    except Exception as e:
        st.error(f"Erreur lors de la recherche de vidéos pour '{series_name}' : {str(e)}")
        logger.error(f"Erreur générale : {str(e)}")
        return []

def main():
    """Lance l'application Streamlit."""
    # CSS personnalisé pour les cartes et le style
    st.markdown("""
        <style>
        .series-card {
            border: 2px solid #d0d0d0;
            border-radius: 15px;
            padding: 15px;
            margin: 15px 0;
            background-color: #ffffff;
            box-shadow: 0 6px 12px rgba(0,0,0,0.15);
            transition: transform 0.3s ease;
        }
        .series-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        }
        .series-title {
            font-size: 20px;
            font-weight: bold;
            color: #333;
            margin-bottom: 8px;
        }
        .series-channel {
            font-size: 14px;
            color: #777;
            font-style: italic;
        }
        .thumbnail {
            border-radius: 10px;
            max-width: 100%;
            border: 1px solid #e0e0e0;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 8px;
            padding: 10px 20px;
            font-weight: bold;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("Analyse des Sentiments pour Séries Populaires")
    st.write("Analysez les sentiments des commentaires YouTube (en anglais) pour Game of Thrones, La Casa de Papel, ou Stranger Things.")
    st.info("Note : L'application utilise l'API YouTube, qui a un quota limité (10 000 unités/jour). Si le quota est dépassé, essayez demain ou créez une nouvelle clé API dans Google Cloud Console : https://console.cloud.google.com/apis/credentials")

    # Bouton pour réinitialiser le cache
    if st.button("Réinitialiser la recherche"):
        st.session_state.series_list = []
        st.session_state.page_index = 0
        st.cache_data.clear()
        st.rerun()

    # Sélection de la série
    series_options = ["Game of Thrones", "La Casa de Papel", "Stranger Things"]
    selected_series = st.selectbox("Choisir une série", series_options)
    search_query_display = selected_series

    # Initialiser l'état de la session pour la pagination
    if 'page_index' not in st.session_state:
        st.session_state.page_index = 0
    if 'series_list' not in st.session_state:
        st.session_state.series_list = []
    if 'current_series' not in st.session_state:
        st.session_state.current_series = None

    # Afficher les performances du modèle
    evaluation_path = 'data/evaluation_results_en.csv'
    if os.path.exists(evaluation_path):
        st.subheader("Performances du Modèle")
        eval_df = pd.read_csv(evaluation_path)
        format_dict = {col: "{:.4f}" for col in eval_df.columns if col != 'Confusion Matrix'}
        st.dataframe(eval_df.style.format(format_dict))

    # Recherche des vidéos pour la série sélectionnée
    st.subheader(f"Vidéos pour : {search_query_display}")
    if selected_series != st.session_state.current_series or not st.session_state.series_list:
        st.session_state.series_list = search_series_videos(selected_series, max_results=15)
        st.session_state.current_series = selected_series
        st.session_state.page_index = 0
    
    series_list = st.session_state.series_list
    videos_per_page = 10
    
    # Calculer les indices pour la page actuelle
    start_idx = st.session_state.page_index * videos_per_page
    end_idx = start_idx + videos_per_page
    total_pages = (len(series_list) + videos_per_page - 1) // videos_per_page
    
    # Afficher les boutons de navigation
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("Précédent", disabled=st.session_state.page_index == 0):
            st.session_state.page_index -= 1
            st.rerun()
    with col2:
        st.write(f"Page {st.session_state.page_index + 1} / {total_pages if total_pages > 0 else 1}")
    with col3:
        if st.button("Suivant", disabled=end_idx >= len(series_list)):
            st.session_state.page_index += 1
            st.rerun()

    # Afficher les vidéos de la page actuelle
    if series_list:
        displayed_series = series_list[start_idx:end_idx]
        cols = st.columns(2)
        for idx, series in enumerate(displayed_series):
            with cols[idx % 2]:
                st.markdown(
                    f"""
                    <div class="series-card">
                        <img src="{series['thumbnail']}" class="thumbnail" alt="thumbnail">
                        <div class="series-title">{series['title'][:60]}...</div>
                        <div class="series-channel">Chaîne: {series['channel']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                if st.button(f"Analyser les commentaires", key=f"analyze_{series['video_id']}"):
                    with st.spinner(f"Collecte des commentaires pour {series['title'][:50]}..."):
                        search_query = f"{series['title']} review reaction discussion analysis english"
                        max_comments = 20  # Réduit pour économiser le quota
                        df = collect_youtube_comments(search_query, max_videos=1, max_comments_per_video=max_comments)
                        
                        if not df.empty:
                            # Filtrer les commentaires en anglais
                            df = filter_english_comments(df)
                            if df.empty:
                                st.warning("Aucun commentaire en anglais trouvé pour cette vidéo.")
                            else:
                                df = preprocess_dataframe(df)
                                classifier = load_classifier()
                                predictions = classifier.predict(df['cleaned_comment'].tolist(), threshold=0.5)
                                results_df = pd.DataFrame(predictions)
                                
                                st.write("Résultats de l'analyse des commentaires (en anglais) :")
                                st.dataframe(results_df[['text', 'label', 'score']].style.format({'score': '{:.2%}'}))

                                # Visualisation améliorée de la distribution des sentiments
                                st.write("Distribution des Sentiments (Positif vs Négatif) :")
                                sentiment_counts = results_df['label'].value_counts().reset_index()
                                sentiment_counts.columns = ['Sentiment', 'Nombre']
                                total = sentiment_counts['Nombre'].sum()
                                sentiment_counts['Pourcentage'] = sentiment_counts['Nombre'] / total * 100
                                
                                fig = px.bar(
                                    sentiment_counts,
                                    x='Sentiment',
                                    y='Nombre',
                                    text=[f'{p:.1f}%' for p in sentiment_counts['Pourcentage']],
                                    title="Distribution des Sentiments",
                                    labels={'Sentiment': 'Sentiment', 'Nombre': 'Nombre de Commentaires'},
                                    color='Sentiment',
                                    color_discrete_map={'POSITIVE': '#1f77b4', 'NEGATIVE': '#ff7f0e'},
                                    template='plotly_white'
                                )
                                fig.update_traces(
                                    marker_line_color='black',
                                    marker_line_width=1.5,
                                    opacity=0.9,
                                    textposition='auto'
                                )
                                fig.update_layout(
                                    title_font_size=20,
                                    xaxis_title_font_size=16,
                                    yaxis_title_font_size=16,
                                    xaxis_tickfont_size=14,
                                    yaxis_tickfont_size=14,
                                    showlegend=False,
                                    margin=dict(t=50, b=50, l=50, r=50),
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    bargap=0.2
                                )
                                st.plotly_chart(fig, use_container_width=True)

                                # Nuage de mots amélioré
                                st.write("Nuage de Mots des Commentaires (en anglais) :")
                                text = ' '.join(results_df['text'].tolist())
                                wordcloud = WordCloud(
                                    width=800,
                                    height=400,
                                    background_color='white',
                                    colormap='viridis',
                                    font_path=None,
                                    min_font_size=10,
                                    max_font_size=150
                                ).generate(text)
                                plt.figure(figsize=(10, 5), facecolor='white')
                                plt.imshow(wordcloud, interpolation='bilinear')
                                plt.axis('off')
                                st.pyplot(plt)

                                # Téléchargement des résultats
                                output_path = f'data/processed_comments_{series["video_id"]}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
                                results_df.to_csv(output_path, index=False)
                                with open(output_path, 'rb') as f:
                                    st.download_button(
                                        label="Télécharger les résultats (CSV)",
                                        data=f,
                                        file_name=os.path.basename(output_path),
                                        mime='text/csv'
                                    )
                                st.success(f"Résultats sauvegardés dans {output_path}")
                        else:
                            st.error("Aucun commentaire collecté. Vérifiez votre clé API ou essayez une autre vidéo.")
    else:
        st.error(f"Aucune vidéo trouvée pour '{search_query_display}'. Vérifiez votre clé API dans Google Cloud Console : https://console.cloud.google.com/apis/credentials ou essayez une requête plus générale.")

if __name__ == "__main__":
    main()