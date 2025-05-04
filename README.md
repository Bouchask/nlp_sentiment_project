NLP Sentiment Analysis Project
This Streamlit application analyzes the sentiment of YouTube comments (in English) for popular TV series: Game of Thrones, La Casa de Papel (Money Heist), and Stranger Things. It fetches videos from YouTube, collects comments, filters English comments, and performs sentiment analysis using a fine-tuned DistilRoBERTa model. Results are displayed with interactive visualizations (bar charts, word clouds) and downloadable CSV files.
Features

Select a series from a dropdown menu (Game of Thrones, La Casa de Papel, Stranger Things).
Display YouTube videos (trailers, reviews, reactions) with pagination (10 videos per page).
Analyze sentiments of English comments with a fine-tuned NLP model.
Visualize sentiment distribution (bar chart) and frequent words (word cloud).
Download analysis results as CSV.

Prerequisites

Python: Version 3.8 or higher.
Git: For cloning the repository.
Google Cloud Account: To obtain a YouTube Data API v3 key.
Weights & Biases (W&B): Optional, for model training logs (requires WANDB_API_KEY).

Installation
Step 1: Clone the Repository
Clone the project to your local machine:
git clone https://github.com/your-username/nlp_sentiment_project.git
cd nlp_sentiment_project

Step 2: Set Up a Virtual Environment
Create and activate a virtual environment to manage dependencies:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Step 3: Install Dependencies
Install the required Python packages listed in requirements.txt:
pip install -r requirements.txt

Step 4: Configure Environment Variables
Create a .env file in the project root and add your API keys:
echo "YOUTUBE_API_KEY=your_youtube_api_key_here
WANDB_API_KEY=your_wandb_api_key_here" > .env


YouTube API Key:
Go to Google Cloud Console.
Create a project or select an existing one.
Enable the YouTube Data API v3: "APIs & Services" > "Library" > Search for "YouTube Data API v3" > "Enable".
Create an API key: "Credentials" > "Create Credentials" > "API Key".
Copy the key and add it to .env.


W&B API Key (optional): Obtain from Weights & Biases if used for model training.

Step 5: Download or Train the Model
The application uses a fine-tuned DistilRoBERTa model (models/distilroberta_finetuned_en). If not present, it falls back to distilroberta-base. To use the fine-tuned model:

Download the model (if provided) and place it in the models/ directory.
Or train your own model using the scripts in src/ (requires W&B for logging).

Running the Application
Launch the Streamlit application:
streamlit run src/app.py


The app will open in your browser at http://localhost:8501.
Select a series, browse videos, and analyze comments.

Usage

Select a Series: Choose Game of Thrones, La Casa de Papel, or Stranger Things from the dropdown.
Browse Videos: View YouTube videos (10 per page) with pagination buttons ("Previous"/"Next").
Analyze Comments: Click "Analyze Comments" on a video to fetch English comments, predict sentiments, and view results.
Visualizations: See sentiment distribution (bar chart) and word cloud.
Download Results: Export analysis as a CSV file.

Technologies Used

Python 3.8+: Core programming language.
Streamlit: Web framework for the interactive UI.
YouTube Data API v3: Fetches videos and comments.
DistilRoBERTa: NLP model for sentiment analysis (via Hugging Face Transformers).
Plotly: Interactive bar charts for sentiment distribution.
WordCloud: Generates word clouds for comment analysis.
Langdetect: Filters English comments.
Pandas: Data manipulation and CSV export.
Google API Client: Interacts with YouTube API.
Weights & Biases (W&B): Optional, for model training and logging.
Dotenv: Manages environment variables.
Matplotlib: Renders word clouds.

Project Structure
```bash
nlp_sentiment_project/
├── data/                   # Output CSVs (processed_comments_*.csv)
├── models/                 # Fine-tuned DistilRoBERTa model (optional)
├── src/                    # Source code
│   ├── app.py              # Main Streamlit application
│   ├── data_collection.py  # YouTube API data fetching
│   ├── nlp.py              # Sentiment analysis logic
│   ├── preprocessing.py    # Text preprocessing
├── .env                    # Environment variables (not tracked)
├── .gitignore              # Ignored files
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
```

Notes

The YouTube API has a daily quota (10,000 units). If exceeded, wait until midnight UTC-8 or create a new API key.
Ensure .env is not pushed to GitHub (included in .gitignore).
For model training, refer to W&B documentation and training scripts in src/.

Troubleshooting

Quota Exceeded: Create a new API key or wait for quota reset.
No Videos Found: Check API key or broaden search terms in app.py.
Model Not Found: Ensure models/distilroberta_finetuned_en exists or use the base model.
Dependency Issues: Verify Python version and re-run pip install -r requirements.txt.

License
This project is licensed under the MIT License.
