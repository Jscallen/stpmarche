# fichier: api.py
import fastapi
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import umap
import string
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Initialisation des objets NLTK
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Création de l'API FastAPI
app = FastAPI()

# Charger les modèles pré-entraînés (pickle)
with open("vectorizer.pkl", "rb") as f:
    vectorizer = joblib.load(f)

with open("umap_reducer.pkl", "rb") as f:
    reducer = joblib.load(f)

with open("kmeans_model.pkl", "rb") as f:
    kmeans = joblib.load(f)

# Modèle Pydantic pour la requête
class TextRequest(BaseModel):
    title: str

# Fonction de prétraitement du texte
def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    processed_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and word not in string.punctuation]
    return " ".join(processed_tokens)

# Route pour prédire le cluster à partir d'un titre
@app.post("/predict/")
async def predict_cluster(data: TextRequest):
    # Prétraiter le titre
    cleaned_title = preprocess_text(data.title)

    # Vectoriser le texte avec TF-IDF
    X = vectorizer.transform([cleaned_title]).toarray()

    # Réduction de dimension avec UMAP
    X_embedded_umap = reducer.transform(X)

    # Prédiction du cluster avec KMeans
    cluster = kmeans.predict(X_embedded_umap)

    return {"cluster": int(cluster[0])}
