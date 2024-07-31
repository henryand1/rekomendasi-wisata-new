from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

app = Flask(__name__)

# Muat model yang telah dilatih
model = SentenceTransformer('sentence-transformer-indobert')

# Muat embedding deskripsi tempat wisata dari file .npy
description_embeddings = np.load('description_embeddings.npy')

# Muat dataset deskripsi tempat wisata
df = pd.read_csv('data/dataset-wisata-new.csv')
descriptions = df['Description'].dropna().tolist()
place_names = df['Place_Name'].dropna().tolist()
ratings = df['Rating'].dropna().tolist()

@app.route('/recommend', methods=['POST'])
def recommend():
    user_input = request.json.get('user_input')
    if not user_input:
        return jsonify({"error": "Description is required"}), 400

    user_embedding = model.encode([user_input])[0]

    # Hitung cosine similarity antara input pengguna dan deskripsi dalam dataset
    similarities = cosine_similarity([user_embedding], description_embeddings)[0]

    # Dapatkan indeks dari deskripsi dengan kesamaan tertinggi
    top_indices = np.argsort(similarities)[::-1][:5]

    # Ambil deskripsi, place_name, dan rating dengan kesamaan tertinggi
    recommendations = []
    for i in top_indices:
        recommendations.append({
            "place_name": place_names[i],
            "description": descriptions[i],
            "rating": ratings[i]
        })

    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True)
