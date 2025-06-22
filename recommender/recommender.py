import numpy as np
import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

# Load movie data
movies_raw = pd.read_csv("ml-100k/u.item", sep="|", encoding="latin-1", header=None)
movies_df = movies_raw[[0, 1]].copy()
movies_df.columns = ["movieId", "title"]
movies_df['genre'] = ""  # Placeholder if no genre column
movies_df['tags'] = (movies_df['title'] + " " + movies_df['genre']).str.lower()

# Content-based vectorization
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies_df['tags']).toarray()

sim_matrix_path = "similarity_matrix.npy"
if os.path.exists(sim_matrix_path):
    similarity_matrix = np.load(sim_matrix_path)
else:
    similarity_matrix = cosine_similarity(vectors)
    np.save(sim_matrix_path, similarity_matrix)

def recommend_content(movie_name):
    movie_name = movie_name.lower()
    if movie_name in movies_df['title'].str.lower().values:
        idx = movies_df[movies_df['title'].str.lower() == movie_name].index[0]
        sim_scores = list(enumerate(similarity_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
        return [movies_df.iloc[i[0]].title for i in sim_scores]
    return ["Movie not found."]

def recommend_collaborative(user_id):
    return movies_df.sample(5)["title"].tolist()

def recommend_hybrid_weighted(user_id, fav_movies):
    fav_movies = [m.strip().lower() for m in fav_movies]
    score_dict = defaultdict(float)

    for fav_movie in fav_movies:
        if fav_movie in movies_df['title'].str.lower().values:
            idx = movies_df[movies_df['title'].str.lower() == fav_movie].index[0]
            sim_scores = list(enumerate(similarity_matrix[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
            for i in sim_scores:
                title = movies_df.iloc[i[0]].title
                score_dict[title] += 1.0 * i[1]

    if score_dict:
        sorted_recs = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
        return [f"{title} — Score: {score:.3f}" for title, score in sorted_recs[:10]]
    return ["No hybrid recommendations found."]
