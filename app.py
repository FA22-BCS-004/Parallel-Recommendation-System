import os
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from book_recommender.recommender import recommend_book
from product_recommender.recommender import train_models, load_product_data, get_product_recommendations
# ✅ CORRECT import for product recommender (parallelized code)
from product_recommender.recommender import train_models, load_product_data, get_product_recommendations


app = Flask(__name__)

# --------------------- Movie Recommender -------------------------
movies_path = "ml-100k/u.item"

movies_df = pd.read_csv(
    movies_path,
    sep="|",
    encoding='latin-1',
    header=None,
    names=[
        "movieId", "title", "release_date", "video_release_date", "IMDb_URL",
        "unknown", "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
        "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
        "Romance", "Sci-Fi", "Thriller", "War", "Western"
    ]
)

genre_cols = ["unknown", "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
              "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
              "Romance", "Sci-Fi", "Thriller", "War", "Western"]

movie_vector_dict = {
    row["movieId"]: row[genre_cols].values
    for _, row in movies_df.iterrows()
}
movie_features = movies_df[["movieId", "title"] + genre_cols]

# --------------------- Product Recommender -------------------------
json_path = os.path.join("product-recommendation", "Electronics_5.json")
df = load_product_data(json_file=json_path, sample_size=5000)
asin_to_summary, asin_to_index, idx_to_asin, item_factors, tfidf_matrix, df = train_models(df)

# ------------------- ROUTES ------------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/guest')
def guest():
    return render_template('index.html')

@app.route('/content')
def content():
    return render_template('content.html')

@app.route('/collaborative')
def collaborative():
    return render_template('collaborative.html')

@app.route('/hybrid')
def hybrid():
    return render_template('hybrid.html')

@app.route('/book')
def book_page():
    print("✅ Reached /book route")
    return render_template('book.html')

@app.route('/book_recommend', methods=['POST'])
def book_recommend():
    query = request.form.get('book_query')
    if not query:
        return "❌ No query received."

    recommendations = recommend_book(query, top_n=5)
    return render_template('book_results.html', recommendations=recommendations)

@app.route('/recommend_content', methods=['POST'])
def recommend_content():
    movie_title = request.form.get("movie_title")
    input_vector = None
    input_movie_id = None

    for _, row in movie_features.iterrows():
        if row["title"].lower() == movie_title.lower():
            input_vector = row[genre_cols].values
            input_movie_id = row["movieId"]
            break

    if input_vector is None:
        return "Movie not found."

    similarities = []
    for movie_id, vector in movie_vector_dict.items():
        if movie_id != input_movie_id:
            sim = cosine_similarity([input_vector], [vector])[0][0]
            similarities.append((movie_id, sim))

    similarities.sort(key=lambda x: x[1], reverse=True)
    top_movies = [movie_id for movie_id, _ in similarities[:5]]
    recommendations = movies_df[movies_df["movieId"].isin(top_movies)]["title"].tolist()

    return render_template("results.html", recommendations=recommendations)

@app.route('/recommend_collaborative', methods=['POST'])
def recommend_collaborative():
    top_movies = movies_df["title"].head(5).tolist()
    return render_template("results.html", recommendations=top_movies)

@app.route('/recommend_hybrid', methods=['POST'])
def recommend_hybrid():
    movie_title = request.form.get("movie_title")
    user_id = request.form.get("user_id")

    content_recs = []
    input_vector = None
    input_movie_id = None

    for _, row in movie_features.iterrows():
        if row["title"].lower() == movie_title.lower():
            input_vector = row[genre_cols].values
            input_movie_id = row["movieId"]
            break

    if input_vector is not None:
        similarities = []
        for movie_id, vector in movie_vector_dict.items():
            if movie_id != input_movie_id:
                sim = cosine_similarity([input_vector], [vector])[0][0]
                similarities.append((movie_id, sim))
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_ids = [movie_id for movie_id, _ in similarities[:5]]
        content_recs = movies_df[movies_df["movieId"].isin(top_ids)]["title"].tolist()

    collaborative_recs = movies_df["title"].sample(5).tolist()
    hybrid_recommendations = list(dict.fromkeys(content_recs + collaborative_recs))

    return render_template("results.html", recommendations=hybrid_recommendations)

@app.route('/product')
def product_page():
    return render_template('products.html')

@app.route('/product_recommend', methods=['POST'])
def product_recommend():
    query = request.form.get('product_query')
    if not query:
        return "❌ No query received from form."

    result = get_product_recommendations(query, df, asin_to_summary, asin_to_index, idx_to_asin, item_factors, tfidf_matrix)
    return render_template('product_results.html', result=result)


if __name__ == '__main__':
    app.run(debug=True)
