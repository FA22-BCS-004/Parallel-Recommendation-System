import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import random

# --- Load & Prepare Data ---
data_path = "book_data"
books_df = pd.read_csv(os.path.join(data_path, "Books.csv"), low_memory=False)
ratings_df = pd.read_csv(os.path.join(data_path, "Ratings.csv"))

ratings_df = ratings_df.dropna().drop_duplicates()
books_df = books_df.dropna(subset=['Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher'])

ratings_df = ratings_df.sample(n=100000, random_state=42).reset_index(drop=True)
merged_df = pd.merge(ratings_df, books_df, on="ISBN")
merged_df = merged_df[['User-ID', 'ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher']].dropna().reset_index(drop=True)

# --- TF-IDF Vectorization ---
book_id_to_desc = {
    row['ISBN']: f"{row['Book-Title']} {row['Book-Author']} {row['Year-Of-Publication']} {row['Publisher']}"
    for _, row in merged_df.iterrows()
}

book_id_to_title = dict(zip(merged_df['ISBN'], merged_df['Book-Title']))
isbn_list = list(book_id_to_desc.keys())

tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
tfidf_matrix = tfidf.fit_transform(book_id_to_desc.values())

isbn_to_tfidf_idx = {isbn: idx for idx, isbn in enumerate(isbn_list)}

# --- Recommendation Function ---
def recommend_book(query, top_n=5):
    try:
        matched_row = None
        for _, row in merged_df.iterrows():
            if query.lower() in row["Book-Title"].lower() or query.lower() in row["ISBN"].lower():
                matched_row = row
                break

        if not matched_row is None:
            isbn = matched_row["ISBN"]
            title = matched_row["Book-Title"]
            tfidf_index = isbn_to_tfidf_idx.get(isbn)

            if tfidf_index is None:
                return [f"No TF-IDF data for book: {title}"]

            cosine_scores = cosine_similarity(tfidf_matrix[tfidf_index], tfidf_matrix).flatten()

            tfidf_scores = {}
            for i in cosine_scores.argsort()[::-1]:
                if i == tfidf_index:
                    continue
                sim_isbn = isbn_list[i]
                tfidf_scores[sim_isbn] = cosine_scores[i]
                if len(tfidf_scores) >= 20:
                    break

            top_books = list(tfidf_scores.items())
            random.shuffle(top_books)
            top_books = top_books[:top_n]

            results = [book_id_to_title.get(isbn, "(No Title)") for isbn, _ in top_books]
            return results
        else:
            return ["❌ Book not found. Try with full or partial title or ISBN."]

    except Exception as e:
        return [f"❌ Error: {e}"]
