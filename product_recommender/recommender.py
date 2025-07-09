
import pandas as pd
import numpy as np
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def parse_json_lines(lines):
    results = []
    for line in lines:
        try:
            results.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return results

def load_product_data(json_file="product-recommendation/Electronics_5.json", sample_size=100000, workers=4):
    with open(json_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    chunk_size = len(lines) // workers
    chunks = [lines[i:i + chunk_size] for i in range(0, len(lines), chunk_size)]

    data = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for result in tqdm(executor.map(parse_json_lines, chunks), total=len(chunks), desc="Parsing JSON"):
            data.extend(result)

    df = pd.DataFrame(data)[['reviewerID', 'asin', 'reviewText', 'overall', 'summary']].dropna()
    df = df.drop_duplicates(subset=["asin", "reviewerID"])
    df = df.sample(n=min(sample_size, len(df)), random_state=42).reset_index(drop=True)
    return df

def train_models(df):
    asin_to_summary = dict(zip(df['asin'], df['summary']))
    asin_to_index = {asin.upper(): i for i, asin in enumerate(df['asin'])}
    tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf_matrix = tfidf.fit_transform(df["reviewText"].fillna(""))
    item_factors = np.random.rand(tfidf_matrix.shape[0], 20)
    idx_to_asin = {v: k for k, v in asin_to_index.items()}
    return asin_to_summary, asin_to_index, idx_to_asin, item_factors, tfidf_matrix, df

def get_product_recommendations(query, df, asin_to_summary, asin_to_index, idx_to_asin, item_factors, tfidf_matrix, top_n=5):
    query = query.strip().lower()
    matched_asin = None
    matched_idx = None

    for asin in asin_to_index:
        if query == asin.lower():
            matched_asin = asin
            matched_idx = asin_to_index[asin]
            break

    if matched_asin is None:
        for i, row in df.iterrows():
            if isinstance(row['summary'], str) and query == row['summary'].strip().lower():
                matched_asin = row['asin']
                matched_idx = asin_to_index.get(matched_asin.upper())
                break

    if matched_asin is None or matched_idx is None:
        return ["❌ Product not found."]

    cosine_scores = cosine_similarity(tfidf_matrix[matched_idx], tfidf_matrix).flatten()
    tfidf_scores = {df.iloc[i]['asin']: score for i, score in enumerate(cosine_scores) if i != matched_idx}

    top_asins = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    results = []
    for asin_id, score in top_asins:
        title = asin_to_summary.get(asin_id, "(No summary)")
        results.append(f"{title} (ASIN: {asin_id}) — Score: {score:.3f}")

    return results
