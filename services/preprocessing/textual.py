import pandas as pd
import numpy as np
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer


def textual_training_dataset(reviews_df, products_df):

    df = reviews_df.copy()
    df["review_text"] = df["review_text"].fillna("")

    # -------------------------------------------------
    # 1️⃣ Basic Structural Features
    # -------------------------------------------------

    df["length"] = df["review_text"].apply(lambda x: len(str(x).split()))

    def capital_ratio(text):
        text = str(text)
        if len(text) == 0:
            return 0
        return sum(c.isupper() for c in text) / len(text)

    df["capital_ratio"] = df["review_text"].apply(capital_ratio)

    def punctuation_density(text):
        text = str(text)
        return len(re.findall(r"[!?,.;:]", text)) / (len(text) + 1)

    df["punctuation_density"] = df["review_text"].apply(punctuation_density)

    # Structural indicator score
    df["structural_score"] = (
        df["capital_ratio"] * 0.4 +
        df["punctuation_density"] * 0.4 +
        (df["length"] < 5).astype(int) * 0.2
    )

    # -------------------------------------------------
    # 2️⃣ Sentiment Score (VADER)
    # -------------------------------------------------

    analyzer = SentimentIntensityAnalyzer()

    def sentiment_scores(text):
        scores = analyzer.polarity_scores(text)
        return pd.Series([
            scores["compound"],
            abs(scores["compound"])
        ])

    df[["sentiment_score", "sentiment_intensity"]] = (
        df["review_text"].apply(sentiment_scores)
    )

    # -------------------------------------------------
    # 3️⃣ Repetition Score (Lexical Diversity)
    # -------------------------------------------------

    def repetition_score(text):
        words = str(text).lower().split()
        if len(words) == 0:
            return 0
        return 1 - (len(set(words)) / len(words))

    df["repetition_score"] = df["review_text"].apply(repetition_score)

    # -------------------------------------------------
    # 4️⃣ TF-IDF (For Promotional Detection + ML Input)
    # -------------------------------------------------

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=300,
        ngram_range=(1, 2)
    )

    tfidf_matrix = vectorizer.fit_transform(df["review_text"])
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=vectorizer.get_feature_names_out()
    )

    # Promotional score:
    # Detect unusually high TF-IDF weight concentration
    df["promotional_score"] = tfidf_matrix.max(axis=1).toarray().flatten()

    # -------------------------------------------------
    # 5️⃣ Product Detail Score (Semantic Similarity)
    # -------------------------------------------------

    model = SentenceTransformer("all-MiniLM-L6-v2")

    df = df.merge(
        products_df[["product_id", "name", "category", "brand"]],
        on="product_id",
        how="left"
    )

    product_text = (
        df["name"].fillna("") + " " +
        df["category"].fillna("") + " " +
        df["brand"].fillna("")
    )

    review_embeddings = model.encode(df["review_text"].tolist())
    product_embeddings = model.encode(product_text.tolist())

    similarities = cosine_similarity(review_embeddings, product_embeddings)
    df["product_detail_score"] = similarities.diagonal()

    # -------------------------------------------------
    # 6️⃣ Rating–Sentiment Mismatch
    # -------------------------------------------------

    def mismatch(row):
        rating = row["rating"]
        sentiment = row["sentiment_score"]

        if rating >= 4 and sentiment < -0.2:
            return 1
        elif rating <= 2 and sentiment > 0.2:
            return 1
        return 0

    df["rating_sentiment_mismatch"] = df.apply(mismatch, axis=1)

    # -------------------------------------------------
    # 7️⃣ Create Label (For Training)
    # -------------------------------------------------
    # For now, pseudo-label:
    # High exaggeration + low product similarity + mismatch

    df["label"] = (
        (
            (df["sentiment_intensity"] > 0.8) &
            (df["product_detail_score"] < 0.3)
        ) |
        (df["rating_sentiment_mismatch"] == 1)
    ).astype(int)

    # -------------------------------------------------
    # Final Feature Set
    # -------------------------------------------------

    final_df = df[
        [
            "review_id",
            "sentiment_score",
            "sentiment_intensity",
            "product_detail_score",
            "length",
            "capital_ratio",
            "punctuation_density",
            "structural_score",
            "repetition_score",
            "promotional_score",
            "rating_sentiment_mismatch",
            "label"
        ]
    ].copy()

    # Rename columns professionally
    final_df = final_df.rename(columns={
        "product_detail_score": "product_relevance_score",
        "length": "review_length",
        "promotional_score": "promotional_intensity"
    })

    return final_df, vectorizer