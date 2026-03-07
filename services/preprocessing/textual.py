import pandas as pd
import numpy as np
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer

def textual_training_dataset(reviews_df, products_df):
    df = reviews_df.copy()
    df["review_text"] = df["review_text"].fillna("").astype(str)

    # -------------------------------------------------
    # 1️⃣ BASIC STRUCTURAL FEATURES
    # -------------------------------------------------

    df["review_length"] = df["review_text"].apply(lambda x: len(x.split()))

    def capital_ratio(text):
        if len(text) == 0:
            return 0
        return sum(c.isupper() for c in text) / len(text)

    df["capital_ratio"] = df["review_text"].apply(capital_ratio)

    def punctuation_density(text):
        return len(re.findall(r"[!?,.;:]", text)) / (len(text) + 1)

    df["punctuation_density"] = df["review_text"].apply(punctuation_density)

    # -------------------------------------------------
    # 2️⃣ ADDITIONAL FRAUD TEXT FEATURES (NEW)
    # -------------------------------------------------

    # ALL CAPS detection
    df["all_caps_flag"] = (df["capital_ratio"] > 0.6).astype(int)

    # Generic short review detection
    df["generic_short_review"] = (df["review_length"] <= 3).astype(int)

    # Promotional phrase detection
    promo_phrases = [
        "must buy",
        "highly recommend",
        "best product",
        "life changing",
        "perfect product",
        "amazing product",
        "worth every penny",
        "buy this",
        "excellent product"
    ]

    def promo_phrase_flag(text):
        text = text.lower()
        return int(any(p in text for p in promo_phrases))

    df["promo_phrase_flag"] = df["review_text"].apply(promo_phrase_flag)

    # Structural indicator score
    df["structural_score"] = (
        df["capital_ratio"] * 0.3 +
        df["punctuation_density"] * 0.3 +
        df["generic_short_review"] * 0.4
    )

    # -------------------------------------------------
    # 3️⃣ SENTIMENT SCORE (VADER)
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
    # 4️⃣ REPETITION SCORE
    # -------------------------------------------------

    def repetition_score(text):
        words = text.lower().split()
        if len(words) == 0:
            return 0
        return 1 - (len(set(words)) / len(words))

    df["repetition_score"] = df["review_text"].apply(repetition_score)

    # -------------------------------------------------
    # 5️⃣ TF-IDF PROMOTIONAL INTENSITY
    # -------------------------------------------------

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=300,
        ngram_range=(1, 2)
    )

    tfidf_matrix = vectorizer.fit_transform(df["review_text"])

    df["promotional_intensity"] = (
        tfidf_matrix.max(axis=1).toarray().flatten()
    )

    # -------------------------------------------------
    # 6️⃣ PRODUCT RELEVANCE (SEMANTIC SIMILARITY)
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

    df["product_relevance_score"] = similarities.diagonal()

    # -------------------------------------------------
    # 7️⃣ RATING SENTIMENT MISMATCH
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
    # 8️⃣ IMPROVED PSEUDO LABEL
    # -------------------------------------------------

    df["label"] = (
        (
            (df["sentiment_intensity"] > 0.8) &
            (df["product_relevance_score"] < 0.35)
        )
        |
        (df["promo_phrase_flag"] == 1)
        |
        (df["all_caps_flag"] == 1)
        |
        (df["rating_sentiment_mismatch"] == 1)
    ).astype(int)

    # -------------------------------------------------
    # FINAL FEATURE SET
    # -------------------------------------------------

    final_df = df[
        [
            "review_id",
            "sentiment_score",
            "sentiment_intensity",
            "product_relevance_score",
            "review_length",
            "capital_ratio",
            "punctuation_density",
            "structural_score",
            "repetition_score",
            "promotional_intensity",
            "promo_phrase_flag",
            "all_caps_flag",
            "generic_short_review",
            "rating_sentiment_mismatch",
            "label"
        ]
    ].copy()

    return final_df, vectorizer
