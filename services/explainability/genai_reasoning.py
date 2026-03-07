import requests


def generate_llm_explanation(review_text, risk_score, top_features, prediction_label, evidence):

    try:

        prompt = f"""
You must generate an explanation for a fraud review detection system.

STRICT OUTPUT FORMAT (YOU MUST FOLLOW THIS EXACT FORMAT):

The Review : "<review text>"

• reason
• reason
• reason

Rules:

1. The first line MUST start exactly with:
   The Review : "<review text>"

2. Then provide bullet points using the symbol "•".

3. Write 2–4 bullet points only.

4. If prediction is Fraudulent or Suspicious:
   explain ONLY the issues in the review.
   explain the top contributing features as 2/3 points.
   if rating sentiment mismatch is a top feature, display the rating value and words in the review justifying sentiment score.
   if not a verified purchase, mention it as a point and briefly explain.
   if refunded product and (rating is high or review  mention as a point.


5. If prediction is Genuine:
   explain ONLY why the review appears trustworthy in 1 or 2 points.

6. Each point MUST use the evidence provided.

7. Do NOT add headings.
8. Do NOT add paragraphs.
9. Do NOT add extra explanations.
10. Do NOT write "analysis", "prediction", or "evidence".
11. Do NOT speculate.

Evidence:

Rating: {evidence["rating"]}
Verified Purchase: {evidence["verified_purchase"]}
Refunded Product: {evidence["refunded_product"]}
Sentiment Score: {evidence["sentiment_score"]}
Sentiment Intensity: {evidence["sentiment_intensity"]}
Product Relevance Score: {evidence["product_relevance"]}
Rating Sentiment Mismatch: {evidence["rating_sentiment_mismatch"]}
Punctuation Density: {evidence["punctuation_density"]}
Review Length: {evidence["review_length"]}

Review Text:
{review_text}

Prediction:
{prediction_label}
"""
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "phi3",
                "prompt": prompt,
                "stream": False
            }
        )

        return response.json()["response"]

    except Exception as e:
        print("LOCAL LLM ERROR:", e)
        return None