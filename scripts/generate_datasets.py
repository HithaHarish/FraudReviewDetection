"""
Generate synthetic datasets for the Review Fraud Detection UI.
Creates: 200 reviews, 80 customers, 3 platforms, 160 orders, 30 products

Review text is designed to align with textual analysis features:
- Structural: capital_ratio, punctuation_density, length < 5
- Sentiment: VADER compound & intensity
- Repetition: lexical diversity
- Promotional: TF-IDF concentration (generic hype words)
- Product detail: semantic similarity (product-specific vs generic)
- Rating-sentiment mismatch: 5 stars + negative text, or 1 star + positive text
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Output directory
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- FRAUD-LIKE reviews (high structural, promotional, low product detail) ---
# Designed to trigger: capital_ratio, punctuation_density, structural_score,
# repetition_score, promotional_score, low product_detail_score, sentiment_intensity
FRAUD_TEMPLATES = [
    # ALL CAPS + excessive punctuation + promotional keywords
    "AMAZING PRODUCT LIFE CHANGING EXPERIENCE MUST BUY IT NOW !!!!!",
    "BEST PURCHASE EVER!!! INCREDIBLE!!! FIVE STARS!!! BUY NOW!!!",
    "PERFECT!!! AMAZING!!! LOVE IT!!! MUST HAVE!!! !!!!!",
    "INCREDIBLE!!! BEST!!! FANTASTIC!!! RECOMMEND!!! ???",
    "WOW!!! AMAZING!!! GAME CHANGER!!! GET IT NOW!!!",
    "BEST BEST BEST!!! AMAZING!!! MUST BUY!!! !!! ??? ...",
    # Short (<5 words) + all caps + high punctuation
    "AMAZING MUST BUY",
    "BEST EVER !!!!!",
    "PERFECT FIVE STARS!!!",
    "INCREDIBLE!!! BUY!!!",
    "WOW!!! BUY!!!",
    # Repetitive (high repetition_score)
    "best best best amazing amazing amazing love love love it it it",
    "great great great product product product buy buy buy now now now",
    "amazing amazing amazing must must must buy buy buy",
    "love love love it it it best best best",
    # Exaggerated sentiment + generic (low product_detail_score)
    "Absolutely INCREDIBLE!!! The BEST thing I have EVER bought!!! Life changing!!!",
    "WOW!!! This is PERFECT!!! Everyone needs this!!! BUY NOW!!!",
    "OMG!!! BEST!!! EVER!!! Changed my LIFE!!! !!!!!",
    "INCREDIBLE!!! PERFECT!!! AMAZING!!! No words!!! !!!",
]

# --- LEGITIMATE reviews (product-specific, natural structure) ---
# These reference product attributes for higher product_detail_score
LEGIT_TEMPLATES = [
    "The {name} from {brand} works well. Good {category} item for the price.",
    "Received the {name} quickly. The {brand} {category} quality is solid.",
    "Decent {category} product. The {name} does what it says.",
    "The {brand} {name} is okay. Expected more for a {category} item.",
    "Good {category} choice. The {name} fits my needs.",
    "The {name} arrived on time. As a {category} product it meets expectations.",
    "Solid {brand} {category} item. The {name} performs adequately.",
    "The {name} is a reasonable {category} option. {brand} quality is fine.",
    "Works as described. The {name} from {brand} is a decent {category} product.",
    "The {category} {name} is functional. No major issues with the {brand} build.",
]

# --- MIXED (some fraud indicators, some legit) ---
MIXED_TEMPLATES = [
    "Great product! The {name} is exactly what I needed for {category}.",
    "Amazing! The {brand} {name} exceeded expectations. Good {category}.",
    "Love the {name}! Best {category} I've tried from {brand}.",
    "The {name} is fantastic. Highly recommend this {brand} {category}.",
]

PRODUCT_CATEGORIES = [
    "Electronics", "Clothing", "Home & Garden", "Sports", "Beauty",
    "Books", "Toys", "Health", "Automotive", "Kitchen"
]

BRANDS = [
    "TechPro", "StyleCo", "HomeEssentials", "SportMax", "BeautyGlow",
    "ReadWell", "PlayFun", "HealthFirst", "AutoParts", "KitchenKing",
    "Generic", "Premium", "Budget", "EcoFriendly", "ProSeries"
]

# Descriptive product names for semantic similarity (product_detail_score)
PRODUCT_NAMES = [
    "Fitness Tracker Pro", "Wireless Earbuds", "Garden Hose 50ft",
    "Yoga Mat Premium", "LED Desk Lamp", "Running Shoes",
    "Bluetooth Speaker", "Kitchen Knife Set", "Vitamin D Supplement",
    "Children's Puzzle Set", "Organic Face Cream", "Car Phone Mount",
    "Coffee Maker", "Camping Tent 4-Person", "Resistance Bands",
    "Desk Organizer", "Electric Toothbrush", "Baby Stroller",
    "Hiking Backpack", "Smart Watch", "Air Fryer",
    "Reading Lamp", "Board Game Family", "Protein Powder",
    "Wireless Mouse", "Plant Pot Set", "Soccer Ball",
    "Cooking Utensils", "Sleep Mask", "Water Bottle Insulated",
    "Handheld Vacuum",
]

PLATFORM_NAMES = ["Amazon", "eBay", "Walmart"]


def generate_platforms(n=3):
    """Generate 3 platforms."""
    return pd.DataFrame({
        "platform_id": range(1, n + 1),
        "platform_name": PLATFORM_NAMES[:n],
    })


def generate_customers(n=80):
    """Generate 80 customers with account_created dates."""
    base_date = datetime(2020, 1, 1)
    return pd.DataFrame({
        "customer_id": [f"C{i:04d}" for i in range(1, n + 1)],
        "account_created": [
            (base_date + timedelta(days=random.randint(0, 1000))).strftime("%Y-%m-%d")
            for _ in range(n)
        ],
    })


def generate_products(n=30):
    """Generate 30 products with descriptive names for product_detail_score."""
    names = (PRODUCT_NAMES * 2)[:n]  # Cycle if needed
    return pd.DataFrame({
        "product_id": [f"P{i:04d}" for i in range(1, n + 1)],
        "name": names,
        "category": random.choices(PRODUCT_CATEGORIES, k=n),
        "brand": random.choices(BRANDS, k=n),
    })


def generate_orders(n=160, n_customers=80, n_products=30):
    """Generate 160 orders linking customers to products."""
    base_date = datetime(2023, 1, 1)
    orders = []
    for i in range(1, n + 1):
        orders.append({
            "order_id": f"ORD{i:05d}",
            "customer_id": f"C{random.randint(1, n_customers):04d}",
            "product_id": f"P{random.randint(1, n_products):04d}",
            "order_date": (base_date + timedelta(days=random.randint(0, 400))).strftime("%Y-%m-%d %H:%M:%S"),
            "amount": round(random.uniform(9.99, 299.99), 2),
        })
    return pd.DataFrame(orders)


def generate_reviews(n=200, n_customers=80, n_products=30, n_platforms=3, products_df=None):
    """Generate 200 reviews with text aligned to textual analysis features."""
    base_date = datetime(2023, 6, 1)
    reviews = []

    # Build product lookup for legit/mixed templates
    product_lookup = {}
    if products_df is not None:
        for _, row in products_df.iterrows():
            product_lookup[row["product_id"]] = {
                "name": row["name"],
                "category": row["category"],
                "brand": row["brand"],
            }

    # Rating-sentiment mismatch pairs: (negative text, rating 5) or (positive text, rating 1)
    MISMATCH_NEGATIVE = [
        "Terrible. Worst purchase. Do not buy. Horrible experience.",
        "Disappointing. Waste of money. Regret buying.",
        "Awful product. Complete waste. Returned immediately.",
    ]
    MISMATCH_POSITIVE = [
        "Amazing product! Best purchase ever! Love it!",
        "Great quality! Exactly what I needed! Perfect!",
        "Fantastic! Highly recommend! Five stars!",
    ]

    for i in range(1, n + 1):
        customer_id = f"C{random.randint(1, n_customers):04d}"
        product_id = f"P{random.randint(1, n_products):04d}"
        platform_id = random.randint(1, n_platforms)
        prod_info = product_lookup.get(product_id, {"name": "Product", "category": "item", "brand": "Brand"})

        # ~40% fraud-like, ~45% legit, ~15% mixed (aligns with textual layer flags)
        r = random.random()
        is_fraud = 0
        if r < 0.40:
            is_fraud = 1
            # 15% of the time, fraud uses legitimate text to simulate "sophisticated" fake reviews
            if random.random() < 0.15:
                template = random.choice(LEGIT_TEMPLATES).format(**prod_info)
                rating = 5
            else:
                template = random.choice(FRAUD_TEMPLATES)
                rating = random.choices([1, 2, 3, 4, 5], weights=[0.05, 0.1, 0.15, 0.35, 0.35])[0]
        elif r < 0.55:
            is_fraud = 1
            if random.random() < 0.5:
                template = random.choice(MISMATCH_NEGATIVE)
                rating = 5
            else:
                template = random.choice(MISMATCH_POSITIVE)
                rating = 1
        elif r < 0.85:
            template = random.choice(LEGIT_TEMPLATES).format(**prod_info)
            rating = random.choices([1, 2, 3, 4, 5], weights=[0.05, 0.1, 0.25, 0.4, 0.2])[0]
            # 5% chance of legitimate review having a rating mismatch (user error/noise)
            if random.random() < 0.05:
                if rating > 3:
                    template = random.choice(MISMATCH_NEGATIVE)
                else:
                    template = random.choice(MISMATCH_POSITIVE)
        else:
            template = random.choice(MIXED_TEMPLATES).format(**prod_info)
            rating = random.choices([3, 4, 5], weights=[0.2, 0.5, 0.3])[0]
            # Sometimes mixed templates are actually fraud
            if random.random() < 0.3:
                is_fraud = 1

        # Introduce 15% pure random label noise to prevent 100% ROC-AUC
        if random.random() < 0.15:
            is_fraud = 1 - is_fraud

        verified = random.random() < 0.7
        refunded = random.random() < 0.1

        review_time = base_date + timedelta(
            days=random.randint(0, 200),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59),
        )

        reviews.append({
            "review_id": f"R{i:05d}",
            "customer_id": customer_id,
            "product_id": product_id,
            "platform_id": platform_id,
            "review_text": template,
            "review_timestamp": review_time.strftime("%Y-%m-%d %H:%M:%S"),
            "rating": rating,
            "verified_purchase": verified,
            "refunded_product": refunded,
            "is_fraud": is_fraud,
        })

    return pd.DataFrame(reviews)


def main():
    print("Generating datasets...")

    platforms_df = generate_platforms(3)
    customers_df = generate_customers(200)
    products_df = generate_products(50)
    orders_df = generate_orders(400, 200, 50)
    reviews_df = generate_reviews(1000, 200, 50, 3, products_df)

    # Save to CSV
    platforms_df.to_csv(os.path.join(OUTPUT_DIR, "platforms.csv"), index=False)
    customers_df.to_csv(os.path.join(OUTPUT_DIR, "customers.csv"), index=False)
    products_df.to_csv(os.path.join(OUTPUT_DIR, "products.csv"), index=False)
    orders_df.to_csv(os.path.join(OUTPUT_DIR, "orders.csv"), index=False)
    reviews_df.to_csv(os.path.join(OUTPUT_DIR, "reviews.csv"), index=False)

    print(f"Saved to {OUTPUT_DIR}:")
    print(f"  - platforms.csv: {len(platforms_df)} rows")
    print(f"  - customers.csv: {len(customers_df)} rows (upload as Users Dataset)")
    print(f"  - products.csv: {len(products_df)} rows")
    print(f"  - orders.csv: {len(orders_df)} rows")
    print(f"  - reviews.csv: {len(reviews_df)} rows")
    print("Done!")


if __name__ == "__main__":
    main()
