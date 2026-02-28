# Review Fraud Detection System

This project implements a **multi-layer review fraud detection system** with:

- **Textual layer** (50% weight)
- **Behavioral layer** (30% weight)
- **Temporal layer** (20% weight)

It integrates customer, product, and review datasets, computes a **Final Fraud Risk Score (0–100)**, aggregates scores at the product level, and exposes an **interactive Streamlit dashboard**.

---

## 1. Project Structure

The expected structure is:

```text
fraud_detection_project/
│
├── data/
│   ├── customers.csv                      # or customer_preprocessed_only_200_rows.csv
│   ├── products.csv                       # or product_preprocessed_only_200_rows.csv
│   └── reviews.xlsx                       # or textual_temporal_200_rows.csv.xlsx
│
├── models/
│   └── fraud_model.py
│
├── utils/
│   ├── textual_layer.py
│   ├── behavioral_layer.py
│   ├── temporal_layer.py
│
├── app.py
├── setup_env.py
├── requirements.txt
└── README.md
```

> **Note**: If you prefer to keep your original filenames, just copy them into the `data` folder. The code will automatically try both the generic names (`customers.csv`, `products.csv`, `reviews.xlsx`) and the original filenames (`customer_preprocessed_only_200_rows.csv`, `product_preprocessed_only_200_rows.csv`, `textual_temporal_200_rows.csv.xlsx`).

---

## 2. Environment Setup (venv + Auto Install)

From the `fraud_detection_project` folder:

### 2.1 Create virtual environment `venv`

```bash
python -m venv venv
```

On **Windows (PowerShell)**:

```bash
.\venv\Scripts\Activate
```

On **macOS / Linux**:

```bash
source venv/bin/activate
```

### 2.2 Install dependencies from `requirements.txt`

You can either install manually:

```bash
pip install -r requirements.txt
```

or let the **automation script** do everything for you (recommended):

```bash
python setup_env.py
```

The `setup_env.py` script will:

- Create the `venv` virtual environment (if it doesn’t exist)
- Install all dependencies from `requirements.txt` **inside `venv`**
- Download NLTK data:
  - `punkt`
  - `stopwords`
- Download spaCy model:
  - `en_core_web_sm`

---

## 3. Expected Data Files

Place your files in the `data` folder. The loader will automatically try multiple naming options:

- **Customers**
  - `data/customers.csv`
  - `data/customer_preprocessed_only_200_rows.csv`
- **Products**
  - `data/products.csv`
  - `data/product_preprocessed_only_200_rows.csv`
- **Reviews**
  - `data/reviews.xlsx`
  - `data/textual_temporal_200_rows.csv.xlsx`

Required logical columns (names can vary slightly; see code for fallbacks/aliases):

- `Customer_ID`
- `Product_ID`
- `Review_ID` (or equivalent)
- `Review_Text`
- Optional: `Text_Fraud_Probability`
- Behavioral:
  - `Account_Age`
  - `Review_Frequency`
  - `Refund_Ratio`
  - `Verified_Purchase_Ratio`
  - `Average_Rating_By_User`
- Temporal:
  - `Reviews_Per_Day`
  - `Burst_Flag`
  - `Review_Date`

---

## 4. Running the Streamlit Dashboard

After the environment is ready and data files are in `data/`:

```bash
streamlit run app.py
```

The dashboard provides:

- Dropdown selection by `Review_ID`
- Display of:
  - Review text
  - Textual score
  - Behavioral score
  - Temporal score
  - Final weighted fraud risk score
  - Risk level (Low / Moderate / High)
- Product-level metrics:
  - Average fraud score per product
  - Suspicious review ratio per product
  - Product authenticity score
- Basic charts:
  - Distribution of fraud scores
  - Per-product fraud metrics

---

## 5. Reproducibility Notes

- Random operations (e.g. train/test splits, model initialization) use a **fixed random seed** for reproducibility.
- All key hyperparameters and column names are defined in one place in the code and can be adjusted easily.

