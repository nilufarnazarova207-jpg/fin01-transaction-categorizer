# FIN-01 — Automatic Transaction Categorization

**Capstone project — FinTech track**

Assigns a spending category to a raw bank transaction description (e.g.
`"CHECK CRD PURCHASE 11/11 STARBUCKS #4521 SEATTLE WA"` → **Retail Trade**),
addressing the brief's business problem: manual and rule-based
categorization of messy, inconsistent merchant text doesn't scale for a
digital banking application.

**[Try the live interactive app →](app/ledger.html)** (open directly in a
browser — no server, no install; the model runs entirely client-side in
JavaScript)

---

## Overview

- **Dataset:** [Wells Fargo Campus Analytics Challenge](https://github.com/utribedi/Bank_transaction_category_predictor)
  — 40,000 real (synthetically-scrubbed) bank transactions, each labeled
  with one of 10 spending categories.
- **Approach:** TF-IDF (word + bigram features) + multinomial Logistic
  Regression, with class-balanced training to address severe category
  imbalance (as low as 130 examples for the smallest original category vs.
  ~13,500 for the largest).
- **Key design decisions:**
  - Merged the three smallest/least-reliable categories into a single
    "Other Services" bucket, improving macro F1 from ~0.50 to ~0.58.
  - Added a confidence-based fallback: predictions below 40% confidence
    are flagged "Needs Review" rather than guessed, trading ~19% coverage
    for a jump from 66% to 75% accuracy on the transactions the model
    does commit to.
  - See [`docs/limitations_and_risks.md`](docs/limitations_and_risks.md)
    for the full reasoning, trade-off tables, and next steps.

## Results

| Metric | Validation | Test (held-out) |
|---|---|---|
| Accuracy | ~66% | ~66% |
| Macro F1 | ~0.58 | ~0.57–0.58 |
| Weighted F1 | ~0.68 | ~0.67 |

Full per-category precision/recall/F1 is printed by `train.py` and discussed
in the limitations doc.

## Repository structure

```
fin01-transaction-categorizer/
├── train.py                       # Reproducible end-to-end training pipeline
├── requirements.txt
├── model/
│   ├── vectorizer_v2.joblib        # Fitted TF-IDF vectorizer (Python reuse)
│   ├── clf_v2.joblib                # Fitted Logistic Regression model (Python reuse)
│   ├── model_export.json           # Same model, exported as plain JSON for browser inference
│   └── preprocess.py               # Shared text-cleaning function
├── app/
│   └── ledger.html                  # Self-contained interactive web app (model embedded, runs in-browser)
├── docs/
│   └── limitations_and_risks.md    # Full write-up: design decisions, limitations, next steps
└── README.md
```

## Running it yourself

```bash
git clone <this-repo-url>
cd fin01-transaction-categorizer
pip install -r requirements.txt
python train.py
```

This downloads the dataset, trains the model from scratch, prints full
evaluation metrics (including the confidence-threshold analysis), and
re-saves all model artifacts into `model/`.

To use the trained model in Python:

```python
import joblib, sys
sys.path.insert(0, "model")
from preprocess import clean_text

vectorizer = joblib.load("model/vectorizer_v2.joblib")
clf = joblib.load("model/clf_v2.joblib")

text = "CHECK CRD PURCHASE 11/11 SHELL OIL 57492013 HOUSTON TX"
X = vectorizer.transform([clean_text(text)])
proba = clf.predict_proba(X)[0]
pred = clf.classes_[proba.argmax()]
print(pred, proba.max())
```

To use it interactively: just open `app/ledger.html` in any browser. It has
a single-transaction mode and a batch/statement mode (paste many
transactions, get a categorized table plus a spending breakdown).

## Functional requirements checklist (per brief §7)

- [x] Accepts a new transaction record, returns a spending category
- [x] Clearly defined, user-facing-appropriate category taxonomy (8 categories)
- [x] Handles previously unseen merchant descriptions (TF-IDF generalizes to
      shared vocabulary; genuinely novel merchants fall back to "Needs Review")
- [x] Ambiguous/low-confidence cases handled via the 40% confidence threshold
- [x] No real credentials, card numbers, or PII exposed (dataset is
      pre-scrubbed by the original challenge organizers; app requires no
      account data, only free-text description)
