# Limitations, Risks & Next Steps

## Model performance

| Metric | Value |
|---|---|
| Test accuracy | ~66% |
| Test macro F1 | ~0.57–0.58 |
| Test weighted F1 | ~0.67 |

Performance is strong on high-frequency categories (Entertainment, Health and
Community Services — both F1 ≈ 0.78–0.79) and materially weaker on
lower-frequency ones (Other Services, Property and Business Services — both
F1 ≈ 0.28–0.32), which is expected given the underlying class imbalance in
the training data (as few as ~130 examples for some original categories,
versus ~9,000+ for Retail Trade).

## Design decisions and why

**Category taxonomy.** The original dataset ships with 10 categories. Three
of them (Communication Services, Finance, Education) had fewer than 300
training examples each and, when kept separate, produced very poor precision
(as low as 0.12–0.17) despite passable recall — the class-balanced training
weighting made the model over-eager to predict them, flooding those labels
with false positives from larger classes. We merged these three into a
single "Other Services" bucket. This is a real taxonomy trade-off: the
merged bucket is less specific, but far more reliable, and matches how a
digital banking app would likely want to surface an uncommon-but-genuine
spend category rather than mislabel it as something else entirely.

**Confidence-based fallback.** Rather than always returning a top-1 label,
the interface flags any prediction below 40% confidence as "Needs Review"
instead of guessing. This threshold was chosen from an explicit
coverage/accuracy trade-off analysis (see `train.py` output):

| Confidence threshold | % of transactions auto-categorized | Accuracy on those |
|---|---|---|
| 0.0 (no fallback) | 100% | ~66% |
| 0.3 | ~94% | ~68% |
| **0.4 (chosen)** | **~81%** | **~75%** |
| 0.5 | ~68% | ~80% |
| 0.6 | ~59% | ~86% |

0.4 was chosen as a reasonable middle ground: it keeps a large majority of
transactions fully automated while meaningfully raising accuracy on the ones
it does commit to, and routes the harder ~19% to a human/analyst rather than
risking a wrong label reaching the user. A production deployment should
tune this threshold against the actual cost of misclassification vs. the
cost of manual review in that specific business context.

**Text preprocessing.** Raw descriptions contain heavy boilerplate (e.g.
`"CHECK CRD PURCHASE 11/11 ... 111-111-1111 ... ?MCC=1111"`) — placeholder
dates, masked card digits, and scrubbed phone/ID codes that carry no
category signal and were explicitly stripped before vectorizing. The
`coalesced_brand` field (present for all rows) was concatenated with the raw
description, since it sometimes carries a cleaner merchant name.

## Known limitations

- **Rare category reliability.** Even after merging, "Other Services" and
  "Property and Business Services" remain the weakest predicted categories
  (F1 ≈ 0.3). Any use case that depends on those specific categories being
  accurate should treat model output there as a weak prior, not a
  determination.
- **English, US-centric merchant text.** The dataset consists of US bank
  transactions with US merchant naming conventions. Performance on
  non-US merchant formats or non-English descriptions is unknown and
  likely much worse.
- **Synthetic training data.** Per the dataset's own documentation, the
  underlying data was scrubbed/synthesized by the original challenge
  organizers (transaction numbers replaced), which may not perfectly
  reflect the noise patterns of a live production feed.
- **No amount or merchant-category-code (MCC) signal used.** The brief's
  "available information" included transaction amount and
  merchant category code; `merchant_cat_code` was ~38% missing in this
  dataset and was excluded to keep the pipeline simple and avoid an
  imputation strategy that could leak label information. This is a natural
  next step (see below) but was out of scope for this iteration.
- **No true out-of-vocabulary handling beyond TF-IDF's default behavior.**
  Previously unseen merchants are handled reasonably well when their
  descriptions share common word patterns with training data (e.g. city
  names, common suffixes), but a completely novel merchant with no
  overlapping vocabulary will fall back on weaker signal and is more likely
  to land in the "Needs Review" bucket — which is the intended, safe
  behavior, not a failure mode.

## Recommended next steps

1. **Incorporate transaction amount as a feature.** Certain categories
   (e.g. Travel, Property and Business Services) likely have distinct
   amount distributions that could meaningfully improve rare-category
   precision.
2. **Active-learning loop on "Needs Review" cases.** Route low-confidence
   predictions to analyst review and feed corrected labels back into
   retraining — this directly targets the weakest part of the current
   model (rare/ambiguous categories) with real labeled data over time.
3. **Evaluate a transformer-based encoder** (e.g. a small BERT variant) as
   an optional upgrade path per the brief's "embedding/transformer
   comparison" direction, trading inference simplicity for potentially
   better handling of abbreviations and novel merchant phrasing.
4. **Merchant-level caching.** Many merchants recur across many
   transactions; a lookup table for high-confidence, frequently-seen
   merchants (updated from model predictions over time) could reduce
   both latency and error rate in production, similar to the two-round
   approach described in prior public implementations of this problem.
