# Design Decisions

---

**Decision:** Implement logistic regression from scratch (standardization +
gradient descent + L2 regularization, in plain Java, zero dependencies)
rather than pulling in a Java ML library such as Smile or Tribuo.

**Why:** For a 7-feature model, hand-rolled logistic regression is well
within "cleanly implementable in Java" — the master brief for this
collection's default position. It also means every line that turns a
customer's features into a probability is auditable in this repository,
with no library version, API surface or transitive dependency to track.
`explain()` — the per-prediction feature-contribution breakdown — falls out
directly from having written the math, rather than needing to reverse-engineer
an opaque library model.

**Trade-off:** No access to more sophisticated model families (gradient-boosted
trees, etc.) or a library's built-in cross-validation tooling.

**Future:** If a future project in this collection needs a materially more
complex model, that's the point where evaluating a Java ML library becomes
worth the dependency — not before.

---

**Decision:** Use class weighting during training (`positiveClassWeight` in
weights.json) rather than shipping the unweighted model.

**Why, with real numbers:** The dataset is ~14% churned. An unweighted model
reached 91.7% test accuracy but only 37.5% recall (catching 9 of 24 actual
churners in the test split) — because predicting "no churn" for almost
everyone is already a low-error strategy on imbalanced data. Weighting the
minority class during training (weight ≈ negatives/positives ≈ 6.35)
changed that to 73.3% accuracy but 91.7% recall (22 of 24 churners caught,
2 missed) and 32.4% precision (46 false positives in 180 test rows).

**Trade-off, stated plainly:** The shipped model's raw accuracy (73.3%) is
*below* the do-nothing baseline of always predicting "no churn" (86.7%).
Accuracy is the wrong headline metric for this problem; recall is not.
Whether this trade-off is the right one depends entirely on the business
cost of a missed churner versus a false alarm — this project ships the
high-recall version because a churn-prediction tool that misses most
churners has limited practical use, but both the numbers and the trade-off
are documented in full in `docs/model-training.md`, not smoothed over.

**Future:** Expose the classification threshold (not just the
low/medium/high tier ceilings) via the API, so a consumer can pick their own
point on the precision/recall curve instead of inheriting this project's choice.

---

**Decision:** Report `is_subscribed`'s learned weight (-0.0091, nearly zero)
honestly rather than omitting it or re-running training until it looked
more decisive.

**Why:** The synthetic data-generating process gave subscription status a
real (negative) effect on churn, but the trained model recovered only a
very weak signal for it — likely diluted by correlation with other features
and the relatively small number of positive (churned) training examples.
That's a real, reportable ML result, and hiding an unflattering number would
undermine every other number in this document.

**Trade-off:** None — this is a transparency choice, not a modeling one.

**Future:** More training data, or an interaction term, might recover a
stronger subscription-status signal; worth revisiting if this model is ever
retrained on a larger dataset.
