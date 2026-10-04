# Customer Churn Prediction API

A Java API serving a **from-scratch logistic regression model**, trained
offline on synthetic data, with per-prediction **feature-level
explainability** — not a black box, and not a call to someone else's ML
service.

Part of the [Commerce Engineering Lab](https://ritangkar.github.io/#commerce-lab) —
17 small, production-minded engineering capabilities by Ritangkar Dey.

---

## 1. Project Overview

Send customer features; get back a churn probability, a risk tier, and the
three features that most influenced that specific prediction — each with a
direction (increases/decreases risk) and a magnitude.

## 2. Problem

Most "churn prediction" demos either fake the model or import a heavyweight
ML library whose internals the demo doesn't actually explain. Neither
demonstrates engineering — one demonstrates nothing, the other demonstrates
someone else's library.

## 3. Solution

`Model → Java API → structured prediction → explainability`, literally:

1. **Offline**: `/tools/ChurnTrainer.java` trains a logistic regression model
   via gradient descent (with class weighting for the imbalanced label) on
   `/sample-data/customers.csv`, evaluates it on a held-out test split, and
   writes the learned weights + real metrics to `weights.json`.
2. **Online**: the Spring Boot service only ever *loads* `weights.json` at
   startup. It never retrains. `ChurnModel.explain()` decomposes every
   prediction into its per-feature contributions.

## 4. Key Features

- Real logistic regression, implemented from scratch (no ML library) —
  standardization, gradient descent, L2 regularization, class weighting
- Explainability: every prediction returns its top 3 contributing factors,
  signed and ranked by magnitude
- `GET /api/churn/model-info` exposes the model's actual training metadata —
  algorithm, sample sizes, accuracy, precision, recall, F1, confusion
  matrix — sourced directly from the training run, not restated by hand
- Configurable risk-tier thresholds (`LOW` / `MEDIUM` / `HIGH`)
- Full input validation on every feature

## 5. Architecture

```
customers.csv (synthetic, labeled)
        │  (offline, one-off)
        ▼
ChurnTrainer.java  ──►  weights.json  (means, stds, weights, bias, trainingMeta)
        │  (committed as an artifact)
        ▼
ChurnModel  (loads once at startup)
   ├─ predictProbability(features)
   └─ explain(features) ──► ranked FeatureContribution list
        │
        ▼
ChurnPredictionService  ──► risk tier
        │
        ▼
ChurnController  ──►  POST /api/churn/predict
```

See [`docs/architecture.md`](docs/architecture.md) for the full diagram.

## 6. Technical Approach

- **Why hand-rolled logistic regression instead of a Java ML library**: see
  [`docs/design-decisions.md`](docs/design-decisions.md) — the short version
  is explainability and zero heavyweight dependencies for a 7-feature model.
- Features are standardized using the **training set's** mean/std (not the
  full dataset) to avoid leakage, and those exact values are shipped inside
  `weights.json` so the serving code standardizes new input identically.
- Class weighting (`positiveClassWeight` in `weights.json`) compensates for
  the ~14% churn base rate — see the honest precision/recall discussion below.

## 7. Design Decisions

See [`docs/design-decisions.md`](docs/design-decisions.md) and
[`docs/model-training.md`](docs/model-training.md) for the full, honest
write-up, including a real before/after comparison of training with and
without class weighting.

## 8. Sample Input

```
POST /api/churn/predict
{
  "tenureMonths": 2,
  "monthlySpend": 400.0,
  "supportTicketsLast90d": 5,
  "daysSinceLastOrder": 150,
  "discountUsageRate": 0.9,
  "npsScore": 1,
  "isSubscribed": false
}
```

## 9. Sample Output

Real output for the input above (verified against the committed model):

```json
{
  "churnProbability": 0.9871,
  "riskLevel": "HIGH",
  "topFactors": [
    { "feature": "support_tickets_last_90d", "rawValue": 5.0, "contribution": 1.0371, "direction": "INCREASES_RISK" },
    { "feature": "days_since_last_order", "rawValue": 150.0, "contribution": 1.0305, "direction": "INCREASES_RISK" },
    { "feature": "monthly_spend", "rawValue": 400.0, "contribution": 0.8935, "direction": "INCREASES_RISK" }
  ],
  "modelVersion": "logistic-regression-v1"
}
```

*(Real output, independently computed from the committed `weights.json` — see `docs/model-training.md`.)*

## 10. How to Run

```bash
git clone <your-repo-url>
cd customer-churn-prediction-api
mvn spring-boot:run
```

API available at `http://localhost:8084`. The model loads from the
committed `weights.json` — no training happens at startup.

To retrain (optional, requires only a JDK — no Maven, no network):

```bash
cd tools
java ChurnDataGen.java   # regenerates sample-data/customers.csv (optional)
java ChurnTrainer.java   # retrains, prints metrics, writes weights.json
```

## 11. How to Test

```bash
mvn test
```

Tests verify the loaded model's predictions on two independently-computed
reference profiles, the explainability ranking/direction logic, and the
risk-tier boundary behavior.

## 12. API Documentation

| Method | Path                     | Description                                      |
|--------|--------------------------|----------------------------------------------------|
| POST   | `/api/churn/predict`     | Body: `CustomerFeatures` → `ChurnPrediction`      |
| GET    | `/api/churn/model-info`  | The real training metadata for the loaded model    |

## 13. Portfolio Demo

**Demo level:** 1 — client-side, deterministic. Because the model is just a
dot product and a sigmoid, `lab/demos/customer-churn-prediction-api.js`
ports the exact learned weights (copied from the committed `weights.json`,
not re-trained in the browser) into vanilla JavaScript, so moving the
sliders recomputes a real prediction from the real model instantly, with no
backend call. See `docs/demo.md`.

## 14. Limitations

- Trained on 900 synthetic rows — real customer data would have different,
  probably stronger, feature relationships
- At the default 0.5 classification threshold and with class weighting,
  the model trades precision for recall (catches 22/24 churners in the test
  set, at the cost of 46 false positives) — see `docs/model-training.md` for
  the full discussion and the un-weighted comparison
- `is_subscribed` came out with a much weaker learned weight than its
  true generative coefficient — documented, not hidden, in `docs/model-training.md`

## 15. Future Enhancements

- Expose the configurable classification threshold via the API itself
- A `/api/churn/batch-predict` endpoint for scoring a customer list at once
- Track prediction drift if weights.json is retrained on newer data later

## 16. License

MIT — see [`LICENSE`](LICENSE).
