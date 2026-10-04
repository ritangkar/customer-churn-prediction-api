# Portfolio Demo

**Demo level:** 1 — client-side, fully deterministic.

Logistic regression inference is a dot product and a sigmoid — small enough
to port exactly. `lab/demos/customer-churn-prediction-api.js` copies the
literal `means`, `stds`, `weights` and `bias` values from the committed
`weights.json` (not re-trained in the browser — the training happened once,
offline, in `tools/ChurnTrainer.java`) and recomputes the same
`predictProbability` / `explain` logic in JavaScript.

Moving a slider (tenure, spend, support tickets, days since last order,
discount usage, NPS, subscription toggle) recomputes a real prediction from
the real model instantly, with the same top-3 contributing factors the Java
API would return for identical input — no backend call, so it works on
GitHub Pages with zero setup.

Running `mvn spring-boot:run` and calling `GET /api/churn/model-info`
locally shows the real training metadata behind the model the demo is using.
