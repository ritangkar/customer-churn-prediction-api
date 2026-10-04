package com.ritangkar.churnpredictor.model;

/**
 * One feature's contribution to a single prediction's logit, in the
 * standardized space the model was trained in. {@code contribution} is
 * {@code weight * standardizedValue} — signed, so its magnitude ranks
 * feature importance for this specific customer and its sign gives the
 * direction.
 */
public record FeatureContribution(
        String feature,
        double rawValue,
        double standardizedValue,
        double weight,
        double contribution,
        String direction   // "INCREASES_RISK" or "DECREASES_RISK"
) {
}
