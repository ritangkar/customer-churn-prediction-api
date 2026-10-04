package com.ritangkar.churnpredictor.model;

import java.util.List;

public record ChurnPrediction(
        double churnProbability,
        String riskLevel,          // LOW, MEDIUM, HIGH
        List<FeatureContribution> topFactors,
        String modelVersion
) {
}
