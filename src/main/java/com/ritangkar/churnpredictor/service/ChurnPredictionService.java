package com.ritangkar.churnpredictor.service;

import com.ritangkar.churnpredictor.ml.ChurnModel;
import com.ritangkar.churnpredictor.model.ChurnPrediction;
import com.ritangkar.churnpredictor.model.CustomerFeatures;
import com.ritangkar.churnpredictor.model.FeatureContribution;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class ChurnPredictionService {

    private static final int TOP_FACTORS_COUNT = 3;
    private static final String MODEL_VERSION = "logistic-regression-v1";

    private final ChurnModel model;
    private final double lowRiskCeiling;
    private final double mediumRiskCeiling;

    public ChurnPredictionService(
            ChurnModel model,
            @Value("${churn.risk.low-ceiling:0.30}") double lowRiskCeiling,
            @Value("${churn.risk.medium-ceiling:0.60}") double mediumRiskCeiling) {
        this.model = model;
        this.lowRiskCeiling = lowRiskCeiling;
        this.mediumRiskCeiling = mediumRiskCeiling;
    }

    public ChurnPrediction predict(CustomerFeatures features) {
        double[] rawVector = model.toOrderedFeatureVector(features);
        double probability = model.predictProbability(rawVector);
        List<FeatureContribution> allFactors = model.explain(rawVector);
        List<FeatureContribution> topFactors = allFactors.stream().limit(TOP_FACTORS_COUNT).toList();

        return new ChurnPrediction(round4(probability), riskLevel(probability), topFactors, MODEL_VERSION);
    }

    // Package-private rather than private specifically so the boundary
    // behavior can be unit-tested directly (see ChurnPredictionServiceTest)
    // without needing to reverse-engineer a feature vector that lands on an
    // exact probability.
    String riskLevel(double probability) {
        if (probability < lowRiskCeiling) return "LOW";
        if (probability < mediumRiskCeiling) return "MEDIUM";
        return "HIGH";
    }

    private static double round4(double v) {
        return Math.round(v * 10000.0) / 10000.0;
    }
}
