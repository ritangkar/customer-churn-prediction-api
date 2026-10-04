package com.ritangkar.churnpredictor.ml;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.ritangkar.churnpredictor.model.CustomerFeatures;
import com.ritangkar.churnpredictor.model.FeatureContribution;
import org.springframework.core.io.ClassPathResource;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.io.InputStream;
import java.util.*;

/**
 * A from-scratch logistic regression model: standardize features, compute
 * {@code sigmoid(bias + sum(weight_i * standardized_i))}. The weights below
 * were learned offline by {@code /tools/ChurnTrainer.java} on the synthetic
 * dataset in {@code /sample-data/customers.csv} and committed as a JSON
 * artifact — this class only ever loads and applies them; it never trains.
 *
 * See {@code trainingMeta} (exposed via {@code GET /api/churn/model-info})
 * for the real accuracy/precision/recall this model achieved on a held-out
 * test split, including the honest trade-off documented in
 * docs/design-decisions.md.
 */
@Component
public class ChurnModel {

    private final List<String> featureNames;
    private final double[] means;
    private final double[] stds;
    private final double[] weights;
    private final double bias;
    private final JsonNode trainingMeta;

    public ChurnModel() {
        ObjectMapper mapper = new ObjectMapper();
        try (InputStream in = new ClassPathResource("model/weights.json").getInputStream()) {
            JsonNode root = mapper.readTree(in);

            this.featureNames = new ArrayList<>();
            root.get("featureNames").forEach(n -> featureNames.add(n.asText()));

            this.means = toDoubleArray(root.get("means"));
            this.stds = toDoubleArray(root.get("stds"));
            this.weights = toDoubleArray(root.get("weights"));
            this.bias = root.get("bias").asDouble();
            this.trainingMeta = root.get("trainingMeta");
        } catch (IOException e) {
            throw new IllegalStateException("Could not load model/weights.json", e);
        }
    }

    public JsonNode trainingMeta() {
        return trainingMeta;
    }

    public List<String> featureNames() {
        return List.copyOf(featureNames);
    }

    /** Raw feature values in the exact order of {@link #featureNames()}. */
    public double[] toOrderedFeatureVector(CustomerFeatures f) {
        Map<String, Double> byName = Map.of(
                "tenure_months", f.tenureMonths().doubleValue(),
                "monthly_spend", f.monthlySpend(),
                "support_tickets_last_90d", f.supportTicketsLast90d().doubleValue(),
                "days_since_last_order", f.daysSinceLastOrder().doubleValue(),
                "discount_usage_rate", f.discountUsageRate(),
                "nps_score", f.npsScore().doubleValue(),
                "is_subscribed", f.isSubscribed() ? 1.0 : 0.0
        );
        double[] ordered = new double[featureNames.size()];
        for (int i = 0; i < featureNames.size(); i++) {
            Double v = byName.get(featureNames.get(i));
            if (v == null) {
                throw new IllegalStateException("weights.json references unknown feature: " + featureNames.get(i));
            }
            ordered[i] = v;
        }
        return ordered;
    }

    public double predictProbability(double[] rawFeatures) {
        double z = bias;
        for (int i = 0; i < rawFeatures.length; i++) {
            double standardized = (rawFeatures[i] - means[i]) / stds[i];
            z += weights[i] * standardized;
        }
        return sigmoid(z);
    }

    public List<FeatureContribution> explain(double[] rawFeatures) {
        List<FeatureContribution> contributions = new ArrayList<>();
        for (int i = 0; i < rawFeatures.length; i++) {
            double standardized = (rawFeatures[i] - means[i]) / stds[i];
            double contribution = weights[i] * standardized;
            contributions.add(new FeatureContribution(
                    featureNames.get(i), rawFeatures[i], round4(standardized), round4(weights[i]),
                    round4(contribution), contribution >= 0 ? "INCREASES_RISK" : "DECREASES_RISK"
            ));
        }
        contributions.sort((a, b) -> Double.compare(Math.abs(b.contribution()), Math.abs(a.contribution())));
        return contributions;
    }

    private double[] toDoubleArray(JsonNode arrayNode) {
        double[] arr = new double[arrayNode.size()];
        for (int i = 0; i < arr.length; i++) arr[i] = arrayNode.get(i).asDouble();
        return arr;
    }

    private double sigmoid(double z) {
        return 1.0 / (1.0 + Math.exp(-z));
    }

    private static double round4(double v) {
        return Math.round(v * 10000.0) / 10000.0;
    }
}
