package com.ritangkar.churnpredictor.ml;

import com.ritangkar.churnpredictor.model.CustomerFeatures;
import com.ritangkar.churnpredictor.model.FeatureContribution;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

/**
 * These expected probabilities were independently computed from the
 * committed weights.json (same formula, verified with a standalone script)
 * before this test was written — see docs/design-decisions.md.
 */
class ChurnModelTest {

    private final ChurnModel model = new ChurnModel();

    @Test
    void loyalHappySubscriberGetsALowChurnProbability() {
        CustomerFeatures loyal = new CustomerFeatures(48, 4500.0, 0, 5, 0.1, 9, true);
        double probability = model.predictProbability(model.toOrderedFeatureVector(loyal));

        assertThat(probability).isCloseTo(0.0115, within(0.005));
    }

    @Test
    void newDisengagedNonSubscriberGetsAHighChurnProbability() {
        CustomerFeatures atRisk = new CustomerFeatures(2, 400.0, 5, 150, 0.9, 1, false);
        double probability = model.predictProbability(model.toOrderedFeatureVector(atRisk));

        assertThat(probability).isCloseTo(0.9871, within(0.005));
    }

    @Test
    void explainReturnsOneContributionPerFeatureSortedByMagnitude() {
        CustomerFeatures atRisk = new CustomerFeatures(2, 400.0, 5, 150, 0.9, 1, false);
        List<FeatureContribution> contributions = model.explain(model.toOrderedFeatureVector(atRisk));

        assertThat(contributions).hasSize(model.featureNames().size());
        for (int i = 0; i < contributions.size() - 1; i++) {
            assertThat(Math.abs(contributions.get(i).contribution()))
                    .isGreaterThanOrEqualTo(Math.abs(contributions.get(i + 1).contribution()));
        }
    }

    @Test
    void directionMatchesTheSignOfTheContribution() {
        CustomerFeatures loyal = new CustomerFeatures(48, 4500.0, 0, 5, 0.1, 9, true);
        List<FeatureContribution> contributions = model.explain(model.toOrderedFeatureVector(loyal));

        for (FeatureContribution c : contributions) {
            String expected = c.contribution() >= 0 ? "INCREASES_RISK" : "DECREASES_RISK";
            assertThat(c.direction()).isEqualTo(expected);
        }
    }

    @Test
    void trainingMetaIsExposedAndReportsAHeldOutTestSplit() {
        assertThat(model.trainingMeta()).isNotNull();
        assertThat(model.trainingMeta().get("testSamples").asInt()).isGreaterThan(0);
        assertThat(model.trainingMeta().get("algorithm").asText()).contains("logistic-regression");
    }
}
