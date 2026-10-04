package com.ritangkar.churnpredictor.service;

import com.ritangkar.churnpredictor.ml.ChurnModel;
import com.ritangkar.churnpredictor.model.ChurnPrediction;
import com.ritangkar.churnpredictor.model.CustomerFeatures;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

class ChurnPredictionServiceTest {

    private final ChurnModel model = new ChurnModel();
    private final ChurnPredictionService service = new ChurnPredictionService(model, 0.30, 0.60);

    @Test
    void classifiesALoyalSubscriberAsLowRisk() {
        CustomerFeatures loyal = new CustomerFeatures(48, 4500.0, 0, 5, 0.1, 9, true);
        ChurnPrediction prediction = service.predict(loyal);

        assertThat(prediction.riskLevel()).isEqualTo("LOW");
        assertThat(prediction.churnProbability()).isLessThan(0.30);
    }

    @Test
    void classifiesADisengagedNonSubscriberAsHighRisk() {
        CustomerFeatures atRisk = new CustomerFeatures(2, 400.0, 5, 150, 0.9, 1, false);
        ChurnPrediction prediction = service.predict(atRisk);

        assertThat(prediction.riskLevel()).isEqualTo("HIGH");
        assertThat(prediction.churnProbability()).isGreaterThan(0.60);
    }

    @Test
    void returnsExactlyThreeTopFactors() {
        CustomerFeatures customer = new CustomerFeatures(12, 1800.0, 2, 45, 0.4, 4, false);
        ChurnPrediction prediction = service.predict(customer);

        assertThat(prediction.topFactors()).hasSize(3);
    }

    @Test
    void riskTierBoundariesAreRespected() {
        ChurnPredictionService boundaryService = new ChurnPredictionService(model, 0.30, 0.60);

        assertThat(boundaryService.riskLevel(0.29)).isEqualTo("LOW");
        assertThat(boundaryService.riskLevel(0.30)).isEqualTo("MEDIUM");
        assertThat(boundaryService.riskLevel(0.59)).isEqualTo("MEDIUM");
        assertThat(boundaryService.riskLevel(0.60)).isEqualTo("HIGH");
    }
}
