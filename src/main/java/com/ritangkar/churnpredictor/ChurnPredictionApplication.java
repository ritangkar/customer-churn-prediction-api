package com.ritangkar.churnpredictor;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

/**
 * Customer Churn Prediction API.
 *
 * Serves a from-scratch logistic regression model — trained offline (see
 * /tools/ChurnTrainer.java) on synthetic data, with its learned weights
 * committed as an artifact ({@code src/main/resources/model/weights.json})
 * — with per-prediction, feature-level explainability.
 */
@SpringBootApplication
public class ChurnPredictionApplication {

    public static void main(String[] args) {
        SpringApplication.run(ChurnPredictionApplication.class, args);
    }
}
