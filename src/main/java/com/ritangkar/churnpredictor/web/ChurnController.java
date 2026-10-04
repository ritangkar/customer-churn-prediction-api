package com.ritangkar.churnpredictor.web;

import com.fasterxml.jackson.databind.JsonNode;
import com.ritangkar.churnpredictor.ml.ChurnModel;
import com.ritangkar.churnpredictor.model.ChurnPrediction;
import com.ritangkar.churnpredictor.model.CustomerFeatures;
import com.ritangkar.churnpredictor.service.ChurnPredictionService;
import jakarta.validation.Valid;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/churn")
public class ChurnController {

    private final ChurnPredictionService predictionService;
    private final ChurnModel model;

    public ChurnController(ChurnPredictionService predictionService, ChurnModel model) {
        this.predictionService = predictionService;
        this.model = model;
    }

    /** POST /api/churn/predict — churn probability, risk tier and top contributing factors. */
    @PostMapping("/predict")
    public ChurnPrediction predict(@Valid @RequestBody CustomerFeatures features) {
        return predictionService.predict(features);
    }

    /**
     * GET /api/churn/model-info — the real training metadata (algorithm, sample
     * sizes, accuracy/precision/recall/F1 on a held-out test split) for the
     * model currently loaded. Nothing here is fabricated; it is exactly what
     * /tools/ChurnTrainer.java printed and wrote when the model was trained.
     */
    @GetMapping("/model-info")
    public JsonNode modelInfo() {
        return model.trainingMeta();
    }
}
