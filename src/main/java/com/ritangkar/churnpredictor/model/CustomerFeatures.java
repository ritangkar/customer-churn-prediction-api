package com.ritangkar.churnpredictor.model;

import jakarta.validation.constraints.DecimalMax;
import jakarta.validation.constraints.DecimalMin;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotNull;

/**
 * The feature set the model was trained on, in the same units as
 * sample-data/customers.csv. Field order here does not need to match the
 * model's internal feature order — {@code ChurnModel} maps by name.
 */
public record CustomerFeatures(
        @NotNull @Min(0) @Max(240) Integer tenureMonths,
        @NotNull @DecimalMin("0.0") @DecimalMax("100000.0") Double monthlySpend,
        @NotNull @Min(0) @Max(50) Integer supportTicketsLast90d,
        @NotNull @Min(0) @Max(1000) Integer daysSinceLastOrder,
        @NotNull @DecimalMin("0.0") @DecimalMax("1.0") Double discountUsageRate,
        @NotNull @Min(0) @Max(10) Integer npsScore,
        @NotNull Boolean isSubscribed
) {
}
