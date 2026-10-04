import java.io.*;
import java.nio.file.*;
import java.util.*;

/**
 * Trains a logistic regression churn model on customers.csv and writes the
 * learned parameters to weights.json. This mirrors a real "train offline,
 * serve online" ML workflow: this class runs once, out of band; the Spring
 * Boot service only ever loads its output at startup and never retrains
 * live.
 */
public class ChurnTrainer {

    static final String[] FEATURE_NAMES = {
            "tenure_months", "monthly_spend", "support_tickets_last_90d",
            "days_since_last_order", "discount_usage_rate", "nps_score", "is_subscribed"
    };

    public static void main(String[] args) throws IOException {
        List<String> lines = Files.readAllLines(Path.of("customers.csv"));
        String[] header = lines.get(0).split(",");

        int n = lines.size() - 1;
        double[][] rawX = new double[n][FEATURE_NAMES.length];
        int[] y = new int[n];

        for (int i = 0; i < n; i++) {
            String[] parts = lines.get(i + 1).split(",");
            for (int f = 0; f < FEATURE_NAMES.length; f++) {
                rawX[i][f] = Double.parseDouble(parts[f]);
            }
            y[i] = Integer.parseInt(parts[FEATURE_NAMES.length]);
        }

        // Deterministic 80/20 split: every 5th row (by original order) goes to test.
        List<Integer> trainIdx = new ArrayList<>();
        List<Integer> testIdx = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            if (i % 5 == 0) testIdx.add(i); else trainIdx.add(i);
        }

        double[] means = new double[FEATURE_NAMES.length];
        double[] stds = new double[FEATURE_NAMES.length];
        for (int f = 0; f < FEATURE_NAMES.length; f++) {
            double sum = 0;
            for (int i : trainIdx) sum += rawX[i][f];
            double mean = sum / trainIdx.size();
            double sqSum = 0;
            for (int i : trainIdx) sqSum += Math.pow(rawX[i][f] - mean, 2);
            double std = Math.sqrt(sqSum / trainIdx.size());
            means[f] = mean;
            stds[f] = std == 0 ? 1.0 : std;
        }

        double[][] X = standardize(rawX, means, stds);

        // Class weighting for the imbalanced label (~14% positive): standard
        // "balanced" weighting so a churn example counts as much as roughly
        // (negatives / positives) unweighted ones, instead of drowning it out.
        int trainPos = 0;
        for (int idx : trainIdx) trainPos += y[idx];
        int trainNeg = trainIdx.size() - trainPos;
        double posWeight = trainPos == 0 ? 1.0 : trainNeg / (double) trainPos;

        double[] weights = new double[FEATURE_NAMES.length];
        double bias = 0.0;
        double learningRate = 0.15;
        double l2Lambda = 0.002;
        int epochs = 4000;

        double finalLoss = 0;
        for (int epoch = 0; epoch < epochs; epoch++) {
            double[] gradW = new double[FEATURE_NAMES.length];
            double gradB = 0;
            double loss = 0;
            double weightSum = 0;

            for (int idx : trainIdx) {
                double sampleWeight = y[idx] == 1 ? posWeight : 1.0;
                double z = bias;
                for (int f = 0; f < FEATURE_NAMES.length; f++) z += weights[f] * X[idx][f];
                double pred = sigmoid(z);
                double error = sampleWeight * (pred - y[idx]);

                for (int f = 0; f < FEATURE_NAMES.length; f++) gradW[f] += error * X[idx][f];
                gradB += error;
                weightSum += sampleWeight;

                double eps = 1e-12;
                loss += sampleWeight * -(y[idx] * Math.log(pred + eps) + (1 - y[idx]) * Math.log(1 - pred + eps));
            }

            for (int f = 0; f < FEATURE_NAMES.length; f++) {
                double reg = l2Lambda * weights[f];
                weights[f] -= learningRate * (gradW[f] / weightSum + reg);
            }
            bias -= learningRate * (gradB / weightSum);

            finalLoss = loss / weightSum;
        }

        // Evaluate on held-out test set.
        int tp = 0, fp = 0, tn = 0, fn = 0;
        int correct = 0;
        int majorityClass = majorityLabel(y, trainIdx);
        int baselineCorrect = 0;

        for (int idx : testIdx) {
            double z = bias;
            for (int f = 0; f < FEATURE_NAMES.length; f++) z += weights[f] * X[idx][f];
            double pred = sigmoid(z);
            int predicted = pred >= 0.5 ? 1 : 0;
            int actual = y[idx];

            if (predicted == actual) correct++;
            if (predicted == 1 && actual == 1) tp++;
            else if (predicted == 1 && actual == 0) fp++;
            else if (predicted == 0 && actual == 0) tn++;
            else fn++;

            if (majorityClass == actual) baselineCorrect++;
        }

        double testAccuracy = correct / (double) testIdx.size();
        double baselineAccuracy = baselineCorrect / (double) testIdx.size();
        double precision = (tp + fp) == 0 ? 0 : tp / (double) (tp + fp);
        double recall = (tp + fn) == 0 ? 0 : tp / (double) (tp + fn);
        double f1 = (precision + recall) == 0 ? 0 : 2 * precision * recall / (precision + recall);

        // Train accuracy for reference.
        int trainCorrect = 0;
        for (int idx : trainIdx) {
            double z = bias;
            for (int f = 0; f < FEATURE_NAMES.length; f++) z += weights[f] * X[idx][f];
            int predicted = sigmoid(z) >= 0.5 ? 1 : 0;
            if (predicted == y[idx]) trainCorrect++;
        }
        double trainAccuracy = trainCorrect / (double) trainIdx.size();

        writeWeightsJson(means, stds, weights, bias, n, trainIdx.size(), testIdx.size(),
                epochs, learningRate, l2Lambda, finalLoss, trainAccuracy,
                testAccuracy, baselineAccuracy, precision, recall, f1, tp, fp, tn, fn, posWeight);

        System.out.printf(
                "Train n=%d test n=%d%ntrainAcc=%.4f testAcc=%.4f baselineAcc=%.4f%nprecision=%.4f recall=%.4f f1=%.4f%ntp=%d fp=%d tn=%d fn=%d%n",
                trainIdx.size(), testIdx.size(), trainAccuracy, testAccuracy, baselineAccuracy, precision, recall, f1, tp, fp, tn, fn);
        System.out.println("bias=" + bias);
        for (int f = 0; f < FEATURE_NAMES.length; f++) {
            System.out.printf("  %-25s weight=%.4f mean=%.3f std=%.3f%n", FEATURE_NAMES[f], weights[f], means[f], stds[f]);
        }
    }

    static double[][] standardize(double[][] raw, double[] means, double[] stds) {
        double[][] out = new double[raw.length][raw[0].length];
        for (int i = 0; i < raw.length; i++) {
            for (int f = 0; f < raw[i].length; f++) {
                out[i][f] = (raw[i][f] - means[f]) / stds[f];
            }
        }
        return out;
    }

    static double sigmoid(double z) {
        return 1.0 / (1.0 + Math.exp(-z));
    }

    static int majorityLabel(int[] y, List<Integer> idxs) {
        int ones = 0;
        for (int i : idxs) ones += y[i];
        return ones * 2 > idxs.size() ? 1 : 0;
    }

    static void writeWeightsJson(double[] means, double[] stds, double[] weights, double bias,
                                  int totalN, int trainN, int testN, int epochs, double lr, double l2,
                                  double finalLoss, double trainAcc, double testAcc, double baselineAcc,
                                  double precision, double recall, double f1,
                                  int tp, int fp, int tn, int fn, double posWeight) throws IOException {
        StringBuilder sb = new StringBuilder();
        sb.append("{\n");
        sb.append("  \"featureNames\": [");
        for (int i = 0; i < FEATURE_NAMES.length; i++) {
            sb.append("\"").append(FEATURE_NAMES[i]).append("\"").append(i < FEATURE_NAMES.length - 1 ? ", " : "");
        }
        sb.append("],\n");

        sb.append("  \"means\": [");
        for (int i = 0; i < means.length; i++) sb.append(means[i]).append(i < means.length - 1 ? ", " : "");
        sb.append("],\n");

        sb.append("  \"stds\": [");
        for (int i = 0; i < stds.length; i++) sb.append(stds[i]).append(i < stds.length - 1 ? ", " : "");
        sb.append("],\n");

        sb.append("  \"weights\": [");
        for (int i = 0; i < weights.length; i++) sb.append(round4(weights[i])).append(i < weights.length - 1 ? ", " : "");
        sb.append("],\n");

        sb.append("  \"bias\": ").append(round4(bias)).append(",\n");

        sb.append("  \"trainingMeta\": {\n");
        sb.append("    \"algorithm\": \"logistic-regression-gradient-descent\",\n");
        sb.append("    \"totalSamples\": ").append(totalN).append(",\n");
        sb.append("    \"trainSamples\": ").append(trainN).append(",\n");
        sb.append("    \"testSamples\": ").append(testN).append(",\n");
        sb.append("    \"epochs\": ").append(epochs).append(",\n");
        sb.append("    \"learningRate\": ").append(lr).append(",\n");
        sb.append("    \"l2Lambda\": ").append(l2).append(",\n");
        sb.append("    \"positiveClassWeight\": ").append(round4(posWeight)).append(",\n");
        sb.append("    \"finalTrainLoss\": ").append(round4(finalLoss)).append(",\n");
        sb.append("    \"trainAccuracy\": ").append(round4(trainAcc)).append(",\n");
        sb.append("    \"testAccuracy\": ").append(round4(testAcc)).append(",\n");
        sb.append("    \"baselineMajorityClassAccuracy\": ").append(round4(baselineAcc)).append(",\n");
        sb.append("    \"testPrecision\": ").append(round4(precision)).append(",\n");
        sb.append("    \"testRecall\": ").append(round4(recall)).append(",\n");
        sb.append("    \"testF1\": ").append(round4(f1)).append(",\n");
        sb.append("    \"testConfusionMatrix\": {\"truePositive\": ").append(tp)
                .append(", \"falsePositive\": ").append(fp)
                .append(", \"trueNegative\": ").append(tn)
                .append(", \"falseNegative\": ").append(fn).append("},\n");
        sb.append("    \"trainedOn\": \"synthetic sample-data/customers.csv, not real customer data\"\n");
        sb.append("  }\n");
        sb.append("}\n");

        Files.writeString(Path.of("weights.json"), sb.toString());
    }

    static double round4(double v) {
        return Math.round(v * 10000.0) / 10000.0;
    }
}
