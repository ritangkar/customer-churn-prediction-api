import java.io.*;
import java.nio.file.*;
import java.util.*;

public class ChurnDataGen {

    public static void main(String[] args) throws IOException {
        long seed = 20260918L;
        Random rng = new Random(seed);
        int n = 900;

        StringBuilder csv = new StringBuilder();
        csv.append("tenure_months,monthly_spend,support_tickets_last_90d,days_since_last_order,discount_usage_rate,nps_score,is_subscribed,churned\n");

        for (int i = 0; i < n; i++) {
            int tenureMonths = 1 + rng.nextInt(60);
            double monthlySpend = round2(200 + rng.nextDouble() * 5800);
            int supportTickets = poissonish(rng, 1.1);
            int daysSinceLastOrder = 1 + rng.nextInt(180);
            double discountUsageRate = round2(rng.nextDouble());
            int npsScore = rng.nextInt(11);
            int isSubscribed = rng.nextDouble() < 0.45 ? 1 : 0;

            double logit = -2.0
                    - 0.030 * tenureMonths
                    - 0.00030 * monthlySpend
                    + 0.35 * supportTickets
                    + 0.020 * daysSinceLastOrder
                    + 1.20 * discountUsageRate
                    - 0.25 * npsScore
                    - 0.60 * isSubscribed;

            double probability = 1.0 / (1.0 + Math.exp(-logit));
            int churned = rng.nextDouble() < probability ? 1 : 0;

            csv.append(tenureMonths).append(',')
                    .append(monthlySpend).append(',')
                    .append(supportTickets).append(',')
                    .append(daysSinceLastOrder).append(',')
                    .append(discountUsageRate).append(',')
                    .append(npsScore).append(',')
                    .append(isSubscribed).append(',')
                    .append(churned).append('\n');
        }

        Files.writeString(Path.of("customers.csv"), csv.toString());
        System.out.println("Generated " + n + " labeled customer rows.");
    }

    static int poissonish(Random rng, double lambda) {
        lambda = Math.max(0.15, lambda);
        double l = Math.exp(-lambda);
        int k = 0;
        double p = 1.0;
        do {
            k++;
            p *= rng.nextDouble();
        } while (p > l && k < 9);
        return k - 1;
    }

    static double round2(double v) {
        return Math.round(v * 100.0) / 100.0;
    }
}
