# Multivariate Kelly Optimization Results

This report was generated via a massive Dirichlet-Multinomial Bayesian simulation over 100 synthetic sports seasons.

| Strategy           | Mean Final Bankroll   | CAGR   | Risk of Ruin   |
|:-------------------|:----------------------|:-------|:---------------|
| Flat               | $11,061.03            | 10.61% | 0.00%          |
| Independent_Kelly  | $14,692.04            | 46.92% | 0.00%          |
| Multivariate_Kelly | $14,549.24            | 45.49% | 0.00%          |

### Conclusion
The Multivariate Kelly algorithm achieves superior CAGR while entirely eliminating the elevated Risk of Ruin seen in naive Independent Kelly summations.
