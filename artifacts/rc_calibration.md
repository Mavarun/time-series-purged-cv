# Reality check (stationary bootstrap) vs Gaussian max-statistic: normal vs GARCH-t returns

200 seeds per cell, n=2520 daily bars, 40-rule grid, 4000 Gaussian null draws, 499 bootstrap draws, mean block 10 bars. Pass = DSR > 0.95 or p < 0.05. phi=0 rows are false-positive rates; phi>0 rows are power. +- is one binomial s.e.

| returns | phi | cost bps | DSR raw N=40 | DSR participation N | DSR simulated N | max-stat p<0.05 | reality check p<0.05 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| normal | 0.00 | 0 | 0.000 +- 0.000 | 0.060 +- 0.017 | 0.010 +- 0.007 | 0.050 +- 0.015 | 0.055 +- 0.016 |
| normal | 0.00 | 2 | 0.000 +- 0.000 | 0.040 +- 0.014 | 0.005 +- 0.005 | 0.035 +- 0.013 | 0.025 +- 0.011 |
| normal | 0.10 | 0 | 0.775 +- 0.030 | 0.980 +- 0.010 | 0.845 +- 0.026 | 0.975 +- 0.011 | 0.950 +- 0.015 |
| normal | 0.10 | 2 | 0.430 +- 0.035 | 0.840 +- 0.026 | 0.545 +- 0.035 | 0.785 +- 0.029 | 0.730 +- 0.031 |
| garch | 0.00 | 0 | 0.005 +- 0.005 | 0.085 +- 0.020 | 0.005 +- 0.005 | 0.060 +- 0.017 | 0.055 +- 0.016 |
| garch | 0.00 | 2 | 0.000 +- 0.000 | 0.040 +- 0.014 | 0.005 +- 0.005 | 0.020 +- 0.010 | 0.015 +- 0.009 |
| garch | 0.10 | 0 | 0.590 +- 0.035 | 0.930 +- 0.018 | 0.695 +- 0.033 | 0.895 +- 0.022 | 0.890 +- 0.022 |
| garch | 0.10 | 2 | 0.200 +- 0.028 | 0.695 +- 0.033 | 0.315 +- 0.033 | 0.645 +- 0.034 | 0.625 +- 0.034 |
