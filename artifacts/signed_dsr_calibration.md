# Slice 14: signed participation DSR (300 seeds per cell)

## False positives (phi = 0)

| returns | bars | DSR participation | DSR signed | DSR simulated | max-stat p<0.05 | mean N participation / signed |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| normal | 252 | 0.073 | 0.013 | 0.000 | 0.047 | 4.4 / 8.8 |
| normal | 504 | 0.057 | 0.010 | 0.003 | 0.037 | 4.5 / 9.0 |
| normal | 1260 | 0.040 | 0.010 | 0.003 | 0.023 | 4.5 / 9.0 |
| GARCH-t | 252 | 0.097 | 0.023 | 0.007 | 0.030 | 4.1 / 8.1 |
| GARCH-t | 504 | 0.077 | 0.023 | 0.010 | 0.030 | 4.2 / 8.4 |
| GARCH-t | 1260 | 0.040 | 0.013 | 0.003 | 0.020 | 4.3 / 8.6 |
| vol regime | 252 | 0.097 | 0.027 | 0.000 | 0.040 | 4.2 / 8.4 |
| vol regime | 504 | 0.063 | 0.017 | 0.003 | 0.040 | 4.2 / 8.4 |
| vol regime | 1260 | 0.060 | 0.010 | 0.007 | 0.017 | 4.3 / 8.5 |

## Power (phi = 0.1)

| returns | bars | DSR participation | DSR signed | DSR simulated | max-stat p<0.05 | mean N participation / signed |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| normal | 252 | 0.193 | 0.067 | 0.020 | 0.123 | 4.3 / 8.6 |
| normal | 504 | 0.240 | 0.127 | 0.043 | 0.177 | 4.4 / 8.8 |
| normal | 1260 | 0.463 | 0.323 | 0.190 | 0.407 | 4.4 / 8.8 |
| GARCH-t | 252 | 0.190 | 0.083 | 0.027 | 0.093 | 3.9 / 7.9 |
| GARCH-t | 504 | 0.247 | 0.120 | 0.047 | 0.173 | 4.1 / 8.2 |
| GARCH-t | 1260 | 0.417 | 0.240 | 0.133 | 0.320 | 4.2 / 8.3 |
| vol regime | 252 | 0.183 | 0.067 | 0.017 | 0.103 | 4.1 / 8.2 |
| vol regime | 504 | 0.197 | 0.093 | 0.033 | 0.130 | 4.1 / 8.2 |
| vol regime | 1260 | 0.377 | 0.237 | 0.123 | 0.313 | 4.2 / 8.3 |

Calls: H1 True, H2 False (1/9 power cells within 0.03 of max-stat), recommend signed: False
