# Logistic power (slice 10)

240 seeds per cell (from 9100), n 500, 20 folds, horizon 20; cells show CV accuracy - 0.5 (s.e.). `*` = more than 2 s.e. above zero (leak).

| learner | shock phi | purge only | 1% after (default) | no purge (control) |
| :--- | ---: | :--- | :--- | :--- |
| logistic | 0.0 | -0.0028 (0.0065) | -0.0022 (0.0064) | +0.0279 (0.0061)* |
| logistic | 0.9 | +0.0008 (0.0083) | -0.0000 (0.0083) | +0.0541 (0.0073)* |
| knn | 0.0 | +0.0074 (0.0045) | +0.0073 (0.0045) | +0.0675 (0.0040)* |
| knn | 0.9 | +0.0078 (0.0061) | +0.0063 (0.0061) | +0.1011 (0.0051)* |

k-NN reproduction of slice 8: **pass**. Logistic control leaks in 2 of 2 cells: testable **yes**; default leaks in 0: default holds **yes**. Largest logistic control optimism + 2 s.e.: 0.0686.
