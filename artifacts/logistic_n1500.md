# Logistic leak at n 1500 (slice 11)

480 seeds per cell (from 9400), n 1500, 20 folds, horizon 20; cells show CV accuracy - 0.5 (s.e.). `*` = more than 2 s.e. above zero (leak).

| shock phi | purge only | 1% after (default) | no purge (control) | slice 10 control (n 500) |
| ---: | :--- | :--- | :--- | ---: |
| 0.0 | +0.0001 (0.0028) | +0.0002 (0.0027) | +0.0073 (0.0028)* | +0.0279 |
| 0.9 | +0.0010 (0.0036) | +0.0004 (0.0036) | +0.0128 (0.0035)* | +0.0541 |

Control leaks in 2 of 2 cells: measured **yes**; default leaks in 0: default holds **yes**; shrinks with n **yes**. Largest control optimism + 2 s.e.: 0.0197 (slice 9 bound 0.040).
