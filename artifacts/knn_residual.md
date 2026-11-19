# Purged k-NN residual at n 500 (slice 12)

480 seeds per cell (from 9900), n 500, 20 folds, horizon 20; cells show CV accuracy - 0.5 (s.e.). `*` = more than 2 s.e. above zero (leak).

| shock phi | purge only | 1% after (default) | no purge (control) | slice 10 default |
| ---: | :--- | :--- | :--- | ---: |
| 0.0 | +0.0024 (0.0034) | +0.0029 (0.0034) | +0.0648 (0.0030)* | +0.0073 |
| 0.9 | +0.0131 (0.0042)* | +0.0110 (0.0042)* | +0.1048 (0.0037)* | +0.0063 |

Control reproduces (leaks in both cells): **yes**. Default leaks in 1 of 2 cells, purge only in 1: residual real **yes**. Largest default optimism + 2 s.e.: 0.0193.
