# Two-sided ACF embargo vs the purged k-NN residual (slice 13)

480 seeds (from 10400), k-NN, n 500, 20 folds, horizon 20, shock phi 0.9. Optimism is CV accuracy - 0.5 (s.e.); `*` = more than 2 s.e. above zero.

| arm | optimism | mean train rows |
| :--- | :--- | ---: |
| purge_only | +0.0018 (0.0041) | 8740 |
| pct1_after | -0.0005 (0.0041) | 8645 |
| acf_both | -0.0109 (0.0041) | 8393 |
| contiguous_no_purge | +0.0904 (0.0037)* | 9500 |

Paired default - acf_both: +0.0104 (s.e. 0.0016).

Reproduced: **NO**. acf_both leak-free: **yes**. Paired drop > 2 s.e.: **yes**. Removes residual: **no**. Row share 0.971 (within the 0.90 budget).
