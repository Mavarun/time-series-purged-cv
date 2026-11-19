# Embargo cost on a real edge (slice 8b)

100 seeds per cell (from 8100), n = 500, beta = 0.15, horizon 20, k-NN. Truth = full-sample model on the next 5000 bars. Cells: CV accuracy, bias vs truth (s.e.), and cost vs purge only (s.e.).

| folds | shock phi | truth | purge only | 1% after (default) | ACF both sides |
| ---: | ---: | ---: | :--- | :--- | :--- |
| 5 | 0.0 | 0.5620 | 0.5619, bias -0.0001 (0.0093), cost +0.0000 (0.0000) | 0.5599, bias -0.0021 (0.0091), cost +0.0020 (0.0016) | 0.5555, bias -0.0065 (0.0093), cost +0.0064 (0.0034) |
| 5 | 0.9 | 0.5062 | 0.4983, bias -0.0080 (0.0107), cost +0.0000 (0.0000) | 0.4962, bias -0.0100 (0.0107), cost +0.0020 (0.0014) | 0.4946, bias -0.0116 (0.0107), cost +0.0036 (0.0024) |
| 20 | 0.0 | 0.5620 | 0.5703, bias +0.0083 (0.0092), cost +0.0000 (0.0000) | 0.5695, bias +0.0075 (0.0093), cost +0.0008 (0.0014) | 0.5620, bias +0.0000 (0.0094), cost +0.0083 (0.0027) |
| 20 | 0.9 | 0.5062 | 0.5106, bias +0.0043 (0.0101), cost +0.0000 (0.0000) | 0.5082, bias +0.0020 (0.0101), cost +0.0024 (0.0012) | 0.4995, bias -0.0067 (0.0104), cost +0.0111 (0.0030) |

Pre-stated call: worst embargo cost +0.0111 (limit 0.005); embargo cheap on a real edge: **no**.
