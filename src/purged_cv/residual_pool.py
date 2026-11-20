"""Post-hoc pooling of the purged k-NN default cell (n 500, 20 folds, shock phi 0.9) across slices 10, 12 and 13.

Not a pre-stated test: written after slice 13 failed to reproduce slice 12's leak. Each slice used
disjoint seeds and the same data pipeline, so the cell means are independent estimates of one
quantity and are pooled by inverse variance. Heterogeneity is Cochran's Q against chi-square with
k - 1 degrees of freedom.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from scipy.stats import chi2

ROOT = Path(__file__).resolve().parents[2]
ARM = "pct1_after"


def cell_estimates(root: Path = ROOT) -> dict[str, dict[str, float]]:
    art = root / "artifacts"
    s10 = json.loads((art / "logistic_power.json").read_text())
    s12 = json.loads((art / "knn_residual.json").read_text())
    s13 = json.loads((art / "acf_residual.json").read_text())
    c10 = next(c for c in s10["cells"] if c["learner"] == "knn" and c["shock_phi"] == 0.9)["arms"][ARM]
    c12 = next(c for c in s12["cells"] if c["shock_phi"] == 0.9)["arms"][ARM]
    c13 = s13["summary"]["arms"][ARM]
    return {name: {"optimism": c["optimism"], "se": c["se"], "n_seeds": n}
            for name, c, n in (("slice10", c10, s10["n_seeds"]), ("slice12", c12, s12["n_seeds"]),
                               ("slice13", c13, s13["n_seeds"]))}


def pool(est: dict[str, dict[str, float]]) -> dict[str, float]:
    w = {k: v["se"] ** -2 for k, v in est.items()}
    W = sum(w.values())
    m = sum(w[k] * est[k]["optimism"] for k in est) / W
    se = W ** -0.5
    q = sum(w[k] * (est[k]["optimism"] - m) ** 2 for k in est)
    df = len(est) - 1
    return {"mean": m, "se": se, "z": m / se, "n_seeds": int(sum(v["n_seeds"] for v in est.values())),
            "q": q, "df": df, "q_p": float(chi2.sf(q, df)), "upper_2se": m + 2 * se,
            "z12_vs13": (est["slice12"]["optimism"] - est["slice13"]["optimism"])
            / math.hypot(est["slice12"]["se"], est["slice13"]["se"])}
