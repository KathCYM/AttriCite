import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "outputs" / "paired_item_bootstrap" / "per_item_correctness.csv"
OUTPUT = ROOT / "outputs" / "paired_item_bootstrap" / "bayesian_paired_summary.json"
DRAWS = 200_000
SEED = 20260828
ROPE = 0.01


def main() -> None:
    item = pd.read_csv(INPUT)
    differences = item["paired_difference"].to_numpy(dtype=float)
    rng = np.random.default_rng(SEED)
    posterior = np.empty(DRAWS, dtype=float)

    batch_size = 2_000
    for start in range(0, DRAWS, batch_size):
        stop = min(start + batch_size, DRAWS)
        weights = rng.dirichlet(np.ones(len(differences)), size=stop - start)
        posterior[start:stop] = weights @ differences

    lower, median, upper = np.quantile(posterior, [0.025, 0.5, 0.975])
    summary = {
        "method": "paired Bayesian bootstrap over test items",
        "item_statistic": "trained minus base correctness averaged across reported runs 1-3",
        "test_items": len(differences),
        "posterior_draws": DRAWS,
        "seed": SEED,
        "posterior_mean_difference": float(posterior.mean()),
        "posterior_median_difference": float(median),
        "credible_interval_95": [float(lower), float(upper)],
        "probability_improvement": float(np.mean(posterior > 0)),
        "probability_improvement_gt_1pp": float(np.mean(posterior > 0.01)),
        "probability_improvement_gt_5pp": float(np.mean(posterior > 0.05)),
        "rope": [-ROPE, ROPE],
        "probability_in_rope": float(np.mean(np.abs(posterior) <= ROPE)),
    }
    OUTPUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
