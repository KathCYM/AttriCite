import json
import csv
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results_v2" / "results" / "temperature_sampling_validation_test"
OUTPUT_DIR = ROOT / "outputs" / "paired_item_bootstrap"
RUNS = (1, 2, 3)
BOOTSTRAP_SAMPLES = 100_000
PERMUTATION_SAMPLES = 200_000
SEED = 20260828
GOLD_PATH = ROOT / "training_data_collection" / "splits" / "test_2025_balanced.csv"


def load_gold() -> dict[int, str]:
    with GOLD_PATH.open(encoding="utf-8-sig", newline="") as handle:
        return {int(row["id"]): row["target_paper_title"] for row in csv.DictReader(handle)}


def corrected(row: dict, gold: dict[int, str]) -> float:
    selected = row.get("selected") or {}
    title = selected.get("title")
    if not title:
        return 0.0
    return float(SequenceMatcher(a=title.lower(), b=gold[int(row["id"])].lower()).ratio() > 0.8)


def load_run(prefix: str, run: int, gold: dict[int, str]) -> pd.Series:
    path = RESULT_DIR / f"{prefix}_temp0p7_run{run}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = [row for row in payload["results"] if int(row["id"]) in gold]
    series = pd.Series(
        {int(row["id"]): corrected(row, gold) for row in rows},
        name=f"{prefix}_run{run}",
    )
    if len(series) != len(gold):
        raise ValueError(f"Expected {len(gold)} examples in {path}, found {len(series)}")
    return series.sort_index()


def percentile_ci(values: np.ndarray) -> list[float]:
    return [float(x) for x in np.quantile(values, [0.025, 0.975])]


def main() -> None:
    gold = load_gold()
    base = pd.concat([load_run("base_test", run, gold) for run in RUNS], axis=1)
    trained = pd.concat([load_run("step475_test", run, gold) for run in RUNS], axis=1)
    if not base.index.equals(trained.index):
        raise ValueError("Base and trained runs do not share identical test IDs")

    item = pd.DataFrame(
        {
            "base_mean_correct": base.mean(axis=1),
            "trained_mean_correct": trained.mean(axis=1),
        }
    )
    item["paired_difference"] = item["trained_mean_correct"] - item["base_mean_correct"]

    rng = np.random.default_rng(SEED)
    n = len(item)
    base_values = item["base_mean_correct"].to_numpy()
    trained_values = item["trained_mean_correct"].to_numpy()
    difference = item["paired_difference"].to_numpy()

    boot_base = np.empty(BOOTSTRAP_SAMPLES)
    boot_trained = np.empty(BOOTSTRAP_SAMPLES)
    boot_difference = np.empty(BOOTSTRAP_SAMPLES)
    batch = 2_000
    for start in range(0, BOOTSTRAP_SAMPLES, batch):
        stop = min(start + batch, BOOTSTRAP_SAMPLES)
        indices = rng.integers(0, n, size=(stop - start, n))
        boot_base[start:stop] = base_values[indices].mean(axis=1)
        boot_trained[start:stop] = trained_values[indices].mean(axis=1)
        boot_difference[start:stop] = difference[indices].mean(axis=1)

    observed_difference = float(difference.mean())
    nonzero_difference = difference[difference != 0]
    extreme = 0
    batch = 5_000
    for start in range(0, PERMUTATION_SAMPLES, batch):
        stop = min(start + batch, PERMUTATION_SAMPLES)
        signs = rng.choice((-1.0, 1.0), size=(stop - start, len(nonzero_difference)))
        permuted = (signs * nonzero_difference).sum(axis=1) / n
        extreme += int(np.sum(np.abs(permuted) >= abs(observed_difference)))
    permutation_p = float((extreme + 1) / (PERMUTATION_SAMPLES + 1))
    summary = {
        "test_items": n,
        "reported_runs": list(RUNS),
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "permutation_samples": PERMUTATION_SAMPLES,
        "seed": SEED,
        "base_accuracy": float(base_values.mean()),
        "base_item_bootstrap_95_ci": percentile_ci(boot_base),
        "trained_accuracy": float(trained_values.mean()),
        "trained_item_bootstrap_95_ci": percentile_ci(boot_trained),
        "paired_difference": observed_difference,
        "paired_difference_item_bootstrap_95_ci": percentile_ci(boot_difference),
        "paired_sign_flip_permutation_p": permutation_p,
        "items_trained_higher": int((difference > 0).sum()),
        "items_base_higher": int((difference < 0).sum()),
        "items_tied": int((difference == 0).sum()),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    item.reset_index(names="id").to_csv(OUTPUT_DIR / "per_item_correctness.csv", index=False)
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
