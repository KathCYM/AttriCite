"""Summarize completed CiteAlign runs and plot end-to-end accuracy.

Only JSON files containing all 299 retained test instances are included. This lets the
same script be rerun after an in-progress replicate finishes without manually
editing the plotted values.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results_v2" / "results"
OUT_DIR = ROOT / "latex" / "images"
SUMMARY = ROOT / "latex" / "main_results_summary.csv"
GOLD_CSV = ROOT / "training_data_collection" / "splits" / "test_2025_balanced.csv"
EXPECTED_N = 299
T_975 = {3: 4.3026527297}

MODELS = [
    ("Qwen3-4B (base)", RESULTS / "temperature_sampling_validation_test", "base_test_temp0p7_run*.json", 3),
    ("gpt-oss-20b", RESULTS / "together_temperature_sampling", "gpt_oss_20b_test_temp0p7_run*.json", 3),
    ("AttriCite", RESULTS / "temperature_sampling_validation_test", "step475_test_temp0p7_run*.json", 3),
    ("GPT-5.4-mini", RESULTS / "openai_temperature_sampling", "gpt54mini_test_temp0p7_run*.json", 3),
    ("Gemma 4 31B IT", RESULTS / "together_temperature_sampling", "gemma4_31b_test_temp0p7_run*.json", 3),
    ("Claude Haiku 4.5", RESULTS / "anthropic_temperature_sampling", "claude_haiku45_test2025_temp0p7_run1.json", 1),
]

BIOMED_MODELS = [
    ("Qwen3-4B (base)", RESULTS / "biomed_temperature_sampling", "qwen3_4b_base_test_biomed_temp0p7_run*.json", 3),
    ("AttriCite", RESULTS / "biomed_temperature_sampling", "qwen3_4b_step475_test_biomed_temp0p7_run*.json", 3),
]


def load_gold() -> dict[str, str]:
    with GOLD_CSV.open(encoding="utf-8-sig", newline="") as handle:
        return {str(row["id"]): row["target_paper_title"] for row in csv.DictReader(handle)}


def accuracy(path: Path, expected_n: int, gold: dict[str, str] | None = None) -> tuple[int, float] | None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("results", payload) if isinstance(payload, dict) else payload
    if gold is not None:
        rows = [row for row in rows if str(row["id"]) in gold]
    if len(rows) != expected_n:
        print(f"Skipping incomplete run: {path.name} ({len(rows)}/{expected_n})")
        return None
    if gold is None:
        correct = sum(row.get("is_correct") is True for row in rows)
    else:
        correct = sum(
            bool((row.get("selected") or {}).get("title"))
            and SequenceMatcher(
                a=(row.get("selected") or {})["title"].lower(),
                b=gold[str(row["id"])].lower(),
            ).ratio() > 0.8
            for row in rows
        )
    return len(rows), 100.0 * correct / len(rows)


def summarize(models=MODELS, expected_n=EXPECTED_N, gold: dict[str, str] | None = None) -> list[dict[str, object]]:
    output = []
    for label, directory, pattern, maximum_runs in models:
        values = []
        for path in sorted(directory.glob(pattern)):
            result = accuracy(path, expected_n, gold)
            if result is not None:
                values.append(result[1])
            if len(values) == maximum_runs:
                break
        if not values:
            continue
        mean = statistics.mean(values)
        sd = statistics.stdev(values) if len(values) > 1 else math.nan
        ci_half = T_975[len(values)] * sd / math.sqrt(len(values)) if len(values) in T_975 else math.nan
        output.append({
            "model": label,
            "runs": len(values),
            "mean": mean,
            "sd": sd,
            "ci_low": mean - ci_half if not math.isnan(ci_half) else math.nan,
            "ci_high": mean + ci_half if not math.isnan(ci_half) else math.nan,
            "provisional": label == "Gemma 4 31B IT" and len(values) < maximum_runs,
        })
    return output


def main() -> None:
    rows = summarize(gold=load_gold())
    biomed_rows = summarize(BIOMED_MODELS, expected_n=143)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with SUMMARY.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["model", "runs", "mean", "sd", "ci_low", "ci_high", "provisional"],
        )
        writer.writeheader()
        writer.writerows(rows)

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print(f"Wrote {SUMMARY}; install matplotlib to render the figure.")
        return

    display_labels = {
        "Qwen3-4B (base)": "Qwen3-4B\n(base)",
        "gpt-oss-20b": "gpt-oss-20b",
        "AttriCite": "AttriCite\n(ours)",
        "GPT-5.4-mini": "GPT-5.4-mini",
        "Gemma 4 31B IT": "Gemma 4\n31B IT",
        "Claude Haiku 4.5": "Claude\nHaiku 4.5",
    }
    palette = {
        "Qwen3-4B (base)": "#DDD9CE",
        "gpt-oss-20b": "#D5DFE8",
        "AttriCite": "#F9A602",
        "GPT-5.4-mini": "#DDEEF5",
        "Gemma 4 31B IT": "#BFDDB9",
        "Claude Haiku 4.5": "#FFF05A",
    }

    def draw_panel(ax, panel_rows, title, show_ylabel=False):
        labels = [display_labels[str(row["model"])] for row in panel_rows]
        means = [float(row["mean"]) for row in panel_rows]
        ci_half_widths = [
            (float(row["ci_high"]) - float(row["ci_low"])) / 2
            if not math.isnan(float(row["ci_low"]))
            else 0.0
            for row in panel_rows
        ]
        colors = [palette[str(row["model"])] for row in panel_rows]
        bars = ax.bar(
            range(len(panel_rows)), means, yerr=ci_half_widths, capsize=3,
            error_kw={"elinewidth": 1.0, "capthick": 1.0}, width=0.72,
            color=colors, edgecolor="#222222", linewidth=0.9,
        )
        ax.set_title(title, fontsize=11, pad=10)
        if show_ylabel:
            ax.set_ylabel("Accuracy (%)", fontsize=10.5)
        ax.set_ylim(0, 80)
        ax.set_xticks(range(len(panel_rows)), labels, rotation=35, ha="right", rotation_mode="anchor")
        for tick, row in zip(ax.get_xticklabels(), panel_rows):
            if row["model"] == "AttriCite":
                tick.set_fontweight("bold")
        ax.tick_params(axis="both", labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.9)
        ax.spines["bottom"].set_linewidth(0.9)
        for bar, row, ci_half in zip(bars, panel_rows, ci_half_widths):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + ci_half + 1.2,
                f'{float(row["mean"]):.1f}',
                ha="center", va="bottom", fontsize=9.5,
            )

    fig, axes = plt.subplots(
        1, 2, figsize=(9.0, 3.8), sharey=True,
        gridspec_kw={"width_ratios": [2.2, 1.0]},
    )
    draw_panel(axes[0], rows, "(a) CiteAlign", show_ylabel=True)
    draw_panel(axes[1], biomed_rows, "(b) Biomedical transfer")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "main_results.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "main_results.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
