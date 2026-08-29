"""Plot validation accuracy across AttriCite training checkpoints."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_DIR = ROOT / "raw_logs" / "logs"
DEFAULT_OUT_DIR = ROOT / "latex" / "images"
LOG_PATTERN = "qwen3_4b_full_grpo_json_fewshot_kl_n8_max5-*.log"
SELECTED_STEP = 475

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
VAL_RE = re.compile(
    r"step:(?P<step>\d+).*?"
    r"val-core/citeguard/acc/mean@1:np\.float64\((?P<accuracy>[0-9.eE+-]+)\)"
)


def extract_points(path: Path) -> list[tuple[int, float]]:
    text = ANSI_RE.sub("", path.read_text(encoding="utf-8", errors="replace"))
    points = {
        int(match.group("step")): 100.0 * float(match.group("accuracy"))
        for match in VAL_RE.finditer(text)
    }
    return sorted(points.items())


def select_log(log_dir: Path) -> tuple[Path, list[tuple[int, float]]]:
    candidates = []
    for path in log_dir.glob(LOG_PATTERN):
        points = extract_points(path)
        if points:
            candidates.append((len(points), path.stat().st_mtime, path, points))
    if not candidates:
        raise FileNotFoundError(f"No validation metrics found in {log_dir / LOG_PATTERN}")
    _, _, path, points = max(candidates, key=lambda item: (item[0], item[1]))
    return path, points


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, help="Training log to parse; defaults to the most complete run.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    if args.log:
        log_path = args.log.resolve()
        points = extract_points(log_path)
        if not points:
            raise ValueError(f"No validation metrics found in {log_path}")
    else:
        log_path, points = select_log(DEFAULT_LOG_DIR)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "checkpoint_validation.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "validation_accuracy"])
        writer.writerows(points)

    print(f"Parsed {len(points)} checkpoints from {log_path}")
    print(f"Wrote {csv_path}")

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("Install matplotlib to render the PDF and PNG.")
        return

    steps, accuracies = zip(*points)
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot(steps, accuracies, color="#2F6F9F", marker="o", markersize=3.5, linewidth=1.6)

    selected = dict(points).get(SELECTED_STEP)
    if selected is not None:
        ax.scatter([SELECTED_STEP], [selected], color="#F9A602", edgecolor="#222222", zorder=3, s=55)
        ax.annotate(
            f"Selected checkpoint ({SELECTED_STEP})",
            (SELECTED_STEP, selected),
            xytext=(-8, 12), textcoords="offset points", ha="right", fontsize=9,
        )

    ax.set_xlabel("Training step")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_ylim(0, max(70, max(accuracies) + 8))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(args.output_dir / "checkpoint_validation.pdf", bbox_inches="tight")
    fig.savefig(args.output_dir / "checkpoint_validation.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
