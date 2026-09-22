"""Render the Kaggle comparison from saved predictions, without inference.

Run: uv run --no-project --python 3.12 --with matplotlib python scripts/render_comparison.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parents[1]
SOURCES = [
    ("Base Laya", "challenge_base_results.json", "#7B8794"),
    ("Fine-tuned Laya", "challenge_results.json", "#237A9B"),
    ("JEV 1.13.0", "jev_challenge_results.json", "#C06C32"),
]


def main():
    rows = []
    reference_gold = None
    for name, filename, color in SOURCES:
        data = json.loads((ROOT / "reports" / filename).read_text())
        predictions = data["predictions"]
        gold = {row["id"]: row["gold"] for row in predictions}
        assert len(predictions) == len(gold) == data["n_sessions"] == 24
        assert data["synthetic_only"] is True
        if reference_gold is None:
            reference_gold = gold
        assert gold == reference_gold, "Models must share the same labeled sessions"
        correct = {}
        for metric in ("content", "action"):
            correct[metric] = sum(
                row["answers"][metric]["choice"] == row["gold"][metric]
                for row in predictions
            )
            assert abs(correct[metric] / 24 - data["metrics"][metric]["accuracy"]) < 1e-9
        rows.append((name, color, correct))

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "svg.fonttype": "none",
    })
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.8), sharey=True)
    fig.subplots_adjust(left=0.17, right=0.97, top=0.72, bottom=0.27, wspace=0.16)
    fig.text(0.035, 0.925, "Coding-session decisions on 24 challenge examples",
             fontsize=22, fontweight="bold", color="#172B3A")
    fig.text(0.035, 0.862, "Same 24 separately written synthetic coding sessions · raw model decisions",
             fontsize=12, color="#526371")

    for ax, metric, title, subtitle in zip(
        axes,
        ("content", "action"),
        ("Content accuracy", "Action accuracy"),
        ("benign / suspicious", "allow / block / review"),
    ):
        for index, (name, color, correct) in enumerate(rows):
            value = correct[metric] / 24
            ax.barh(index, value, height=0.53, color=color, zorder=3)
            ax.text(value - 0.018, index, f"{correct[metric]}/24  ·  {value:.1%}",
                    va="center", ha="right", color="white", fontsize=12,
                    fontweight="bold", zorder=4)
        ax.set_title(f"{title}\n{subtitle}", fontsize=13, loc="left", pad=14)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.set_yticks(range(len(rows)), [row[0] for row in rows])
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#DEE5EA", linewidth=0.8)
        ax.tick_params(axis="y", length=0, pad=12)
        ax.tick_params(axis="x", length=0, pad=8, labelsize=10)
        ax.spines["bottom"].set_color("#CBD5DD")
    axes[0].invert_yaxis()
    fig.text(0.035, 0.13, "Small synthetic sample. These results do not establish production safety or general superiority.",
             fontsize=11, color="#526371")
    fig.text(0.035, 0.075, "Source: saved challenge predictions in reports/ · Exact-match accuracy · No confidence-threshold wrapper",
             fontsize=10, color="#526371")
    destination = ROOT / "kaggle" / "assets"
    destination.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        path = destination / f"challenge-comparison.{extension}"
        fig.savefig(path, dpi=180, facecolor="white", metadata={"Creator": "laya-session-guard"})
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
