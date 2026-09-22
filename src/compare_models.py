from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

BASELINE_PATH = RESULTS_DIR / "baseline_results.csv"
HYBRID_DEFAULT_PATH = RESULTS_DIR / "hybrid_attention_results.csv"
HYBRID_SELECTED_PATH = RESULTS_DIR / "hybrid_attention_threshold_selected.csv"

COMPARISON_PATH = RESULTS_DIR / "model_comparison.csv"
FIGURE_PATH = FIGURES_DIR / "model_comparison.png"


COMPARISON_COLUMNS = [
    "model",
    "operating_point",
    "roc_auc",
    "pr_auc",
    "accuracy",
    "precision",
    "recall",
    "f1",
]


def load_results():
    print("Loading baseline results...")
    baseline = pd.read_csv(BASELINE_PATH)

    print("Loading Hybrid Attention default-threshold results...")
    hybrid_default = pd.read_csv(HYBRID_DEFAULT_PATH)

    print("Loading Hybrid Attention validation-selected-threshold results...")
    hybrid_selected = pd.read_csv(HYBRID_SELECTED_PATH)

    # ------------------------------------------------------------------
    # Normalize baseline result schema
    # ------------------------------------------------------------------
    baseline["operating_point"] = (
        "Threshold "
        + baseline["threshold"].astype(float).map(lambda x: f"{x:.2f}")
    )

    # ------------------------------------------------------------------
    # Normalize Hybrid result schemas
    # ------------------------------------------------------------------
    hybrid_default["operating_point"] = (
        "Threshold "
        + hybrid_default["threshold"].astype(float).map(lambda x: f"{x:.2f}")
    )

    hybrid_selected["operating_point"] = (
        "Validation-selected threshold "
        + hybrid_selected["threshold"].astype(float).map(lambda x: f"{x:.2f}")
    )

    # ------------------------------------------------------------------
    # Keep only the common comparison fields
    # ------------------------------------------------------------------
    baseline = baseline[COMPARISON_COLUMNS]
    hybrid_default = hybrid_default[COMPARISON_COLUMNS]
    hybrid_selected = hybrid_selected[COMPARISON_COLUMNS]

    results = pd.concat(
        [
            baseline,
            hybrid_default,
            hybrid_selected,
        ],
        ignore_index=True,
    )

    return results


def create_comparison_figure(results):
    metrics = [
        "roc_auc",
        "pr_auc",
        "accuracy",
        "precision",
        "recall",
        "f1",
    ]

    labels = [
        f"{row.model}\n{row.operating_point}"
        for row in results.itertuples()
    ]

    x_positions = list(range(len(labels)))

    fig, ax = plt.subplots(figsize=(14, 7))

    width = 0.12

    for index, metric in enumerate(metrics):
        values = results[metric].astype(float)

        positions = [
            position
            + (index - (len(metrics) - 1) / 2) * width
            for position in x_positions
        ]

        ax.bar(
            positions,
            values,
            width=width,
            label=metric.upper(),
        )

    ax.set_title("Clean Test-Set Model Comparison")
    ax.set_ylabel("Score")

    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        labels,
        rotation=20,
        ha="right",
    )

    ax.set_ylim(0, 1)

    ax.legend(
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
    )

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.tight_layout()

    fig.savefig(
        FIGURE_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def main():
    results = load_results()

    print("\nModel comparison")
    print("=" * 100)

    print(
        results.to_string(
            index=False
        )
    )

    results.to_csv(
        COMPARISON_PATH,
        index=False,
    )

    create_comparison_figure(results)

    print("\nComparison results saved to:")
    print(COMPARISON_PATH)

    print("\nComparison figure saved to:")
    print(FIGURE_PATH)


if __name__ == "__main__":
    main()