"""Render the CPU latency chart from the recorded benchmark (requires matplotlib)."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    source = Path(__file__).with_name("cpu-20261024.json")
    snapshot = json.loads(source.read_text(encoding="utf-8"))
    models = sorted(snapshot["models"], key=lambda m: m["latency"]["predict_ms"]["p50"])
    values = [model["latency"]["predict_ms"]["p50"] for model in models]
    colors = ["#2563eb" if model["id"] == "trueocr" else "#94a3b8" for model in models]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12})
    fig, ax = plt.subplots(figsize=(11.5, 6.1), dpi=180, facecolor="white")
    fig.subplots_adjust(left=0.34, right=0.96, top=0.81, bottom=0.17)
    fig.text(0.03, 0.94, "CPU line recognition", fontsize=20, weight="bold", color="#0f172a")
    fig.text(0.03, 0.89, "Median latency (p50) · lower is faster", color="#475569")

    ax.barh(range(len(models)), values, color=colors, height=0.62, zorder=3)
    ax.set_yticks(range(len(models)), [model["label"] for model in models])
    ax.invert_yaxis()
    ax.set_xlim(0, 160)
    ax.set_xticks([0, 40, 80, 120, 160])
    ax.set_xlabel("p50 (ms/crop)", color="#475569", labelpad=10)
    ax.tick_params(axis="both", length=0, colors="#475569", pad=10)
    ax.grid(axis="x", color="#e2e8f0", linewidth=0.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for index, (model, value, label) in enumerate(zip(models, values, ax.get_yticklabels())):
        highlight = model["id"] == "trueocr"
        color = "#2563eb" if highlight else "#334155"
        label.set_color(color)
        label.set_fontweight("bold" if highlight else "normal")
        ax.text(value + 2, index, f"{value:.2f}", va="center", color=color,
                fontsize=12, weight="bold" if highlight else "normal")

    protocol = snapshot["protocol"]
    fig.text(0.03, 0.055,
             f"Xeon Platinum 8358P · {protocol['threads']} thread · batch {protocol['batch_size']} · "
             f"{protocol['unique_images']} crops × {protocol['repeats']} runs/model · {snapshot['date']}",
             fontsize=10, color="#475569")
    fig.text(0.03, 0.02, "Recognizer only · excludes model loading and image I/O.",
             fontsize=10, color="#64748b")
    output = source.parents[1] / "assets" / "cpu-latency.png"
    output.parent.mkdir(exist_ok=True)
    fig.savefig(output, facecolor="white")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
