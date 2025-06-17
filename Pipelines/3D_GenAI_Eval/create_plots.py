#!/usr/bin/env python3
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pipeline.config import (
    IOU_THRESHOLDS,
    CHAMFER_THRESHOLDS,
    HAUSDORFF_THRESHOLDS,
    NORMAL_CONSISTENCY_THRESHOLDS,
    MEAN_CURVATURE_THRESHOLDS,
    EMD_THRESHOLDS,
)

# --- CONFIG ---
SUMMARY_PATH = os.path.join("results", "summary.json")
OUT_DIR     = os.path.join("results", "comparisons")

# which thresholds map to which metric
THRESHOLDS = {
    "voxel_iou":            IOU_THRESHOLDS,
    "chamfer_distance":     CHAMFER_THRESHOLDS,
    "hausdorff_distance":   HAUSDORFF_THRESHOLDS,
    "normal_consistency":   NORMAL_CONSISTENCY_THRESHOLDS,
    "mean_curvature_error": MEAN_CURVATURE_THRESHOLDS,
    "emd":                  EMD_THRESHOLDS,
}

METRICS = {
    "voxel_iou": {
        "label":  "Voxel IoU",
        "better": "↑ higher is better",
        "unit":   "(unitless fraction)"
    },
    "chamfer_distance": {
        "label":  "Chamfer Distance",
        "better": "↓ lower is better",
        "unit":   "(normalized - mm²/diag²)"
    },
    "hausdorff_distance": {
        "label":  "Hausdorff Distance",
        "better": "↓ lower is better",
        "unit":   "(normalized - mm/diag)"
    },
    "normal_consistency": {
        "label":  "Normal Consistency",
        "better": "↑ higher is better",
        "unit":   "(cosine similarity)"
    },
    "mean_curvature_error": {
        "label":  "Mean Curvature Error",
        "better": "↓ lower is better",
        "unit":   "(dimensionless proxy)"
    },
    "emd": {
        "label":  "Earth Mover’s Dist.",
        "better": "↓ lower is better",
        "unit":   "(normalized - mm/diag)"
    },
}

PALETTES = {
    "dreamfusion": ["#4292c6", "#6baed6", "#9ecae1"],
    "magic123":    ["#ef3b2c", "#fb6a4a", "#fcae91"],
    "zero123":     ["#41ab5d", "#74c476", "#a1d99b"],
}

_time_to_minutes = {
    "10_mins": 10,
    "30_mins": 30,
    "1_hour":  60,
}

# -----------------------------------------------------------------------------

def load_summary(path):
    with open(path, "r") as f:
        data = json.load(f)
    rows = []
    for entry in data:
        md   = entry["metadata"]
        mets = entry.get("metrics", {})
        obj  = os.path.splitext(md["ai_model"])[0]
        row  = {
            "ai_method": md["category"],
            "time":      md["time"],
            "object":    obj,
        }
        for m in METRICS:
            row[m] = mets.get(m, {}).get("score", np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_metric(df, metric_key, out_dir):
    """Plot one metric over methods/times, and save into out_dir."""

    if df.empty or df[metric_key].dropna().empty:
        return


    info    = METRICS[metric_key]
    title   = info["label"]
    note    = info["better"]
    unit    = info.get("unit", "")
    cuts    = THRESHOLDS[metric_key]

    fig, ax = plt.subplots(figsize=(10,5))
    fig.subplots_adjust(right=0.80)

    # pivot
    pivot = df.pivot_table(
        index="object",
        columns=["ai_method", "time"],
        values=metric_key,
        observed=True
    )

    # reorder columns → (method,10_mins),(method,30_mins),(method,1_hour)
    cols_sorted = sorted(
        pivot.columns,
        key=lambda mt: (mt[0], _time_to_minutes.get(mt[1], float("inf")))
    )
    pivot = pivot[cols_sorted]

    # sorted times for color indexing
    times_sorted = sorted(df["time"].unique(), key=lambda t: _time_to_minutes.get(t, float("inf")))

    # build colors
    colors = []
    for method, time in pivot.columns:
        pal = PALETTES.get(method, ["#888"]*len(times_sorted))
        idx = times_sorted.index(time)
        colors.append(pal[idx])

    # bar plot
    pivot.plot(
        kind="bar", ax=ax, rot=45, width=0.8,
        color=colors, legend=False
    )

    # threshold lines (only good & bad)
    for label in ("good","bad"):
        if label in cuts:
            val = cuts[label]
            ax.axhline(val, linestyle="--", color="black", linewidth=1)
            ax.text(
                0.99, val,
                f"{label} @ {val:.2f}",
                transform=ax.get_yaxis_transform(),
                ha="right", va="bottom",
                fontsize="x-small"
            )

    # titles & axis
    ax.set_title(f"Comparison of {title}")
    ax.set_ylabel(title)
    ax.set_xlabel("Object")

    # legend
    handles, labels = [], []
    for method in PALETTES:
        for i, time in enumerate(times_sorted):
            handles.append(plt.Rectangle((0,0),1,1,color=PALETTES[method][i]))
            labels.append(f"{method}, {time}")
    ax.legend(
        handles, labels, title="Method / Time",
        bbox_to_anchor=(1.02,0.98), loc="upper left", borderaxespad=0
    )

    # better-is & unit
    ax.annotate(
        note,
        xy=(0.82,0.35), xycoords="figure fraction",
        ha="left", va="top", fontsize="small"
    )
    ax.annotate(
        unit,
        xy=(0.82,0.30), xycoords="figure fraction",
        ha="left", va="top", fontsize="small"
    )

    os.makedirs(out_dir, exist_ok=True)
    plt.tight_layout()
    fn = f"{metric_key}.png"
    fig.savefig(os.path.join(out_dir, fn), bbox_inches="tight")
    plt.close(fig)


def main():
    df = load_summary(SUMMARY_PATH)
    df["object"] = pd.Categorical(df["object"], sorted(df["object"].unique()))
    df = df.sort_values(["object","ai_method","time"])

    # 1) global comparisons (one big grid per metric)
    for m in METRICS:
        plot_metric(df, m, OUT_DIR)
        print(f"Saved global comparison for {m}")

    # 2) per-object comparisons
    for obj in df["object"].unique():
        sub = df[df["object"] == obj]
        obj_dir = os.path.join(OUT_DIR, obj)
        for m in METRICS:
            plot_metric(sub, m, obj_dir)
        print(f"Saved per-object plots for '{obj}'")

if __name__ == "__main__":
    main()
