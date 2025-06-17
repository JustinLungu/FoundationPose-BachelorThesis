#!/usr/bin/env python3
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- CONFIG ---
SUMMARY_PATH = os.path.join("results", "summary.json")
OUT_DIR     = os.path.join("results", "comparisons")
METRICS     = {
    "voxel_iou":           {"label": "Voxel IoU",            "better": "↑ higher is better"},
    "chamfer_distance":    {"label": "Chamfer Distance",     "better": "↓ lower is better"},
    "hausdorff_distance":  {"label": "Hausdorff Distance",   "better": "↓ lower is better"},
    "normal_consistency":  {"label": "Normal Consistency",   "better": "↑ higher is better"},
    "mean_curvature_error":{"label": "Mean Curvature Error", "better": "↓ lower is better"},
    "emd":                 {"label": "Earth Mover’s Dist.",  "better": "↓ lower is better"},
}
PALETTES = {
    "dreamfusion": ["#08306b", "#2171b5", "#6baed6"],
    "magic123":    ["#67000d", "#cb181d", "#fb6a4a"],
    "zero123":     ["#00441b", "#238b45", "#74c476"],
}
# --------------

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

def plot_metric(df, metric_key):
    info  = METRICS[metric_key]
    title = info["label"]
    note  = info["better"]

    fig, ax = plt.subplots(figsize=(10,5))

    pivot = df.pivot_table(
        index="object",
        columns=["ai_method", "time"],
        values=metric_key,
        observed=True
    )

    # build colors
    colors = []
    times_sorted = sorted(df["time"].unique())
    for method, time in pivot.columns:
        pal = PALETTES.get(method, ["#888"]*len(times_sorted))
        idx = times_sorted.index(time)
        colors.append(pal[idx % len(pal)])

    pivot.plot(
        kind="bar",
        ax=ax,
        rot=45,
        width=0.8,
        color=colors,
        legend=False
    )

    ax.set_title(f"Comparison of {title}")
    ax.set_ylabel(title)
    ax.set_xlabel("Object")

    # custom legend
    handles = []
    labels = []
    for method in PALETTES:
        for i, time in enumerate(times_sorted):
            handles.append(plt.Rectangle((0,0),1,1, color=PALETTES[method][i]))
            labels.append(f"{method}, {time}")
    leg = ax.legend(handles, labels, title="Method / Time",
                    bbox_to_anchor=(1.05, 1), loc="upper left")

    # add the better‐is note right below the legend
    ax.annotate(
        note,
        xy=(1.02, 0.85),
        xycoords="axes fraction",
        ha="left", va="top"
    )

    plt.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"{metric_key}.png")
    plt.savefig(out_path)
    plt.close(fig)

def main():
    df = load_summary(SUMMARY_PATH)
    df["object"] = pd.Categorical(df["object"], sorted(df["object"].unique()))
    df = df.sort_values(["object", "ai_method", "time"])
    for m in METRICS:
        plot_metric(df, m)
        print(f"Saved comparison plot for {m}")

if __name__ == "__main__":
    main()
