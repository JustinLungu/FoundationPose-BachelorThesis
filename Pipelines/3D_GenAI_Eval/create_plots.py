#!/usr/bin/env python3
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- CONFIG ---
SUMMARY_PATH = os.path.join("results", "summary.json")
OUT_DIR     = os.path.join("results", "comparisons")
METRICS     = [
    "voxel_iou",
    "chamfer_distance",
    "hausdorff_distance",
    "normal_consistency",
    "mean_curvature_error",
    "emd"
]
# --------------


# def load_summary(path):
#     with open(path, "r") as f:
#         data = json.load(f)
#     rows = []
#     for entry in data:
#         md = entry["metadata"]
#         metrics = entry["metrics"]
#         obj = os.path.splitext(md["ai_model"])[0]
#         rows.append({
#             "ai_method": md["category"],   # e.g. zero123, magic123, dreamfusion
#             "time":      md["time"],       # e.g. 10_mins, 30_mins, 1_hour
#             "object":    obj,
#             **{m: metrics[m]["score"] for m in METRICS}
#         })
#     return pd.DataFrame(rows)


def load_summary(path):
    with open(path, "r") as f:
        data = json.load(f)
    rows = []
    for entry in data:
        md = entry["metadata"]
        mets = entry.get("metrics", {})
        obj = os.path.splitext(md["ai_model"])[0]
        row = {
            "ai_method": md["category"],
            "time":      md["time"],
            "object":    obj,
        }
        # for each metric, try to grab its score or set NaN
        for m in METRICS:
            row[m] = mets.get(m, {}).get("score", np.nan)
        rows.append(row)
    return pd.DataFrame(rows)

def plot_metric(df, metric):
    plt.figure(figsize=(10,5))
    ax = plt.gca()

    # pivot into table: index=object, columns=(ai_method, time), values=metric
    pivot = df.pivot_table(
        index="object",
        columns=["ai_method", "time"],
        values=metric
    )
    # draw grouped bars
    pivot.plot(
        kind="bar",
        ax=ax,
        rot=45,
        width=0.8
    )

    ax.set_title(f"Comparison of {metric.replace('_',' ').title()}")
    ax.set_ylabel(metric)
    ax.set_xlabel("Object")
    ax.legend(title="Method / Time", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    # save
    out_path = os.path.join(OUT_DIR, f"{metric}.png")
    plt.savefig(out_path)
    plt.close()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_summary(SUMMARY_PATH)

    # ensure consistent object order
    df["object"] = pd.Categorical(df["object"], sorted(df["object"].unique()))
    df = df.sort_values(["object", "ai_method", "time"])

    for metric in METRICS:
        plot_metric(df, metric)
        print(f"Saved comparison plot for {metric}")

if __name__ == "__main__":
    main()
