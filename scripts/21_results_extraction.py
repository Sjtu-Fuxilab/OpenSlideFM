#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenSlideFM - Results Extraction
============================================================

This script is part of the OpenSlideFM pipeline for computational pathology.

Paper: "OpenSlideFM: A Resource-Efficient Foundation Model for 
        Computational Pathology on Whole Slide Images"

Authors: Sanwal Ahmad Zafar, Wei Qin
Institution: Shanghai Jiao Tong University

License: Apache 2.0
"""

"""
Fixed Diagnostic - handles missing variants gracefully
Run this in Jupyter and paste the output
"""
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path

def convert_numpy(obj):
    """Convert numpy types to native Python"""
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(item) for item in obj]
    return obj

WORKSPACE = Path(os.environ.get("WORKSPACE", "./workspace")))
RESULTS = WORKSPACE / "results" / "ablations_complete" / "tcga"

metrics = {}

# ==============================================================================
# 1. ABLATION STUDY RESULTS
# ==============================================================================
print("="*80)
print("EXTRACTING ABLATION STUDY RESULTS")
print("="*80)

ablation_files = {
    "scale": "scale_ablation.csv",
    "pooling": "pooling_ablation.csv",
    "token_budget": "token_budget_ablation.csv",
    "fusion": "feature_fusion_ablation.csv",
    "classifier": "classifier_ablation.csv"
}

metrics["ablation"] = {}

for component, filename in ablation_files.items():
    filepath = RESULTS / filename
    if filepath.exists():
        df = pd.read_csv(filepath)
        print(f"\n{component.upper()}:")
        print(df.to_string(index=False))
        metrics["ablation"][component] = convert_numpy(df.to_dict('records'))
    else:
        print(f"\n{component.upper()}: FILE NOT FOUND")
        metrics["ablation"][component] = None

# ==============================================================================
# 2. PRIMARY RESULTS (BEST MODEL)
# ==============================================================================
print("\n" + "="*80)
print("EXTRACTING PRIMARY RESULTS")
print("="*80)

# Find the best configuration from scale ablation
if metrics["ablation"]["scale"]:
    scale_df = pd.DataFrame(metrics["ablation"]["scale"])
    
    print(f"\nAvailable variants in scale ablation:")
    print(scale_df["variant"].unique())
    
    # Try to find the best performing variant based on accuracy
    if "metric" in scale_df.columns and "mean" in scale_df.columns:
        acc_rows = scale_df[scale_df["metric"] == "accuracy"]
        if not acc_rows.empty:
            best_idx = acc_rows["mean"].idxmax()
            best_row = acc_rows.loc[best_idx]
            
            print(f"\nBest Model ({best_row['variant']}):")
            print(f"  Accuracy: {best_row['mean']:.4f} ± {best_row['std']:.4f}")
            
            metrics["primary"] = {
                "best_variant": str(best_row["variant"]),
                "accuracy": {"mean": float(best_row["mean"]), "std": float(best_row["std"])}
            }
            
            # Try to get other metrics for this variant
            for metric_name in ["auc", "balanced_accuracy", "f1_macro"]:
                metric_row = scale_df[
                    (scale_df["variant"] == best_row["variant"]) & 
                    (scale_df["metric"] == metric_name)
                ]
                if not metric_row.empty:
                    metrics["primary"][metric_name] = {
                        "mean": float(metric_row.iloc[0]["mean"]),
                        "std": float(metric_row.iloc[0]["std"])
                    }
                else:
                    metrics["primary"][metric_name] = None
        else:
            print("No accuracy metric found in scale ablation")
            metrics["primary"] = None
    else:
        print("Unexpected scale ablation format")
        metrics["primary"] = None
else:
    print("No scale ablation data available")
    metrics["primary"] = None

# ==============================================================================
# 3. PER-CANCER-TYPE RESULTS
# ==============================================================================
print("\n" + "="*80)
print("EXTRACTING PER-CANCER-TYPE RESULTS")
print("="*80)

per_cancer_file = RESULTS / "per_cancer_results.csv"
if per_cancer_file.exists():
    df = pd.read_csv(per_cancer_file)
    print(df.to_string(index=False))
    metrics["per_cancer"] = convert_numpy(df.to_dict('records'))
else:
    print("per_cancer_results.csv NOT FOUND")
    metrics["per_cancer"] = None

# ==============================================================================
# 4. CONFUSION MATRIX
# ==============================================================================
print("\n" + "="*80)
print("EXTRACTING CONFUSION MATRIX")
print("="*80)

confusion_file = RESULTS / "confusion_matrix.csv"
if confusion_file.exists():
    df = pd.read_csv(confusion_file, index_col=0)
    print(df.to_string())
    metrics["confusion_matrix"] = {
        "matrix": convert_numpy(df.values.tolist()),
        "labels": list(df.columns)
    }
else:
    print("confusion_matrix.csv NOT FOUND")
    metrics["confusion_matrix"] = None

# ==============================================================================
# 5. DATASET STATISTICS
# ==============================================================================
print("\n" + "="*80)
print("EXTRACTING DATASET STATISTICS")
print("="*80)

# Check for dataset stats files
stats_files = {
    "tcga": RESULTS.parent.parent / "manifest" / "tcga_stats.csv",
    "cam16": RESULTS.parent.parent / "manifest" / "cam16_stats.csv",
    "cam17": RESULTS.parent.parent / "manifest" / "cam17_stats.csv",
    "panda": RESULTS.parent.parent / "manifest" / "panda_stats.csv"
}

metrics["datasets"] = {}
for dataset, filepath in stats_files.items():
    if filepath.exists():
        df = pd.read_csv(filepath)
        print(f"\n{dataset.upper()}:")
        print(df.to_string(index=False))
        metrics["datasets"][dataset] = convert_numpy(df.to_dict('records'))
    else:
        print(f"\n{dataset.upper()}: FILE NOT FOUND")
        metrics["datasets"][dataset] = None

# ==============================================================================
# 6. COMPUTATIONAL REQUIREMENTS
# ==============================================================================
print("\n" + "="*80)
print("COMPUTATIONAL REQUIREMENTS")
print("="*80)

timing_file = RESULTS / "timing_stats.json"
if timing_file.exists():
    with open(timing_file) as f:
        metrics["computational"] = json.load(f)
    print(json.dumps(metrics["computational"], indent=2))
else:
    print("timing_stats.json NOT FOUND")
    metrics["computational"] = None

# ==============================================================================
# 7. CHECK FOR ADDITIONAL METRIC FILES
# ==============================================================================
print("\n" + "="*80)
print("CHECKING FOR ADDITIONAL FILES")
print("="*80)

# List all CSV files in results directory
csv_files = list(RESULTS.glob("*.csv"))
print(f"\nAll CSV files in {RESULTS}:")
for f in csv_files:
    print(f"  - {f.name}")

# List all JSON files
json_files = list(RESULTS.glob("*.json"))
print(f"\nAll JSON files:")
for f in json_files:
    print(f"  - {f.name}")

# ==============================================================================
# SAVE & DISPLAY JSON
# ==============================================================================
print("\n" + "="*80)
print("SAVING COMPREHENSIVE METRICS")
print("="*80)

output_file = WORKSPACE / "comprehensive_metrics.json"
with open(output_file, 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"\n✅ Saved to: {output_file}")
print(f"\n📋 PASTE THIS JSON BACK TO CLAUDE:\n")
print("="*80)
print(json.dumps(metrics, indent=2))
print("="*80)