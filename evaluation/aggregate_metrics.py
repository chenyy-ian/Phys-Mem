import argparse
import csv
import json
import os
from typing import Dict

try:
    from .memory_metrics import compute_memory_metrics
except ImportError:
    from memory_metrics import compute_memory_metrics


def load_json(path: str) -> Dict:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def flatten_strategy_distribution(value) -> str:
    if not isinstance(value, dict):
        return ""
    return ";".join(f"{key}:{value.get(key, 0):.4f}" for key in sorted(value))


def pick(data: Dict, *names, default=""):
    for name in names:
        if name in data:
            return data[name]
    metrics = data.get("metrics")
    if isinstance(metrics, dict):
        for name in names:
            if name in metrics:
                return metrics[name]
    results = data.get("results")
    if isinstance(results, dict):
        for name in names:
            if name in results:
                return results[name]
    return default


def build_row(method: str, debug_dir: str, vbench: Dict, stream: Dict, dover: Dict) -> Dict:
    memory = compute_memory_metrics(debug_dir) if debug_dir else {}
    return {
        "Method": method,
        "Runtime": memory.get("Runtime", 0.0),
        "BackgroundConsistency": pick(vbench, "background_consistency", "Background Consistency"),
        "TemporalFlickering": pick(vbench, "temporal_flickering", "Temporal Flickering"),
        "MotionSmoothness": pick(vbench, "motion_smoothness", "Motion Smoothness"),
        "SubjectConsistency": pick(vbench, "subject_consistency", "Subject Consistency"),
        "ImagingQuality": pick(vbench, "imaging_quality", "Imaging Quality"),
        "SpatialScore": pick(stream, "spatial_score", "Spatial Score"),
        "TemporalScore": pick(stream, "temporal_score", "Temporal Score"),
        "TechnicalQuality": pick(dover, "technical_quality", "Technical Quality"),
        "AestheticQuality": pick(dover, "aesthetic_quality", "Aesthetic Quality"),
        "OverallQuality": pick(dover, "overall_quality", "Overall Quality"),
        "MemoryReplacementRate": memory.get("MemoryReplacementRate", 0.0),
        "MemoryStabilityScore": memory.get("MemoryStabilityScore", 0.0),
        "StrategyDistribution": flatten_strategy_distribution(memory.get("StrategyDistribution", {})),
        "StrategyTransitionSmoothness": memory.get("StrategyTransitionSmoothness", 0.0),
        "MCS": memory.get("MCS", 0.0),
        "PCS": memory.get("PCS", 0.0),
        "FRR": memory.get("FRR", 0.0),
        "MSS": memory.get("MSS", 0.0),
        "MRR": memory.get("MRR", 0.0),
        "AFS": memory.get("AFS", 0.0),
    }


def main():
    parser = argparse.ArgumentParser(description="Aggregate Stage15 public and PhysMem metrics into paper_results.csv.")
    parser.add_argument("--method", required=True)
    parser.add_argument("--debug_dir", required=True, help="Method output folder or stableworld_debug folder")
    parser.add_argument("--vbench_json", default="")
    parser.add_argument("--stream_json", default="")
    parser.add_argument("--dover_json", default="")
    parser.add_argument("--output", default="evaluation/results/paper_results.csv")
    args = parser.parse_args()

    row = build_row(
        method=args.method,
        debug_dir=args.debug_dir,
        vbench=load_json(args.vbench_json),
        stream=load_json(args.stream_json),
        dover=load_json(args.dover_json),
    )
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    exists = os.path.exists(args.output)
    with open(args.output, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
