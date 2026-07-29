import argparse
import csv
import json
import os
from collections import Counter
from typing import Dict, List

import numpy as np


STATES = ["KEEP", "REFRESH", "INSERT", "REPLACE", "EVICT"]


def read_csv_dicts(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def safe_float(value, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def find_debug_dir(method_dir: str) -> str:
    if os.path.basename(method_dir) == "stableworld_debug":
        return method_dir
    candidate = os.path.join(method_dir, "stableworld_debug")
    return candidate if os.path.isdir(candidate) else method_dir


def strategy_distribution(rows: List[Dict]) -> Dict[str, float]:
    counts = Counter(row.get("memory_state", "") for row in rows)
    total = max(sum(counts.get(state, 0) for state in STATES), 1)
    return {state: counts.get(state, 0) / total for state in STATES}


def strategy_transition_smoothness(rows: List[Dict]) -> float:
    states = [row.get("memory_state", "") for row in rows if row.get("memory_state", "") in STATES]
    if len(states) <= 1:
        return 1.0 if states else 0.0
    transitions = sum(1 for prev, cur in zip(states, states[1:]) if prev != cur)
    return float(1.0 - transitions / (len(states) - 1))


def compute_memory_metrics(debug_dir: str) -> Dict:
    debug_dir = find_debug_dir(debug_dir)
    frame_log = read_csv_dicts(os.path.join(debug_dir, "stableworld_frame_log.csv"))
    fusion_log = read_csv_dicts(os.path.join(debug_dir, "fusion", "fusion_evidence_log.csv"))
    physmem_log = read_csv_dicts(os.path.join(debug_dir, "physmem_decision_timeline.csv"))
    experiment_records = []
    exp_root = os.path.join(debug_dir, "experiment_tracking", "experiments")
    if os.path.isdir(exp_root):
        for root, _, files in os.walk(exp_root):
            if "experiment_records.csv" in files:
                experiment_records.extend(read_csv_dicts(os.path.join(root, "experiment_records.csv")))

    rows_for_state = physmem_log or frame_log
    similarities = [safe_float(row.get("similarity")) for row in frame_log if row.get("similarity") not in (None, "")]
    runtimes = [safe_float(row.get("runtime_ms") or row.get("runtime")) for row in frame_log if row.get("runtime_ms") or row.get("runtime")]
    fusion_scores = [safe_float(row.get("unified_memory_score")) for row in fusion_log if row.get("unified_memory_score") not in (None, "")]

    decision_rows = frame_log or experiment_records
    replace_like = {"delete_middle", "delete_oldest", "physmem_refresh", "physmem_replace", "physmem_evict"}
    memory_replacements = [1.0 if row.get("decision") in replace_like else 0.0 for row in decision_rows]
    total_events = len(decision_rows)

    dist = strategy_distribution(rows_for_state)
    mrr = float(np.mean(memory_replacements)) if memory_replacements else 0.0
    mss = 1.0 - mrr
    mcs = float(np.mean(similarities)) if similarities else 0.0
    afs = float(np.mean(fusion_scores)) if fusion_scores else 0.0
    pcs = float(np.mean([mcs, afs])) if fusion_scores and similarities else max(mcs, afs)

    physmem_states = [row.get("memory_state", "") for row in physmem_log]
    frr_candidates = [state in {"REFRESH", "REPLACE", "EVICT"} for state in physmem_states]
    frr = float(np.mean(frr_candidates)) if frr_candidates else mrr

    return {
        "Runtime": float(np.mean(runtimes)) if runtimes else 0.0,
        "MCS": mcs,
        "PCS": pcs,
        "FRR": frr,
        "MSS": mss,
        "MRR": mrr,
        "AFS": afs,
        "MemoryReplacementRate": mrr,
        "MemoryStabilityScore": mss,
        "StrategyDistribution": dist,
        "StrategyTransitionSmoothness": strategy_transition_smoothness(rows_for_state),
        "NumEvents": total_events,
    }


def main():
    parser = argparse.ArgumentParser(description="Compute Stage15 memory scheduling metrics from existing logs.")
    parser.add_argument("--debug_dir", required=True, help="Method output folder or stableworld_debug folder")
    parser.add_argument("--output", default="evaluation/results/memory_metrics.json")
    args = parser.parse_args()

    result = compute_memory_metrics(args.debug_dir)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
