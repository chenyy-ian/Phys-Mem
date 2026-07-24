import csv
import os
from typing import Dict, List

import numpy as np


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


class ExperimentAnalyzer:
    def summarize_method(self, method_name: str, debug_dir: str) -> Dict:
        frame_log = read_csv_dicts(os.path.join(debug_dir, "stableworld_frame_log.csv"))
        action_log = read_csv_dicts(os.path.join(debug_dir, "action_timeline", "action_timeline.csv"))
        fusion_log = read_csv_dicts(os.path.join(debug_dir, "fusion", "fusion_evidence_log.csv"))
        physmem_log = read_csv_dicts(os.path.join(debug_dir, "physmem_decision_timeline.csv"))

        similarities = [safe_float(x.get("similarity")) for x in frame_log]
        runtimes = [safe_float(x.get("runtime_ms")) for x in frame_log]
        replacements = [1.0 if x.get("decision") in {"delete_middle", "physmem_keep", "physmem_refresh"} else 0.0 for x in frame_log]
        matching_points = [safe_float(x.get("matching_points")) for x in frame_log]
        rotation = [safe_float(x.get("rotation_speed")) for x in action_log]
        movement = [safe_float(x.get("movement_speed")) for x in action_log]
        fusion_scores = [safe_float(x.get("unified_memory_score")) for x in fusion_log]

        states = ["KEEP", "REFRESH", "INSERT", "REPLACE", "EVICT"]
        state_counts = {state: 0 for state in states}
        for row in physmem_log:
            state = row.get("memory_state", "")
            if state in state_counts:
                state_counts[state] += 1

        total_states = max(len(physmem_log), 1)
        result = {
            "method": method_name,
            "num_memory_events": len(frame_log),
            "average_similarity": float(np.mean(similarities)) if similarities else 0.0,
            "average_runtime_ms": float(np.mean(runtimes)) if runtimes else 0.0,
            "average_matching_points": float(np.mean(matching_points)) if matching_points else 0.0,
            "memory_replacement_rate": float(np.mean(replacements)) if replacements else 0.0,
            "average_rotation": float(np.mean(rotation)) if rotation else 0.0,
            "average_movement": float(np.mean(movement)) if movement else 0.0,
            "average_fusion_score": float(np.mean(fusion_scores)) if fusion_scores else 0.0,
        }
        for state in states:
            result[f"{state.lower()}_ratio"] = state_counts[state] / total_states
        return result

    def build_tables(self, summaries: List[Dict]) -> Dict[str, List[Dict]]:
        main = []
        runtime = []
        memory = []
        ablation = []
        for row in summaries:
            main.append({
                "method": row["method"],
                "memory_stability": 1.0 - row["memory_replacement_rate"],
                "frame_consistency_proxy": row["average_similarity"],
                "fusion_score": row["average_fusion_score"],
            })
            runtime.append({
                "method": row["method"],
                "average_runtime_ms": row["average_runtime_ms"],
                "average_matching_points": row["average_matching_points"],
            })
            memory.append({
                "method": row["method"],
                "replacement_rate": row["memory_replacement_rate"],
                "keep_ratio": row["keep_ratio"],
                "refresh_ratio": row["refresh_ratio"],
                "insert_ratio": row["insert_ratio"],
                "replace_ratio": row["replace_ratio"],
                "evict_ratio": row["evict_ratio"],
            })
            ablation.append({
                "method": row["method"],
                "similarity": row["average_similarity"],
                "fusion_score": row["average_fusion_score"],
                "rotation": row["average_rotation"],
                "movement": row["average_movement"],
            })
        return {
            "main_results": main,
            "runtime": runtime,
            "memory": memory,
            "ablation": ablation,
        }
