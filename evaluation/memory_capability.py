"""P0: Memory Capability metrics (PhysMem custom, log-only).

All metrics here are built on the PhysMem framework's own logs
(stableworld_frame_log.csv), i.e. they are CUSTOM metrics:
  - First Visit Hit Rate
  - Retrieval Accuracy
  - Memory Transition Stability (MTS) across repeated runs
  - Anchor Drift (mean / max / locomotion-segment)

No video frames are required.
"""

import argparse
import csv
import json
import os


RETRIEVE_MODES = {"retrieve_window", "soft_reuse_window"}


def read_frame_log(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def safe_float(value, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def per_run_metrics(rows: list[dict]) -> dict:
    retrieves = [r for r in rows if r.get("memory_selection_mode") in RETRIEVE_MODES]
    first_visit = [
        r for r in retrieves if safe_float(r.get("best_candidate_first_visit")) >= 0.999
    ]
    revisits = [r for r in rows if r.get("view_state") == "Revisit"]
    loop_closures = [r for r in rows if r.get("loop_closure_detected") == "True"]

    lags = [
        safe_float(r.get("anchor_lag"), -1.0)
        for r in rows
        if r.get("anchor_lag") not in (None, "", "-1.0")
    ]
    loco_lags = [
        safe_float(r.get("anchor_lag"), -1.0)
        for r in rows
        if r.get("anchor_lag") not in (None, "", "-1.0")
        and "Locomotion" in (r.get("action_mode") or "")
    ]

    return {
        "retrieve_count": len(retrieves),
        "first_visit_hit_rate": (len(first_visit) / len(retrieves)) if retrieves else 0.0,
        "first_visit_hits": len(first_visit),
        "retrieval_accuracy": (len(loop_closures) / len(revisits)) if revisits else 0.0,
        "revisit_count": len(revisits),
        "loop_closure_count": len(loop_closures),
        "anchor_drift_mean": (sum(lags) / len(lags)) if lags else 0.0,
        "anchor_drift_max": max(lags) if lags else 0.0,
        "anchor_drift_loco_mean": (sum(loco_lags) / len(loco_lags)) if loco_lags else 0.0,
    }


def memory_transition_stability(all_runs: list[list[dict]]) -> dict:
    """MTS: per-frame agreement of window_after across repeated runs.

    Also reports mean pairwise Jaccard of the whole window sequence as a
    secondary, more fine-grained agreement measure.
    """
    by_frame: dict[str, list[str]] = {}
    for rows in all_runs:
        for r in rows:
            by_frame.setdefault(r.get("frame_index", ""), []).append(r.get("window_after", ""))

    total_frames = len(by_frame)
    full_agree = 0
    for ws in by_frame.values():
        if len(set(ws)) == 1:
            full_agree += 1

    # mean pairwise Jaccard of window-id sets between runs
    def _window_set(w: str) -> set:
        return set(w.split()) if w else set()

    jaccards = []
    frame_ids = list(by_frame.keys())
    for fid in frame_ids:
        ws = by_frame[fid]
        for i in range(len(ws)):
            for j in range(i + 1, len(ws)):
                a = _window_set(ws[i])
                b = _window_set(ws[j])
                union = a | b
                if union:
                    jaccards.append(len(a & b) / len(union))

    return {
        "total_frames": total_frames,
        "full_agreement_frames": full_agree,
        "window_agreement_rate": (full_agree / total_frames) if total_frames else 0.0,
        "mean_pairwise_jaccard": (sum(jaccards) / len(jaccards)) if jaccards else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="P0 PhysMem Memory Capability metrics (log-only, custom)")
    parser.add_argument("--run_dirs", nargs="+", required=True, help="Directories each containing stableworld_frame_log.csv (one per repeated run)")
    parser.add_argument("--output", default="evaluation/results/memory_capability.json")
    args = parser.parse_args()

    all_runs = []
    per_run = []
    for run_dir in args.run_dirs:
        log_path = os.path.join(run_dir, "stableworld_frame_log.csv")
        if not os.path.exists(log_path):
            print(f"[warn] missing {log_path}, skipped")
            continue
        rows = read_frame_log(log_path)
        all_runs.append(rows)
        per_run.append(per_run_metrics(rows))

    if not per_run:
        raise SystemExit("No valid run logs found.")

    mean = {
        key: sum(r[key] for r in per_run) / len(per_run)
        for key in per_run[0]
    }
    result = {
        "per_run": per_run,
        "mean": mean,
        "memory_transition_stability": memory_transition_stability(all_runs),
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
