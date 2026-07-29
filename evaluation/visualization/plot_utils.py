import csv
import json
import os
from collections import Counter
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


STRATEGIES = ["KEEP", "REFRESH", "INSERT", "REPLACE", "EVICT"]
STRATEGY_COLORS = {
    "KEEP": "#2ca02c",
    "REFRESH": "#1f77b4",
    "INSERT": "#ff7f0e",
    "REPLACE": "#9467bd",
    "EVICT": "#d62728",
    "UNKNOWN": "#7f7f7f",
}
WINDOW_COLORS = {
    "Retained": "#2ca02c",
    "Deleted": "#d62728",
    "Inserted": "#1f77b4",
}


def apply_paper_style():
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "lines.linewidth": 2.5,
        "savefig.bbox": "tight",
    })


def finalize_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.2, linewidth=0.8)


def ensure_output_dirs(output_dir: str) -> Dict[str, str]:
    paths = {
        "root": output_dir,
        "paper": os.path.join(output_dir, "paper_figures"),
        "preview": os.path.join(output_dir, "preview"),
        "svg": os.path.join(output_dir, "svg"),
    }
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    return paths


def save_figure(fig, output_dirs: Dict[str, str], name: str):
    fig.savefig(os.path.join(output_dirs["paper"], f"{name}.pdf"))
    fig.savefig(os.path.join(output_dirs["svg"], f"{name}.svg"))
    fig.savefig(os.path.join(output_dirs["preview"], f"{name}.png"), dpi=300)
    plt.close(fig)


def read_records(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def safe_float(value, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value, default: int = 0) -> int:
    try:
        if value in (None, ""):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def frame_ids(records: List[Dict]) -> List[int]:
    return [safe_int(row.get("FrameID"), idx) for idx, row in enumerate(records)]


def strategy_of(row: Dict) -> str:
    value = (row.get("FusionDecision") or row.get("Decision") or "").upper()
    for strategy in STRATEGIES:
        if strategy in value:
            return strategy
    return "UNKNOWN"


def strategy_indices(records: List[Dict]) -> List[int]:
    return [STRATEGIES.index(strategy_of(row)) if strategy_of(row) in STRATEGIES else -1 for row in records]


def parse_memory_state(value) -> List[int]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return [safe_int(x) for x in value]
    text = str(value).strip()
    if text.startswith("[") and text.endswith("]"):
        try:
            return [safe_int(x) for x in json.loads(text)]
        except Exception:
            pass
    return [safe_int(part) for part in text.replace(",", " ").split() if part.strip()]


def memory_windows(records: List[Dict]) -> List[List[int]]:
    return [parse_memory_state(row.get("MemoryState")) for row in records]


def numeric_series(records: List[Dict], field: str) -> List[float]:
    return [safe_float(row.get(field), np.nan) for row in records]


def normalize(values: Iterable[float]) -> List[float]:
    arr = np.array([x for x in values], dtype=np.float32)
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        return [0.0 for _ in arr]
    lo, hi = float(np.min(valid)), float(np.max(valid))
    if abs(hi - lo) < 1e-8:
        return [0.0 if np.isnan(x) else 1.0 for x in arr]
    return [0.0 if np.isnan(x) else float((x - lo) / (hi - lo)) for x in arr]


def compute_memory_lifetimes(records: List[Dict]) -> Dict[int, int]:
    first_seen: Dict[int, int] = {}
    last_seen: Dict[int, int] = {}
    frames = frame_ids(records)
    for frame, window in zip(frames, memory_windows(records)):
        for memory_id in window:
            first_seen.setdefault(memory_id, frame)
            last_seen[memory_id] = frame
    return {memory_id: last_seen[memory_id] - first_seen[memory_id] for memory_id in first_seen}


def strategy_distribution(records: List[Dict]) -> Dict[str, float]:
    counts = Counter(strategy_of(row) for row in records)
    total = max(sum(counts.get(strategy, 0) for strategy in STRATEGIES), 1)
    return {strategy: counts.get(strategy, 0) / total for strategy in STRATEGIES}


def transition_matrix(records: List[Dict]) -> np.ndarray:
    matrix = np.zeros((len(STRATEGIES), len(STRATEGIES)), dtype=np.float32)
    states = [strategy_of(row) for row in records if strategy_of(row) in STRATEGIES]
    for previous, current in zip(states, states[1:]):
        matrix[STRATEGIES.index(previous), STRATEGIES.index(current)] += 1.0
    row_sums = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums > 0)


def write_summary(records: List[Dict], output_dir: str):
    lifetimes = list(compute_memory_lifetimes(records).values())
    summary = {
        "Strategy Ratio": strategy_distribution(records),
        "Average Runtime": float(np.nanmean(numeric_series(records, "Runtime"))) if records else 0.0,
        "Average Similarity": float(np.nanmean(numeric_series(records, "Similarity"))) if records else 0.0,
        "Average Geometry": float(np.nanmean(numeric_series(records, "GeometryScore"))) if records else 0.0,
        "Average Semantic": float(np.nanmean(numeric_series(records, "SemanticScore"))) if records else 0.0,
        "Average Fusion Score": float(np.nanmean(numeric_series(records, "FusionWeight"))) if records else 0.0,
        "Average Confidence": float(np.nanmean(numeric_series(records, "Confidence"))) if records else 0.0,
        "Average Policy Confidence": float(np.nanmean(numeric_series(records, "PolicyConfidence"))) if records else 0.0,
        "Average Decision Confidence": float(np.nanmean(numeric_series(records, "DecisionConfidence"))) if records else 0.0,
        "Memory Lifetime Mean": float(np.mean(lifetimes)) if lifetimes else 0.0,
        "Memory Lifetime Std": float(np.std(lifetimes)) if lifetimes else 0.0,
    }
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def scatter_by_strategy(ax, x_values: List[float], y_values: List[float], records: List[Dict]):
    states = [strategy_of(row) for row in records]
    for strategy in STRATEGIES:
        xs = [x for x, state in zip(x_values, states) if state == strategy and not np.isnan(x)]
        ys = [y for y, state in zip(y_values, states) if state == strategy and not np.isnan(y)]
        if xs:
            ax.scatter(xs, ys, label=strategy, color=STRATEGY_COLORS[strategy], s=45, alpha=0.85)
