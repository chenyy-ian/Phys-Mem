import matplotlib.pyplot as plt
import numpy as np

from .plot_utils import (
    STRATEGIES,
    STRATEGY_COLORS,
    finalize_axis,
    frame_ids,
    normalize,
    numeric_series,
    save_figure,
    strategy_of,
)


def plot_unified_memory_score(records, output_dirs):
    frames = frame_ids(records)
    scores = numeric_series(records, "FusionWeight")
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(frames, scores, color="#333333", marker="o", label="Unified Memory Score")
    for strategy in STRATEGIES:
        xs = [frame for frame, row in zip(frames, records) if strategy_of(row) == strategy]
        ys = [score for score, row in zip(scores, records) if strategy_of(row) == strategy]
        if xs:
            ax.scatter(xs, ys, color=STRATEGY_COLORS[strategy], label=strategy, s=60, zorder=4)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Score")
    ax.set_title("Unified Memory Score")
    ax.legend(ncol=3, frameon=False)
    finalize_axis(ax)
    save_figure(fig, output_dirs, "figure3_unified_memory_score")


def plot_evidence_score_curve(records, output_dirs):
    frames = frame_ids(records)
    series = {
        "Similarity": numeric_series(records, "Similarity"),
        "Semantic": numeric_series(records, "SemanticScore"),
        "Geometry": numeric_series(records, "GeometryScore"),
        "Intent": normalize(numeric_series(records, "ActionMagnitude")),
        "Unified": numeric_series(records, "FusionWeight"),
    }
    colors = {
        "Similarity": "#4c78a8",
        "Semantic": "#f58518",
        "Geometry": "#54a24b",
        "Intent": "#b279a2",
        "Unified": "#333333",
    }
    fig, ax = plt.subplots(figsize=(12, 5.0))
    for name, values in series.items():
        ax.plot(frames, values, marker="o", label=name, color=colors[name])
    ax.set_xlabel("Frame")
    ax.set_ylabel("Evidence Score")
    ax.set_title("Evidence Score Curve")
    ax.legend(ncol=3, frameon=False)
    finalize_axis(ax)
    save_figure(fig, output_dirs, "figure4_evidence_score_curve")


def plot_confidence_curve(records, output_dirs):
    frames = frame_ids(records)
    series = {
        "Confidence": numeric_series(records, "Confidence"),
        "DecisionConfidence": numeric_series(records, "DecisionConfidence"),
        "PolicyConfidence": numeric_series(records, "PolicyConfidence"),
    }
    fig, ax = plt.subplots(figsize=(12, 4.8))
    for name, values in series.items():
        ax.plot(frames, values, marker="o", label=name)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Confidence")
    ax.set_title("Confidence Curve")
    ax.legend(frameon=False)
    finalize_axis(ax)
    save_figure(fig, output_dirs, "figure10_confidence_curve")


def plot_decision_confidence_histogram(records, output_dirs):
    values = [x for x in numeric_series(records, "DecisionConfidence") if not np.isnan(x)]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.hist(values, bins=min(20, max(5, len(set(values)))), color="#4c78a8", edgecolor="white")
    ax.set_xlabel("Decision Confidence")
    ax.set_ylabel("Count")
    ax.set_title("Decision Confidence Histogram")
    finalize_axis(ax)
    save_figure(fig, output_dirs, "figure12_decision_confidence_histogram")


def plot_policy_confidence_distribution(records, output_dirs):
    values_by_strategy = []
    labels = []
    for strategy in STRATEGIES:
        values = [
            value for value, row in zip(numeric_series(records, "PolicyConfidence"), records)
            if strategy_of(row) == strategy and not np.isnan(value)
        ]
        if values:
            values_by_strategy.append(values)
            labels.append(strategy)
    if not values_by_strategy:
        values_by_strategy = [[0.0]]
        labels = ["N/A"]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    parts = ax.violinplot(values_by_strategy, showmeans=True, showextrema=True)
    for body in parts["bodies"]:
        body.set_facecolor("#4c78a8")
        body.set_alpha(0.65)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Strategy")
    ax.set_ylabel("Policy Confidence")
    ax.set_title("Policy Confidence Distribution")
    finalize_axis(ax)
    save_figure(fig, output_dirs, "figure16_policy_confidence_distribution")
