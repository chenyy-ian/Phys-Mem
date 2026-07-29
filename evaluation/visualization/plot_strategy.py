import matplotlib.pyplot as plt
import numpy as np

from .plot_utils import (
    STRATEGIES,
    STRATEGY_COLORS,
    finalize_axis,
    frame_ids,
    save_figure,
    scatter_by_strategy,
    strategy_distribution,
    strategy_indices,
    strategy_of,
    transition_matrix,
    numeric_series,
)


def plot_strategy_timeline(records, output_dirs):
    frames = frame_ids(records)
    indices = strategy_indices(records)
    fig, ax = plt.subplots(figsize=(12, 3.8))
    for strategy in STRATEGIES:
        xs = [frame for frame, row in zip(frames, records) if strategy_of(row) == strategy]
        ys = [STRATEGIES.index(strategy) for row in records if strategy_of(row) == strategy]
        ax.scatter(xs, ys, color=STRATEGY_COLORS[strategy], label=strategy, s=55, marker="o")
    ax.plot(frames, indices, color="#444444", linewidth=1.2, alpha=0.35)
    ax.set_yticks(range(len(STRATEGIES)))
    ax.set_yticklabels(STRATEGIES)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Strategy")
    ax.set_title("Strategy Timeline")
    ax.legend(ncol=len(STRATEGIES), loc="upper center", bbox_to_anchor=(0.5, 1.24), frameon=False)
    finalize_axis(ax)
    save_figure(fig, output_dirs, "figure1_strategy_timeline")


def plot_strategy_distribution(records, output_dirs):
    dist = strategy_distribution(records)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    values = [dist[strategy] * 100.0 for strategy in STRATEGIES]
    ax.bar(STRATEGIES, values, color=[STRATEGY_COLORS[s] for s in STRATEGIES])
    ax.set_xlabel("Strategy")
    ax.set_ylabel("Ratio (%)")
    ax.set_title("Strategy Distribution")
    finalize_axis(ax)
    save_figure(fig, output_dirs, "figure5_strategy_distribution")


def plot_transition_matrix(records, output_dirs):
    matrix = transition_matrix(records)
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    image = ax.imshow(matrix, cmap="Blues", vmin=0.0, vmax=max(1.0, float(np.max(matrix))))
    ax.set_xticks(range(len(STRATEGIES)))
    ax.set_yticks(range(len(STRATEGIES)))
    ax.set_xticklabels(STRATEGIES, rotation=35, ha="right")
    ax.set_yticklabels(STRATEGIES)
    ax.set_xlabel("Next Strategy")
    ax.set_ylabel("Previous Strategy")
    ax.set_title("Strategy Transition Matrix")
    for row in range(len(STRATEGIES)):
        for col in range(len(STRATEGIES)):
            ax.text(col, row, f"{matrix[row, col]:.2f}", ha="center", va="center", fontsize=12)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Transition Probability")
    finalize_axis(ax)
    save_figure(fig, output_dirs, "figure6_transition_matrix")


def plot_similarity_vs_strategy(records, output_dirs):
    frames = [strategy_of(row) for row in records]
    x_values = numeric_series(records, "Similarity")
    y_values = [STRATEGIES.index(state) if state in STRATEGIES else np.nan for state in frames]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    scatter_by_strategy(ax, x_values, y_values, records)
    ax.set_yticks(range(len(STRATEGIES)))
    ax.set_yticklabels(STRATEGIES)
    ax.set_xlabel("Similarity")
    ax.set_ylabel("Strategy")
    ax.set_title("Similarity vs Strategy")
    ax.legend(frameon=False)
    finalize_axis(ax)
    save_figure(fig, output_dirs, "figure13_similarity_vs_strategy")


def plot_geometry_vs_semantic(records, output_dirs):
    x_values = numeric_series(records, "GeometryScore")
    y_values = numeric_series(records, "SemanticScore")
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    scatter_by_strategy(ax, x_values, y_values, records)
    ax.set_xlabel("Geometry Score")
    ax.set_ylabel("Semantic Score")
    ax.set_title("Geometry vs Semantic")
    ax.legend(frameon=False)
    finalize_axis(ax)
    save_figure(fig, output_dirs, "figure14_geometry_vs_semantic")


def plot_action_magnitude_vs_strategy(records, output_dirs):
    states = [strategy_of(row) for row in records]
    x_values = numeric_series(records, "ActionMagnitude")
    y_values = [STRATEGIES.index(state) if state in STRATEGIES else np.nan for state in states]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    scatter_by_strategy(ax, x_values, y_values, records)
    ax.set_yticks(range(len(STRATEGIES)))
    ax.set_yticklabels(STRATEGIES)
    ax.set_xlabel("Action Magnitude")
    ax.set_ylabel("Strategy")
    ax.set_title("Action Magnitude vs Strategy")
    ax.legend(frameon=False)
    finalize_axis(ax)
    save_figure(fig, output_dirs, "figure15_action_magnitude_vs_strategy")
