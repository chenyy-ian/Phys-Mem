import matplotlib.pyplot as plt
import numpy as np

from .plot_utils import (
    WINDOW_COLORS,
    compute_memory_lifetimes,
    finalize_axis,
    frame_ids,
    memory_windows,
    save_figure,
)


def _window_matrix(windows):
    max_slots = max((len(window) for window in windows), default=0)
    matrix = np.full((len(windows), max_slots), np.nan, dtype=np.float32)
    for row, window in enumerate(windows):
        for col, value in enumerate(window):
            matrix[row, col] = value
    return matrix


def plot_memory_window_timeline(records, output_dirs):
    frames = frame_ids(records)
    windows = memory_windows(records)
    matrix = _window_matrix(windows).T
    fig, ax = plt.subplots(figsize=(12, 5.2))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Memory Slot")
    ax.set_title("Memory Window Timeline")
    step = max(1, len(frames) // 8)
    ax.set_xticks(range(0, len(frames), step))
    ax.set_xticklabels([frames[i] for i in range(0, len(frames), step)])
    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02, label="Stored Frame ID")
    finalize_axis(ax)
    save_figure(fig, output_dirs, "figure2_memory_window_timeline")


def plot_memory_lifetime(records, output_dirs):
    lifetimes = list(compute_memory_lifetimes(records).values())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].hist(lifetimes, bins=min(20, max(5, len(set(lifetimes)))), color="#4c78a8", edgecolor="white")
    axes[0].set_xlabel("Lifetime")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Memory Lifetime Histogram")
    finalize_axis(axes[0])
    axes[1].boxplot(lifetimes, patch_artist=True, boxprops={"facecolor": "#4c78a8", "alpha": 0.75})
    axes[1].set_ylabel("Lifetime")
    axes[1].set_title("Memory Lifetime Boxplot")
    finalize_axis(axes[1])
    save_figure(fig, output_dirs, "figure7_memory_lifetime")


def plot_window_evolution(records, output_dirs):
    frames = frame_ids(records)
    windows = memory_windows(records)
    changes = []
    labels = []
    for idx in range(1, len(windows)):
        previous = set(windows[idx - 1])
        current = set(windows[idx])
        retained = len(previous & current)
        deleted = len(previous - current)
        inserted = len(current - previous)
        changes.append([retained, deleted, inserted])
        labels.append(frames[idx])
    if not changes:
        changes = [[0, 0, 0]]
        labels = [frames[0] if frames else 0]
    arr = np.array(changes, dtype=np.float32)
    fig, ax = plt.subplots(figsize=(12, 4.8))
    x = np.arange(len(labels))
    bottom = np.zeros(len(labels), dtype=np.float32)
    for col, name in enumerate(["Retained", "Deleted", "Inserted"]):
        ax.bar(x, arr[:, col], bottom=bottom, color=WINDOW_COLORS[name], label=name)
        bottom += arr[:, col]
    step = max(1, len(labels) // 8)
    ax.set_xticks(x[::step])
    ax.set_xticklabels([labels[i] for i in range(0, len(labels), step)])
    ax.set_xlabel("Frame")
    ax.set_ylabel("Window Change Count")
    ax.set_title("Window Evolution")
    ax.legend(frameon=False)
    finalize_axis(ax)
    save_figure(fig, output_dirs, "figure8_window_evolution")
