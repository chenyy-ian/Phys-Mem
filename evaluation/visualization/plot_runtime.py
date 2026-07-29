import matplotlib.pyplot as plt

from .plot_utils import (
    STRATEGIES,
    STRATEGY_COLORS,
    finalize_axis,
    frame_ids,
    numeric_series,
    save_figure,
    strategy_of,
)


def plot_runtime_curve(records, output_dirs):
    frames = frame_ids(records)
    runtime = numeric_series(records, "Runtime")
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.plot(frames, runtime, color="#333333", marker="o", label="Runtime")
    for strategy in STRATEGIES:
        xs = [frame for frame, row in zip(frames, records) if strategy_of(row) == strategy]
        ys = [value for value, row in zip(runtime, records) if strategy_of(row) == strategy]
        if xs:
            ax.scatter(xs, ys, color=STRATEGY_COLORS[strategy], label=strategy, s=60, zorder=4)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Inference Time")
    ax.set_title("Runtime Curve")
    ax.legend(ncol=3, frameon=False)
    finalize_axis(ax)
    save_figure(fig, output_dirs, "figure9_runtime_curve")


def plot_memory_size_curve(records, output_dirs):
    frames = frame_ids(records)
    memory_size = numeric_series(records, "MemorySize")
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(frames, memory_size, color="#4c78a8", marker="o", label="Memory Size")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Memory Size")
    ax.set_title("Memory Size Curve")
    ax.legend(frameon=False)
    finalize_axis(ax)
    save_figure(fig, output_dirs, "figure11_memory_size_curve")
