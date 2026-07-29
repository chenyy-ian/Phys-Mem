import matplotlib.patches as patches
import matplotlib.pyplot as plt

from .plot_utils import WINDOW_COLORS, finalize_axis, frame_ids, memory_windows, save_figure, strategy_of


def _select_cases(records, target_states=("KEEP", "INSERT", "REPLACE")):
    cases = []
    frames = frame_ids(records)
    windows = memory_windows(records)
    for target in target_states:
        for idx in range(1, len(records)):
            if strategy_of(records[idx]) == target and windows[idx - 1] and windows[idx]:
                cases.append((target, frames[idx - 1], windows[idx - 1], frames[idx], windows[idx]))
                break
    return cases


def _draw_window(ax, y, frame, window, previous=None):
    previous_set = set(previous or window)
    current_set = set(window)
    for slot, value in enumerate(window):
        if previous is None:
            label = "Retained"
        elif value in previous_set:
            label = "Retained"
        else:
            label = "Inserted"
        rect = patches.Rectangle((slot, y), 0.9, 0.7, facecolor=WINDOW_COLORS[label], alpha=0.8, edgecolor="white")
        ax.add_patch(rect)
        ax.text(slot + 0.45, y + 0.35, str(value), ha="center", va="center", fontsize=11, color="white")
    if previous is not None:
        deleted = [value for value in previous if value not in current_set]
        for offset, value in enumerate(deleted):
            x = len(window) + 0.5 + offset
            rect = patches.Rectangle((x, y), 0.9, 0.7, facecolor=WINDOW_COLORS["Deleted"], alpha=0.8, edgecolor="white")
            ax.add_patch(rect)
            ax.text(x + 0.45, y + 0.35, str(value), ha="center", va="center", fontsize=11, color="white")
    ax.text(-0.4, y + 0.35, f"F{frame}", ha="right", va="center", fontsize=13)


def plot_case_study(records, output_dirs):
    cases = _select_cases(records)
    if not cases:
        return
    fig_height = max(4.0, 2.1 * len(cases))
    fig, ax = plt.subplots(figsize=(12, fig_height))
    y = 0
    max_width = 0
    for state, prev_frame, prev_window, cur_frame, cur_window in cases:
        ax.text(-0.4, y + 1.25, state, ha="right", va="center", fontsize=15, weight="bold")
        _draw_window(ax, y + 0.75, prev_frame, prev_window)
        _draw_window(ax, y, cur_frame, cur_window, previous=prev_window)
        max_width = max(max_width, len(cur_window) + len([x for x in prev_window if x not in set(cur_window)]) + 2)
        y += 2.0
    ax.set_xlim(-1.8, max_width)
    ax.set_ylim(-0.2, y + 0.7)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title("Case Study: Window Update")
    legend_handles = [
        patches.Patch(color=WINDOW_COLORS["Retained"], label="Retained"),
        patches.Patch(color=WINDOW_COLORS["Deleted"], label="Deleted"),
        patches.Patch(color=WINDOW_COLORS["Inserted"], label="Inserted"),
    ]
    ax.legend(handles=legend_handles, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.08), frameon=False)
    finalize_axis(ax)
    ax.grid(False)
    save_figure(fig, output_dirs, "case_study_window_update")
