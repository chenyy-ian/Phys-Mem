"""Paper-grade visualizations from ExperimentRecord.csv only.

This package keeps a minimal EvaluationVisualizer shim so the existing Stage10
runner import remains compatible after introducing the visualization package.
"""

import os


class EvaluationVisualizer:
    def __init__(self, output_root: str):
        self.output_root = output_root
        self.figure_dir = os.path.join(output_root, "figures")
        os.makedirs(self.figure_dir, exist_ok=True)

    def save_all(self, summaries, tables):
        manifest = os.path.join(self.figure_dir, "stage10_visualization_manifest.txt")
        with open(manifest, "w", encoding="utf-8") as f:
            f.write("Stage10 compatibility visualizer.\n")
            f.write(f"summaries={len(summaries)}\n")
            f.write(f"tables={','.join(sorted(tables.keys())) if isinstance(tables, dict) else ''}\n")
