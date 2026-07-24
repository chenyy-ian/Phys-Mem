import os
from typing import Dict, List

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


class EvaluationVisualizer:
    def __init__(self, output_root: str):
        self.output_root = output_root
        self.figure_dir = os.path.join(output_root, "figures")
        os.makedirs(self.figure_dir, exist_ok=True)

    def save_all(self, summaries: List[Dict], tables: Dict[str, List[Dict]]):
        self._framework_figure("Figure1_Framework.png")
        self._bar_figure("Figure2_Memory_Timeline.png", summaries, "memory_replacement_rate", "Memory Replacement")
        self._bar_figure("Figure3_Similarity.png", summaries, "average_similarity", "Similarity")
        self._bar_figure("Figure4_Depth.png", summaries, "average_fusion_score", "Geometry/Fusion Score")
        self._bar_figure("Figure5_Action.png", summaries, "average_rotation", "Action Rotation")
        self._bar_figure("Figure6_Fusion.png", summaries, "average_fusion_score", "Fusion Score")
        self._bar_figure("Figure7_Ablation.png", summaries, "average_runtime_ms", "Runtime")
        self._failure_case_figure("Figure8_Failure_Cases.png")
        self._write_pdf_placeholder()

    def _framework_figure(self, name: str):
        path = os.path.join(self.figure_dir, name)
        if cv2 is None:
            self._write_placeholder_png(path, "Figure1 Framework")
            return
        canvas = np.zeros((640, 1200, 3), dtype=np.uint8) + 255
        nodes = [
            ("Appearance", 60, 150),
            ("Semantic", 60, 260),
            ("Geometry", 60, 370),
            ("Intent", 60, 480),
            ("Fusion", 430, 315),
            ("Phys-Mem Scheduler", 720, 315),
            ("Memory State", 980, 315),
        ]
        for label, x, y in nodes:
            cv2.rectangle(canvas, (x, y - 35), (x + 170, y + 35), (245, 248, 252), -1)
            cv2.rectangle(canvas, (x, y - 35), (x + 170, y + 35), (80, 95, 120), 2)
            cv2.putText(canvas, label, (x + 12, y + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (35, 35, 35), 1)
        for _, x, y in nodes[:4]:
            cv2.arrowedLine(canvas, (x + 170, y), (430, 315), (60, 60, 60), 2, tipLength=0.04)
        cv2.arrowedLine(canvas, (600, 315), (720, 315), (60, 60, 60), 2, tipLength=0.04)
        cv2.arrowedLine(canvas, (890, 315), (980, 315), (60, 60, 60), 2, tipLength=0.04)
        cv2.imwrite(path, canvas)

    def _bar_figure(self, name: str, summaries: List[Dict], key: str, title: str):
        path = os.path.join(self.figure_dir, name)
        if cv2 is None:
            self._write_placeholder_png(path, title)
            return
        canvas = np.zeros((650, 1400, 3), dtype=np.uint8) + 255
        cv2.putText(canvas, title, (35, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (35, 35, 35), 2)
        if not summaries:
            cv2.putText(canvas, "No completed experiment logs found.", (35, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 80), 2)
            cv2.imwrite(path, canvas)
            return
        values = [float(x.get(key, 0.0)) for x in summaries]
        max_value = max(max(values), 1.0)
        x = 60
        baseline = 540
        bar_w = max(55, min(120, 900 // max(len(summaries), 1)))
        for row, value in zip(summaries, values):
            h = int((value / max_value) * 390)
            cv2.rectangle(canvas, (x, baseline - h), (x + bar_w, baseline), (70, 130, 210), -1)
            cv2.putText(canvas, f"{value:.3f}", (x, baseline - h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1)
            cv2.putText(canvas, row["method"][:16], (x - 5, baseline + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (40, 40, 40), 1)
            x += bar_w + 34
        cv2.imwrite(path, canvas)

    def _failure_case_figure(self, name: str):
        path = os.path.join(self.figure_dir, name)
        if cv2 is None:
            self._write_placeholder_png(path, "Figure8 Failure Cases")
            return
        canvas = np.zeros((640, 1200, 3), dtype=np.uint8) + 255
        cases = [
            "Repeated Texture",
            "Fast Rotation",
            "Occlusion",
            "Low Texture",
            "Depth Ambiguity",
            "Intent Mismatch",
        ]
        x0, y0 = 60, 110
        for idx, case in enumerate(cases):
            x = x0 + (idx % 3) * 370
            y = y0 + (idx // 3) * 230
            cv2.rectangle(canvas, (x, y), (x + 300, y + 150), (248, 248, 248), -1)
            cv2.rectangle(canvas, (x, y), (x + 300, y + 150), (95, 95, 95), 2)
            cv2.putText(canvas, case, (x + 18, y + 82), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 40, 40), 2)
        cv2.imwrite(path, canvas)

    def _write_pdf_placeholder(self):
        path = os.path.join(self.figure_dir, "figures_manifest.pdf")
        with open(path, "wb") as f:
            f.write(b"%PDF-1.4\n% Evaluation figure manifest placeholder\n")
            f.write(b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n")
            f.write(b"2 0 obj << /Type /Pages /Count 0 >> endobj\n")
            f.write(b"trailer << /Root 1 0 R >>\n%%EOF\n")

    @staticmethod
    def _write_placeholder_png(path: str, title: str):
        del title
        png_1x1_white = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
            b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
            b"\x00\x00\x0cIDAT\x08\xd7c\xf8\xff\xff?\x00\x05\xfe"
            b"\x02\xfeA\xe2-\x9b\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        with open(path, "wb") as f:
            f.write(png_1x1_white)
