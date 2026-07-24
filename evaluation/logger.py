import csv
import json
import os
import time
from typing import Dict, Iterable, List


class EvaluationLogger:
    def __init__(self, output_root: str):
        self.output_root = output_root
        os.makedirs(output_root, exist_ok=True)

    def write_json(self, relative_path: str, payload: Dict):
        path = os.path.join(self.output_root, relative_path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def write_csv(self, relative_path: str, rows: Iterable[Dict], fieldnames: List[str]):
        path = os.path.join(self.output_root, relative_path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def append_event(self, message: str):
        path = os.path.join(self.output_root, "evaluation_events.log")
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n")
