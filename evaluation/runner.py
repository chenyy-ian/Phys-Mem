import os
import shlex
import subprocess
from typing import Dict, List

from .analysis import ExperimentAnalyzer
from .config import ExperimentConfig, ExperimentMethod, default_experiment_suite
from .logger import EvaluationLogger
from .visualization import EvaluationVisualizer


class BenchmarkRunner:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        if not self.config.methods:
            self.config.methods = default_experiment_suite()
        self.logger = EvaluationLogger(config.output_root)
        self.analyzer = ExperimentAnalyzer()
        self.visualizer = EvaluationVisualizer(config.output_root)

    def build_command(self, method: ExperimentMethod) -> List[str]:
        output_folder = os.path.join(self.config.output_root, "runs", method.name)
        debug_dir = os.path.join(output_folder, "stableworld_debug")
        command = [
            "python",
            self.config.inference_script,
            "--config_path",
            self.config.config_path,
            "--img_path",
            self.config.image_path,
            "--output_folder",
            output_folder,
            "--num_output_frames",
            str(self.config.num_output_frames),
            "--seed",
            str(self.config.seed),
            "--pretrained_model_path",
            self.config.pretrained_model_path,
            "--Threshold",
            str(self.config.threshold),
            "--stableworld_debug_dir",
            debug_dir,
        ]
        if self.config.checkpoint_path:
            command += ["--checkpoint_path", self.config.checkpoint_path]
        if self.config.depth_checkpoint and method.similarity_estimator == "depth":
            command += ["--depth_checkpoint", self.config.depth_checkpoint]
        command += method.command_args()
        return command

    def export_plan(self):
        methods = []
        commands = []
        for method in self.config.methods:
            command = self.build_command(method)
            methods.append({
                "name": method.name,
                "similarity_estimator": method.similarity_estimator,
                "memory_scheduler": method.memory_scheduler,
                "evidence_mode": method.evidence_mode,
                "notes": method.notes,
            })
            commands.append({"method": method.name, "command": " ".join(shlex.quote(x) for x in command)})
        self.logger.write_json("experiment_plan.json", {
            "output_root": self.config.output_root,
            "scenarios": self.config.scenarios,
            "methods": methods,
            "experiment_groups": self.experiment_groups(),
        })
        self.logger.write_csv("commands/run_commands.csv", commands, ["method", "command"])
        return commands

    def run_all(self, dry_run: bool = True):
        commands = self.export_plan()
        if dry_run:
            self.logger.append_event("Dry run completed; commands exported without launching inference.")
            return commands
        for item in commands:
            self.logger.append_event(f"Running {item['method']}")
            method = next(x for x in self.config.methods if x.name == item["method"])
            subprocess.run(self.build_command(method), check=True)
        return commands

    def collect_results(self):
        summaries = []
        for method in self.config.methods:
            debug_dir = os.path.join(self.config.output_root, "runs", method.name, "stableworld_debug")
            summaries.append(self.analyzer.summarize_method(method.name, debug_dir))
        tables = self.analyzer.build_tables(summaries)
        self._write_summaries(summaries, tables)
        self.visualizer.save_all(summaries, tables)
        self._write_report(summaries, tables)
        return summaries

    def _write_summaries(self, summaries: List[Dict], tables: Dict[str, List[Dict]]):
        if summaries:
            self.logger.write_csv("metrics/method_summary.csv", summaries, list(summaries[0].keys()))
        for name, rows in tables.items():
            if rows:
                self.logger.write_csv(f"tables/{name}.csv", rows, list(rows[0].keys()))

    def _write_report(self, summaries: List[Dict], tables: Dict[str, List[Dict]]):
        path = os.path.join(self.config.output_root, "experiment_report.md")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("# Evaluation Report\n\n")
            f.write("## Main Comparison\n\n")
            f.write("Methods: Baseline, ORB, LightGlue, LightGlue+Penalty, Depth, Depth+Action, Fusion, Phys-Mem.\n\n")
            f.write("## Experiment Groups\n\n")
            for name, desc in self.experiment_groups().items():
                f.write(f"- {name}: {desc}\n")
            f.write("\n## Available Tables\n\n")
            for table_name in tables:
                f.write(f"- tables/{table_name}.csv\n")
            f.write("\n## Available Figures\n\n")
            for idx in range(1, 9):
                f.write(f"- figures/Figure{idx}_*.png\n")
            f.write("\n## Notes\n\n")
            f.write("This framework aggregates frozen algorithm outputs and does not modify algorithm implementations.\n")

    @staticmethod
    def experiment_groups() -> Dict[str, str]:
        return {
            "Main Comparison": "Compare Baseline, StableWorld ORB, LightGlue, Depth, Fusion, and Phys-Mem.",
            "Ablation Study": "Remove appearance, semantic, geometry, intent, fusion, or Phys-Mem components.",
            "Sensitivity Analysis": "Sweep thresholds, fusion weights, LightGlue penalty alpha, and depth metrics.",
            "Runtime Analysis": "Compare runtime overhead and matching/depth/action statistics.",
            "Memory Analysis": "Compare replacement rate, Phys-Mem state ratios, and KV update behavior.",
            "Failure Cases": "Catalog repeated texture, fast rotation, occlusion, low texture, depth ambiguity, and intent mismatch.",
        }
