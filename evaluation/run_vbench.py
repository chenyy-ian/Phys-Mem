import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


Vbench_DIMENSIONS = [
    "background_consistency",
    "subject_consistency",
    "motion_smoothness",
    "temporal_flickering",
    "imaging_quality",
]


def run_command(command, cwd=None):
    proc = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "VBench official evaluation failed.\n"
            f"Command: {' '.join(command)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}\n"
        )
    return proc.stdout


def main():
    parser = argparse.ArgumentParser(description="Run official VBench metrics for generated videos.")
    parser.add_argument("--video_dir", required=True, help="Folder containing generated videos")
    parser.add_argument("--output_dir", default="evaluation/results/vbench")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--vbench_config", default=None, help="Optional VBench full_info JSON/config path")
    parser.add_argument("--prompt_file", default=None, help="Optional VBench prompt file if required by the installed version")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "vbench",
        "evaluate",
        "--videos_path",
        args.video_dir,
        "--output_path",
        args.output_dir,
        "--dimension",
        *Vbench_DIMENSIONS,
        "--device",
        args.device,
    ]
    if args.vbench_config:
        command += ["--full_info_path", args.vbench_config]
    if args.prompt_file:
        command += ["--prompt_file", args.prompt_file]

    try:
        stdout = run_command(command)
        status = {"status": "ok", "stdout": stdout, "command": command}
    except Exception as exc:
        status = {
            "status": "failed",
            "error": str(exc),
            "command": command,
            "official_repo": "https://github.com/Vchitect/VBench",
            "note": "Install and configure official VBench; this wrapper does not reimplement VBench metrics.",
        }

    output = Path(args.output_dir) / "vbench_metrics.json"
    with open(output, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
