import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def run_command(command, cwd=None):
    proc = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "DOVER official evaluation failed.\n"
            f"Command: {' '.join(command)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}\n"
        )
    return proc.stdout


def main():
    parser = argparse.ArgumentParser(description="Run official DOVER video quality assessment.")
    parser.add_argument("--video_dir", required=True, help="Folder containing generated videos")
    parser.add_argument("--output", default="evaluation/results/dover.json")
    parser.add_argument("--dover_repo", default=None, help="Path to cloned official DOVER repo")
    parser.add_argument("--dover_command", nargs=argparse.REMAINDER, help="Explicit official DOVER command. Use {video_dir} and {output} placeholders.")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    if args.dover_command:
        command = [part.format(video_dir=args.video_dir, output=args.output) for part in args.dover_command]
        cwd = args.dover_repo
    else:
        if not args.dover_repo:
            status = {
                "status": "failed",
                "error": "DOVER repo path or --dover_command is required.",
                "official_repo": "https://github.com/VQAssessment/DOVER",
                "expected_metrics": ["Technical Quality", "Aesthetic Quality", "Overall Quality"],
                "note": "This wrapper calls the official DOVER implementation and does not reimplement the model.",
            }
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(status, f, indent=2)
            print(json.dumps(status, indent=2))
            return
        command = [sys.executable, "evaluate.py", "--video_dir", args.video_dir, "--output", args.output]
        cwd = args.dover_repo

    try:
        stdout = run_command(command, cwd=cwd)
        status = {"status": "ok", "stdout": stdout, "command": command, "cwd": cwd}
    except Exception as exc:
        status = {
            "status": "failed",
            "error": str(exc),
            "command": command,
            "cwd": cwd,
            "official_repo": "https://github.com/VQAssessment/DOVER",
        }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
