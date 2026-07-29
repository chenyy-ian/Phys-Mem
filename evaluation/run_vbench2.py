import argparse
import json
import os
import subprocess
import sys


VBENCH2_DIMENSIONS = [
    "human_fidelity",
    "controllability",
    "creativity",
    "physics",
    "commonsense",
]


def run_command(command, cwd=None):
    proc = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "VBench-2.0 official evaluation failed.\n"
            f"Command: {' '.join(command)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}\n"
        )
    return proc.stdout


def main():
    parser = argparse.ArgumentParser(description="Run official VBench-2.0 metrics for generated videos.")
    parser.add_argument("--video_dir", required=True, help="Folder containing generated videos")
    parser.add_argument("--output", default="evaluation/results/vbench2.json")
    parser.add_argument("--vbench2_repo", default=None, help="Path to VBench/VBench-2.0 official repo folder")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dimensions", nargs="*", default=VBENCH2_DIMENSIONS)
    parser.add_argument("--vbench2_command", nargs=argparse.REMAINDER, help="Explicit official VBench-2.0 command. Use {video_dir}, {output}, {device}, {dimensions} placeholders.")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    dimensions = ",".join(args.dimensions)
    if args.vbench2_command:
        command = [
            part.format(
                video_dir=args.video_dir,
                output=args.output,
                device=args.device,
                dimensions=dimensions,
            )
            for part in args.vbench2_command
        ]
        cwd = args.vbench2_repo
    elif args.vbench2_repo:
        command = [
            sys.executable,
            "evaluate.py",
            "--videos_path",
            args.video_dir,
            "--output_path",
            args.output,
            "--device",
            args.device,
            "--dimension",
            *args.dimensions,
        ]
        cwd = args.vbench2_repo
    else:
        command = [
            sys.executable,
            "-m",
            "vbench2",
            "evaluate",
            "--videos_path",
            args.video_dir,
            "--output_path",
            args.output,
            "--device",
            args.device,
            "--dimension",
            *args.dimensions,
        ]
        cwd = None

    try:
        stdout = run_command(command, cwd=cwd)
        status = {"status": "ok", "stdout": stdout, "command": command, "cwd": cwd}
    except Exception as exc:
        status = {
            "status": "failed",
            "error": str(exc),
            "command": command,
            "cwd": cwd,
            "official_repo": "https://github.com/Vchitect/VBench/tree/master/VBench-2.0",
            "official_project": "https://vchitect.github.io/VBench-2.0-project/",
            "expected_metrics": args.dimensions,
            "note": "This wrapper calls the official VBench-2.0 implementation and does not reimplement any metric.",
        }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
