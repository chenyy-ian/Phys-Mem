import argparse
import json
import os
import subprocess
import sys


def run_command(command, cwd=None):
    proc = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "STREAM official evaluation failed.\n"
            f"Command: {' '.join(command)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}\n"
        )
    return proc.stdout


def main():
    parser = argparse.ArgumentParser(description="Run STREAM temporal/spatial consistency evaluation.")
    parser.add_argument("--video_dir", required=True, help="Folder containing generated videos")
    parser.add_argument("--output", default="evaluation/results/stream.json")
    parser.add_argument("--stream_repo", default=None, help="Path to cloned official STREAM repo")
    parser.add_argument("--stream_command", nargs=argparse.REMAINDER, help="Explicit official STREAM command. Use {video_dir} and {output} placeholders.")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    if not args.stream_command:
        status = {
            "status": "failed",
            "error": "STREAM official command is not configured. Pass --stream_repo and --stream_command.",
            "official_paper": "https://openreview.net/forum?id=fZwY0JQZes",
            "official_code": "https://github.com/pro2nit/STREAM",
            "expected_metrics": ["Temporal Score", "Spatial Score"],
            "required_inputs": ["generated video files", "official STREAM model/config/checkpoints as required by the repo"],
            "note": "This wrapper intentionally does not reimplement STREAM.",
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(status, f, indent=2)
        print(json.dumps(status, indent=2))
        return

    command = [part.format(video_dir=args.video_dir, output=args.output) for part in args.stream_command]
    try:
        stdout = run_command(command, cwd=args.stream_repo)
        status = {"status": "ok", "stdout": stdout, "command": command, "cwd": args.stream_repo}
    except Exception as exc:
        status = {
            "status": "failed",
            "error": str(exc),
            "command": command,
            "cwd": args.stream_repo,
            "official_paper": "https://openreview.net/forum?id=fZwY0JQZes",
            "official_code": "https://github.com/pro2nit/STREAM",
        }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
