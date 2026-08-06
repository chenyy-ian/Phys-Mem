import argparse
import glob
import json
import os
import subprocess
import time

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


def rename_latest_eval(output_dir, dim):
    """Rename the just-produced VBench eval_results.json to a clean name."""
    candidates = sorted(
        glob.glob(os.path.join(output_dir, "*_eval_results.json")),
        key=os.path.getmtime,
    )
    if not candidates:
        print(f"[warn] no eval_results for {dim}")
        return
    latest = candidates[-1]
    target = os.path.join(output_dir, f"results_{dim}_eval_results.json")
    if os.path.abspath(latest) != os.path.abspath(target):
        if os.path.exists(target):
            os.remove(target)
        os.rename(latest, target)
        print(f"[renamed] {os.path.basename(latest)} -> {os.path.basename(target)}")


def main():
    parser = argparse.ArgumentParser(description="Run official VBench metrics for generated videos.")
    parser.add_argument("--video_dir", required=True, help="Folder containing generated videos")
    parser.add_argument("--output_dir", default="evaluation/results/vbench")
    parser.add_argument("--vbench_config", default=None, help="Optional VBench full_info JSON/config path")
    parser.add_argument("--prompt_file", default=None, help="Optional VBench prompt file if required")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    all_stdouts = []
    commands_run = []

    try:
        for dim in Vbench_DIMENSIONS:
            print(f"正在评估维度: {dim} ...")
            command = [
                "vbench", "evaluate",
                "--videos_path", args.video_dir,
                "--output_path", args.output_dir,
                "--dimension", dim,
                "--mode", "custom_input",
            ]
            if args.vbench_config:
                command += ["--full_info_path", args.vbench_config]
            if args.prompt_file:
                command += ["--prompt_file", args.prompt_file]

            stdout = run_command(command)
            all_stdouts.append(f"--- Output for {dim} ---\n{stdout}")
            commands_run.append(" ".join(command))
            time.sleep(1)  # 确保 mtime 排序稳定
            rename_latest_eval(args.output_dir, dim)

        status = {
            "status": "ok",
            "stdout": "\n\n".join(all_stdouts),
            "commands": commands_run,
        }
        print("✅ 所有维度评估完成！")

    except Exception as exc:
        status = {
            "status": "failed",
            "error": str(exc),
            "commands": commands_run,
            "official_repo": "https://github.com/Vchitect/VBench",
            "note": "Install and configure official VBench; this wrapper does not reimplement VBench metrics.",
        }

    output = os.path.join(args.output_dir, "vbench_metrics.json")
    with open(output, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()