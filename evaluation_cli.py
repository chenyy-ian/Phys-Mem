import argparse

from evaluation import BenchmarkRunner, ExperimentConfig, default_experiment_suite


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", type=str, default="outputs/evaluation")
    parser.add_argument("--config_path", type=str, default="configs/inference_yaml/inference_universal.yaml")
    parser.add_argument("--img_path", type=str, default="demo_images/universal/0011.png")
    parser.add_argument("--pretrained_model_path", type=str, default="Matrix-Game-2.0")
    parser.add_argument("--checkpoint_path", type=str, default="")
    parser.add_argument("--depth_checkpoint", type=str, default="")
    parser.add_argument("--num_output_frames", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.78)
    parser.add_argument("--run", action="store_true", help="Launch inference jobs. Default only exports commands and aggregates existing logs.")
    parser.add_argument("--collect_only", action="store_true", help="Only collect existing logs and regenerate tables/figures.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = ExperimentConfig(
        output_root=args.output_root,
        config_path=args.config_path,
        image_path=args.img_path,
        pretrained_model_path=args.pretrained_model_path,
        checkpoint_path=args.checkpoint_path,
        depth_checkpoint=args.depth_checkpoint,
        num_output_frames=args.num_output_frames,
        seed=args.seed,
        threshold=args.threshold,
        methods=default_experiment_suite(),
    )
    runner = BenchmarkRunner(config)
    if not args.collect_only:
        runner.run_all(dry_run=not args.run)
    runner.collect_results()


if __name__ == "__main__":
    main()
