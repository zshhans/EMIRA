import argparse

from config.emira_config import DATASETS


def get_parser():
    parser = argparse.ArgumentParser(
        description="EMIRA: irregular multivariate time series forecasting"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser(
        "forecasting", help="Train and evaluate the four benchmark datasets"
    )
    train.add_argument(
        "--dataset", nargs="+", choices=("all",) + DATASETS, default=["all"]
    )
    train.add_argument("--seeds", type=int, nargs="+", default=[30, 31, 32])
    evaluate = commands.add_parser(
        "evaluate", help="Evaluate a saved submission checkpoint"
    )
    evaluate.add_argument("--checkpoint", required=True)
    for command in (train, evaluate):
        command.add_argument(
            "--data-root", default="dataset", help="Parent of the four dataset folders"
        )
        command.add_argument("--output-dir", default="logs")
        command.add_argument("--device", default="cuda:0", help="cuda:<index> or cpu")
        command.add_argument("--num-workers", type=int, default=8)
        command.add_argument(
            "--deterministic",
            action=argparse.BooleanOptionalAction,
            default=False if command is train else None,
            help="Use deterministic CUDA backward (evaluation defaults to checkpoint policy)",
        )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    if args.num_workers < 0:
        parser.error("--num-workers must be nonnegative")
    if args.command == "forecasting":
        if "all" in args.dataset and len(args.dataset) > 1:
            parser.error("Use --dataset all by itself")
        from experiments.forecasting import run_forecasting

        run_forecasting(args)
    else:
        from experiments.forecasting import run_evaluation

        run_evaluation(args)


if __name__ == "__main__":
    main()
