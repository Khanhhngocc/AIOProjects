import argparse

from vic_ltsf_forecasting.pipeline import run_pipeline


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="VIC LTSF-Linear Forecasting"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )

    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable visualization",
    )

    parser.add_argument(
        "--force-train",
        action="store_true",
        help="Train models again even when checkpoints already exist",
    )

    return parser.parse_args()


def main():
    """Run the VIC forecasting application."""
    args = parse_args()

    run_pipeline(
        num_epochs=args.epochs,
        show_plots=not args.no_plots,
        force_train=args.force_train,
    )


if __name__ == "__main__":
    main()