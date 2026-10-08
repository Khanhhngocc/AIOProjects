from vic_ltsf_forecasting.pipeline import run_pipeline

def main():
    """Train all VIC forecasting experiments."""
    run_pipeline(
        show_plots=False,
    )

if __name__ == "__main__":
    main()