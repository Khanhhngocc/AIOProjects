import json
import random
import numpy as np
import pandas as pd
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_run_dir(base_dir):
    base_dir.mkdir(parents=True, exist_ok=True)

    run_numbers = []

    for path in base_dir.iterdir():
        if path.is_dir() and path.name.startswith("run_"):
            try:
                run_numbers.append(int(path.name.split("_")[1]))
            except ValueError:
                pass

    next_run = max(run_numbers, default=0) + 1
    run_dir = base_dir / f"run_{next_run:02d}"
    run_dir.mkdir()

    return run_dir


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def save_dataframe(dataframe, path):
    dataframe.to_csv(path, index=False)