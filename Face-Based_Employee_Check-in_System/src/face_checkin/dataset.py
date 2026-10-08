from pathlib import Path

from .config import IMAGE_EXTENSIONS


def load_dataset(dataset_dir):
    dataset_dir = Path(dataset_dir)

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    records = []

    for image_path in sorted(dataset_dir.iterdir()):
        if not image_path.is_file():
            continue

        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        filename = image_path.stem

        if filename.startswith("Avatar_"):
            label = filename[len("Avatar_"):]
        else:
            label = filename

        records.append({
            "image_path": str(image_path),
            "label": label
        })

    return records