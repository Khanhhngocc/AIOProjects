from pathlib import Path

import pandas as pd

from cocktail_suggestions.database import DatabaseSetup
from cocktail_suggestions.data_processor import CocktailDataProcessor


def main():
    print("Starting database setup...")

    db_setup = DatabaseSetup()
    db_setup.create_database()
    db_setup.setup_pgvector()

    project_root = Path(__file__).resolve().parent.parent
    csv_path = project_root / "data" / "raw" / "cocktails.csv"

    print(f"Loading dataset: {csv_path}")

    df = pd.read_csv(csv_path)

    processor = CocktailDataProcessor()
    df = processor.clean_data(df)
    processor.store_cocktails(df)

    print("Database setup completed.")


if __name__ == "__main__":
    main()