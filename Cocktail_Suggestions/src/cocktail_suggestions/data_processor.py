from pgvector import Vector
from sentence_transformers import SentenceTransformer

from cocktail_suggestions.config import MODEL_NAME
from cocktail_suggestions.database import DatabaseSetup


class CocktailDataProcessor:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)
        self.db_setup = DatabaseSetup()

    @staticmethod
    def _find_column(df, candidates):
        """Return the first matching column."""
        for column in candidates:
            if column in df.columns:
                return column
        return None

    @staticmethod
    def _join_numbered_fields(row, prefix):
        """Combine strIngredient1...15 or strMeasure1...15."""
        values = []

        for i in range(1, 16):
            value = str(row.get(f"{prefix}{i}", "")).strip()

            if value and value.lower() not in {"none", "nan"}:
                values.append(value)

        return ", ".join(values)

    def clean_data(self, df):
        """Clean and normalize cocktail dataset."""
        df = df.copy()
        df.columns = df.columns.str.strip()
        df = df.fillna("")

        name_col = self._find_column(df, ["name", "Cocktail", "strDrink"])
        category_col = self._find_column(df, ["category", "Category", "strCategory"])
        alcoholic_col = self._find_column(df, ["alcoholic", "Alcoholic", "strAlcoholic"])
        glass_col = self._find_column(df, ["glassType", "glass", "Glass", "strGlass"])
        instructions_col = self._find_column(df, ["instructions", "Instructions", "strInstructions"])
        description_col = self._find_column(df, ["description", "Description"])
        ingredients_col = self._find_column(df, ["ingredients", "Ingredients"])
        measure_col = self._find_column(df, ["ingredientMeasures", "measure", "Measure"])
        iba_col = self._find_column(df, ["iba", "strIBA"])

        if name_col is None:
            raise ValueError("Cannot find cocktail name column.")

        df = df.drop_duplicates(subset=[name_col]).reset_index(drop=True)

        df["name"] = df[name_col].astype(str).str.strip()
        df["category"] = df[category_col].astype(str).str.strip() if category_col else ""
        df["alcoholic"] = df[alcoholic_col].astype(str).str.strip() if alcoholic_col else ""
        df["glass"] = df[glass_col].astype(str).str.strip() if glass_col else ""
        df["instructions"] = df[instructions_col].astype(str).str.strip() if instructions_col else ""
        df["description"] = df[description_col].astype(str).str.strip() if description_col else ""
        df["iba"] = df[iba_col].astype(str).str.strip() if iba_col else ""

        if ingredients_col:
            df["ingredients"] = df[ingredients_col].astype(str).str.strip()
            df["measure"] = df[measure_col].astype(str).str.strip() if measure_col else ""
        else:
            df["ingredients"] = df.apply(
                lambda row: self._join_numbered_fields(row, "strIngredient"),
                axis=1
            )
            df["measure"] = df.apply(
                lambda row: self._join_numbered_fields(row, "strMeasure"),
                axis=1
            )

        df["combined_text"] = (
            "Name: " + df["name"]
            + ". Category: " + df["category"]
            + ". Alcohol: " + df["alcoholic"]
            + ". Glass: " + df["glass"]
            + ". Ingredients: " + df["ingredients"]
            + ". Description: " + df["description"]
            + ". Instructions: " + df["instructions"]
        )

        df["combined_text"] = df["combined_text"].str.replace(r"\s+", " ", regex=True).str.strip()

        print(f"After cleaning: {len(df)} cocktails")
        return df

    def generate_embeddings(self, texts):
        """Convert text into embedding vectors."""
        return self.model.encode(texts)

    def create_recipe_text(self, row):
        """Create readable cocktail recipe."""
        ingredients = [
            item.strip()
            for item in str(row.get("ingredients", "")).split(",")
            if item.strip()
        ]

        measures = [
            item.strip()
            for item in str(row.get("measure", "")).split(",")
            if item.strip()
        ]

        recipe_lines = [
            f"Drink: {row.get('name', '')}",
            f"Category: {row.get('category', '')}",
            f"Type: {row.get('alcoholic', '')}",
            f"Glass: {row.get('glass', '')}",
            "Ingredients:"
        ]

        for i, ingredient in enumerate(ingredients):
            measure = measures[i] if i < len(measures) else ""
            recipe_lines.append(f"- {measure} {ingredient}" if measure else f"- {ingredient}")

        if row.get("instructions"):
            recipe_lines.append(f"Instructions: {row['instructions']}")

        return "\n".join(recipe_lines)

    def store_cocktails(self, df):
        """Generate embeddings and store cocktails in PostgreSQL."""
        conn = None
        cursor = None

        try:
            conn = self.db_setup.get_connection()
            cursor = conn.cursor()

            cursor.execute("DELETE FROM cocktails")

            print(f"Generating embeddings for {len(df)} cocktails...")
            embeddings = self.generate_embeddings(df["combined_text"].tolist())

            print("Storing cocktails in database...")

            for index, (_, row) in enumerate(df.iterrows()):
                recipe = self.create_recipe_text(row)

                cursor.execute("""
                    INSERT INTO cocktails (
                        name, ingredients, recipe, glass,
                        category, iba, alcoholic, embedding
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    row.get("name", ""),
                    row.get("ingredients", ""),
                    recipe,
                    row.get("glass", ""),
                    row.get("category", ""),
                    row.get("iba", ""),
                    row.get("alcoholic", ""),
                    Vector(embeddings[index])
                ))

            conn.commit()
            print(f"Successfully stored {len(df)} cocktails.")

        except Exception as e:
            if conn:
                conn.rollback()
            raise

        finally:
            if cursor:
                cursor.close()

            if conn:
                conn.close()