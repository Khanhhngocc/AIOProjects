# ============================================================
# COCKTAIL SUGGESTION SYSTEM
# Chapter 9 - Project 2
# Vector Database with PostgreSQL + pgvector
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import os
import ast
import traceback

import pandas as pd
import psycopg2

from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from sentence_transformers import SentenceTransformer


# ============================================================
# 2. DATABASE SETUP
# ============================================================

class DatabaseSetup:

    # ========================================================
    # 3. DATABASE CONFIGURATION
    # ========================================================

    def __init__(self):
        self.host = os.getenv("DB_HOST", "localhost")
        self.port = os.getenv("DB_PORT", "5432")
        self.user = os.getenv("DB_USER", "postgres")
        self.password = os.getenv("DB_PASSWORD", "postgres")
        self.db_name = os.getenv("DB_NAME", "cocktail_db")


    # ========================================================
    # 4. CREATE POSTGRESQL DATABASE
    # ========================================================

    def create_database(self):
        """Create the database if it does not exist."""

        try:
            # Connect to default PostgreSQL database
            conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database="postgres"
            )

            # CREATE DATABASE cannot run inside a normal transaction
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cursor = conn.cursor()

            # Check whether database already exists
            cursor.execute(
                """
                SELECT 1
                FROM pg_catalog.pg_database
                WHERE datname = %s
                """,
                (self.db_name,)
            )
            exists = cursor.fetchone()

            # Create database if necessary
            if not exists:
                cursor.execute(f'CREATE DATABASE "{self.db_name}"')
                print(f"Database '{self.db_name}' created successfully")
            else:
                print(f"Database '{self.db_name}' already exists")

            cursor.close()
            conn.close()

        except Exception as e:
            print(f"Error creating database: {e}")


    # ========================================================
    # 5. GET DATABASE CONNECTION
    # ========================================================

    def get_connection(self):
        return psycopg2.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            database=self.db_name
        )


    # ========================================================
    # 6. ENABLE PGVECTOR AND CREATE TABLE
    # ========================================================

    def setup_pgvector(self):
        """Setup pgvector extension and create tables."""

        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            # Enable pgvector extension
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")

            # Create cocktails table
            # all-MiniLM-L6-v2 produces 384-dimensional vectors
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS cocktails (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    ingredients TEXT NOT NULL,
                    recipe TEXT,
                    glass VARCHAR(100),
                    category VARCHAR(100),
                    iba VARCHAR(100),
                    alcoholic VARCHAR(50),
                    embedding vector(384)
                )
                """
            )

            # Create IVFFlat index using cosine distance
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS cocktails_embedding_idx
                ON cocktails USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
                """
            )

            conn.commit()
            cursor.close()
            conn.close()

            print("Database tables and pgvector extension set up successfully")

        except Exception as e:
            print(f"Error setting up pgvector: {e}")


# ============================================================
# 7. DATA PROCESSING
# ============================================================
class CocktailDataProcessor:

    # ========================================================
    # 8. INITIALIZE EMBEDDING MODEL
    # ========================================================

    def __init__(self):
        self.model_name = os.getenv("MODEL_NAME", "all-MiniLM-L6-v2")
        self.model = SentenceTransformer(self.model_name)
        self.db_setup = DatabaseSetup()


    # ========================================================
    # 9. CLEAN AND PREPROCESS DATA
    # ========================================================
    def clean_data(self, df):
        """Clean and preprocess the cocktail data."""

        # Auto-detect column names
        # The lecture supports two possible dataset formats
        name_col = "name" if "name" in df.columns else "strDrink"
        category_col = "category" if "category" in df.columns else "strCategory"
        alcoholic_col = "alcoholic" if "alcoholic" in df.columns else "strAlcoholic"
        glass_col = "glassType" if "glassType" in df.columns else "strGlass"
        instructions_col = "instructions" if "instructions" in df.columns else "strInstructions"

        print(
            f"Detected columns: name='{name_col}', "
            f"category='{category_col}', "
            f"alcoholic='{alcoholic_col}', "
            f"glass='{glass_col}'"
        )

        # Remove duplicated cocktails
        if name_col in df.columns:
            df = df.drop_duplicates(subset=[name_col])
            print(f"After removing duplicates: {len(df)} cocktails")

        # Handle missing values
        df = df.fillna("")

        # Create combined text for embedding
        df["combined_text"] = ""

        # Cocktail name
        if name_col in df.columns:
            df["combined_text"] += df[name_col].astype(str) + " "

        # Category
        if category_col in df.columns:
            df["combined_text"] += df[category_col].astype(str) + " "

        # Alcoholic / Non-Alcoholic
        if alcoholic_col in df.columns:
            df["combined_text"] += df[alcoholic_col].astype(str) + " "

        # Glass type
        if glass_col in df.columns:
            df["combined_text"] += df[glass_col].astype(str) + " "

        # Ingredients
        if "ingredients" in df.columns:
            # New dataset format
            df["combined_text"] += df["ingredients"].astype(str) + " "
        else:
            # Old dataset format: strIngredient1, strIngredient2, ...
            ingredient_cols = [
                col for col in df.columns
                if col.startswith("strIngredient")
            ]

            for col in ingredient_cols:
                df["combined_text"] += df[col].astype(str) + " "

        # Instructions
        if instructions_col in df.columns:
            df["combined_text"] += df[instructions_col].astype(str) + " "

        # Clean repeated whitespace
        df["combined_text"] = (
            df["combined_text"]
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

        print("Sample combined text:", df["combined_text"].iloc[0][:100], "...")

        return df


    # ========================================================
    # 10. GENERATE TEXT EMBEDDINGS
    # ========================================================
    def generate_embeddings(self, texts):
        """
        Generate embeddings for cocktail descriptions.

        Note:
        The PDF calls this method later but does not display
        its implementation in the 'Generate embeddings' section.
        """
        embeddings = self.model.encode(texts)
        return embeddings


    # ========================================================
    # 11. CREATE READABLE COCKTAIL RECIPE
    # ========================================================
    def create_recipe_text(self, row):
        """Create a readable recipe from the row data."""

        # Auto-detect column names
        name_col = "name" if "name" in row else "strDrink"
        category_col = "category" if "category" in row else "strCategory"
        alcoholic_col = "alcoholic" if "alcoholic" in row else "strAlcoholic"
        glass_col = "glassType" if "glassType" in row else "strGlass"
        instructions_col = "instructions" if "instructions" in row else "strInstructions"

        # Basic cocktail information
        recipe = f"Drink: {row.get(name_col, '')}\n"
        recipe += f"Category: {row.get(category_col, '')}\n"
        recipe += f"Type: {row.get(alcoholic_col, '')}\n"
        recipe += f"Glass: {row.get(glass_col, '')}\n"

        if row.get(instructions_col):
            recipe += f"Instructions: {row[instructions_col]}\n"

        recipe += "Ingredients:\n"

        # New dataset format: ingredients stored as string/list
        if "ingredients" in row and row["ingredients"]:
            try:
                ingredients_str = row["ingredients"]

                if ingredients_str.startswith("["):
                    ingredients = ast.literal_eval(ingredients_str)
                else:
                    ingredients = [ingredients_str]

                # Parse measures if available
                measures = []

                if "ingredientMeasures" in row and row["ingredientMeasures"]:
                    measures_str = row["ingredientMeasures"]

                    if measures_str.startswith("["):
                        measures = ast.literal_eval(measures_str)
                    else:
                        measures = [measures_str]

                # Combine ingredients and measures
                for i, ingredient in enumerate(ingredients):
                    if ingredient and str(ingredient).strip() and str(ingredient).strip() != "None":
                        if i < len(measures) and measures[i] and str(measures[i]).strip() != "None":
                            recipe += f"- {measures[i]} {ingredient}\n"
                        else:
                            recipe += f"- {ingredient}\n"

            except Exception:
                # Fallback: treat ingredients as simple text
                recipe += f"- {row['ingredients']}\n"

        # Old dataset format:
        # strIngredient1, strIngredient2, ...
        # strMeasure1, strMeasure2, ...
        else:
            for i in range(1, 16):
                ingredient = row.get(f"strIngredient{i}")
                measure = row.get(f"strMeasure{i}")

                if ingredient and str(ingredient).strip() and str(ingredient).strip() != "nan":
                    if measure and str(measure).strip() and str(measure).strip() != "nan":
                        recipe += f"- {measure} {ingredient}\n"
                    else:
                        recipe += f"- {ingredient}\n"

        return recipe


    # ========================================================
    # 12. EXTRACT INGREDIENT LIST
    # ========================================================
    def get_ingredients_list(self, row):
        """Extract ingredients as a comma-separated string."""

        ingredients = []

        # New dataset format
        if "ingredients" in row and row["ingredients"]:
            try:
                ingredients_str = row["ingredients"]

                if ingredients_str.startswith("["):
                    ingredients_list = ast.literal_eval(ingredients_str)

                    for ingredient in ingredients_list:
                        if ingredient and str(ingredient).strip() and str(ingredient).strip() != "None":
                            ingredients.append(str(ingredient).strip())

                # Single ingredient as string
                elif ingredients_str.strip():
                    ingredients.append(ingredients_str.strip())

            except Exception:
                # Fallback: treat ingredients as simple text
                if row["ingredients"].strip():
                    ingredients.append(row["ingredients"].strip())

        # Old dataset format
        else:
            for i in range(1, 16):
                ingredient = row.get(f"strIngredient{i}")

                if ingredient and str(ingredient).strip() and str(ingredient).strip() != "nan":
                    ingredients.append(str(ingredient).strip())

        return ", ".join(ingredients)


    # ========================================================
    # 13. STORE COCKTAILS + EMBEDDINGS IN POSTGRESQL
    # ========================================================
    def store_cocktails(self, df):
        """Store cocktails with embeddings in the database."""

        try:
            conn = self.db_setup.get_connection()
            cursor = conn.cursor()

            # Clear existing cocktail data
            cursor.execute("DELETE FROM cocktails")

            # Generate all embeddings at once
            print(f"Generating embeddings for {len(df)} cocktails...")
            all_embeddings = self.generate_embeddings(df["combined_text"].tolist())

            print("Storing cocktails in database...")

            for idx, (_, row) in enumerate(df.iterrows()):
                # Get pre-computed embedding
                embedding = all_embeddings[idx]

                # Auto-detect fields
                name_col = "name" if "name" in row else "strDrink"
                category_col = "category" if "category" in row else "strCategory"
                alcoholic_col = "alcoholic" if "alcoholic" in row else "strAlcoholic"
                glass_col = "glassType" if "glassType" in row else "strGlass"

                # Prepare cocktail data
                name = row.get(name_col, "")
                ingredients = self.get_ingredients_list(row)
                recipe = self.create_recipe_text(row)
                glass = row.get(glass_col, "")
                category = row.get(category_col, "")
                iba = row.get("strIBA", "")
                alcoholic = row.get(alcoholic_col, "")

                # Insert cocktail + embedding
                cursor.execute(
                    """
                    INSERT INTO cocktails (
                        name, ingredients, recipe, glass,
                        category, iba, alcoholic, embedding
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        name,
                        ingredients,
                        recipe,
                        glass,
                        category,
                        iba,
                        alcoholic,
                        embedding.tolist()
                    )
                )

                if (idx + 1) % 100 == 0:
                    print(f"Stored {idx + 1} cocktails...")

            conn.commit()
            cursor.close()
            conn.close()

            print(f"Successfully stored {len(df)} cocktails in the database")

        except Exception as e:
            print(f"Error storing cocktails: {e}")

            if "conn" in locals():
                conn.rollback()
                conn.close()


# ============================================================
# 14. COCKTAIL RECOMMENDATION SYSTEM
# ============================================================

class CocktailRecommender:

    # ========================================================
    # 15. INITIALIZE EMBEDDING MODEL
    # ========================================================

    def __init__(self):
        # Use the same embedding model as the stored cocktail vectors
        self.model_name = os.getenv("MODEL_NAME", "all-MiniLM-L6-v2")
        self.model = SentenceTransformer(self.model_name)
        self.db_setup = DatabaseSetup()


    # ========================================================
    # 16. CREATE USER QUERY EMBEDDING
    # ========================================================

    def get_user_preferences_embedding(self, preferences):
        preference_text = " ".join(preferences)
        embedding = self.model.encode([preference_text])[0]
        return embedding


    # ========================================================
    # 17. VECTOR SIMILARITY SEARCH
    # ========================================================

    def search_similar_cocktails(
        self,
        query_embedding,
        limit=10,
        similarity_threshold=0.3
    ):
        """
        Search for similar cocktails using vector similarity.
        """

        try:
            conn = self.db_setup.get_connection()
            cursor = conn.cursor()

            # Check whether database contains cocktails
            cursor.execute("SELECT COUNT(*) FROM cocktails")
            count = cursor.fetchone()[0]

            print(f"Database contains {count} cocktails")

            if count == 0:
                print("Warning: No cocktails found in database.")
                cursor.close()
                conn.close()
                return []

            # NumPy embedding -> Python list
            if hasattr(query_embedding, "tolist"):
                embedding_list = query_embedding.tolist()
            else:
                embedding_list = list(query_embedding)

            # pgvector <=> means cosine distance
            # similarity = 1 - cosine distance
            cursor.execute(
                """
                SELECT
                    id,
                    name,
                    ingredients,
                    recipe,
                    glass,
                    category,
                    iba,
                    alcoholic,
                    1 - (embedding <=> %s::vector) AS similarity
                FROM cocktails
                WHERE 1 - (embedding <=> %s::vector) > %s
                ORDER BY similarity DESC
                LIMIT %s
                """,
                (
                    embedding_list,
                    embedding_list,
                    similarity_threshold,
                    limit
                )
            )

            results = cursor.fetchall()

            print(
                f"Found {len(results)} cocktails "
                f"with similarity > {similarity_threshold}"
            )

            cursor.close()
            conn.close()

            return results

        except Exception as e:
            print(f"Error searching cocktails: {e}")
            traceback.print_exc()
            return []


    # ========================================================
    # 18. RECOMMEND BY INGREDIENTS
    # ========================================================

    def recommend_by_ingredients(self, ingredients, limit=10):
        ingredients_text = f"cocktail with {' and '.join(ingredients)}"

        query_embedding = self.get_user_preferences_embedding(
            [ingredients_text]
        )

        return self.search_similar_cocktails(
            query_embedding,
            limit
        )


    # ========================================================
    # 19. RECOMMEND BY STYLE
    # ========================================================

    def recommend_by_style(self, style_preferences, limit=10):
        style_text = f"cocktail that is {' and '.join(style_preferences)}"

        query_embedding = self.get_user_preferences_embedding(
            [style_text]
        )

        return self.search_similar_cocktails(
            query_embedding,
            limit
        )


    # ========================================================
    # 20. RECOMMEND BY OCCASION
    # ========================================================

    def recommend_by_occasion(self, occasion, limit=10):
        occasion_text = f"cocktail for {occasion}"

        query_embedding = self.get_user_preferences_embedding(
            [occasion_text]
        )

        return self.search_similar_cocktails(
            query_embedding,
            limit
        )


    # ========================================================
    # 21. RECOMMEND USING MIXED PREFERENCES
    # ========================================================

    def recommend_by_mixed_preferences(
        self,
        ingredients=None,
        style=None,
        occasion=None,
        alcoholic_preference=None,
        limit=10
    ):
        preferences = []

        if ingredients:
            preferences.append(f"contains {' and '.join(ingredients)}")

        if style:
            preferences.append(f"is {' and '.join(style)}")

        if occasion:
            preferences.append(f"perfect for {occasion}")

        if alcoholic_preference:
            preferences.append(f"is {alcoholic_preference}")

        if not preferences:
            return []

        query_embedding = self.get_user_preferences_embedding(preferences)

        return self.search_similar_cocktails(
            query_embedding,
            limit
        )


    # ========================================================
    # 22. SEARCH COCKTAIL BY NAME
    # ========================================================

    def get_cocktail_by_name(self, name):
        conn = self.db_setup.get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT
                id,
                name,
                ingredients,
                recipe,
                glass,
                category,
                iba,
                alcoholic
            FROM cocktails
            WHERE LOWER(name) LIKE LOWER(%s)
            LIMIT 5
            """,
            (f"%{name}%",)
        )

        results = cursor.fetchall()

        cursor.close()
        conn.close()

        return results


    # ========================================================
    # 23. GET RANDOM COCKTAILS
    # ========================================================

    def get_random_cocktails(self, limit=5):
        conn = self.db_setup.get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT
                id,
                name,
                ingredients,
                recipe,
                glass,
                category,
                iba,
                alcoholic
            FROM cocktails
            ORDER BY RANDOM()
            LIMIT %s
            """,
            (limit,)
        )

        results = cursor.fetchall()

        cursor.close()
        conn.close()

        return results


    # ========================================================
    # 24. FILTER COCKTAILS BY CATEGORY
    # ========================================================

    def get_cocktails_by_category(self, category, limit=10):
        conn = self.db_setup.get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT
                id,
                name,
                ingredients,
                recipe,
                glass,
                category,
                iba,
                alcoholic
            FROM cocktails
            WHERE LOWER(category) LIKE LOWER(%s)
            ORDER BY name
            LIMIT %s
            """,
            (f"%{category}%", limit)
        )

        results = cursor.fetchall()

        cursor.close()
        conn.close()

        return results


# ============================================================
# 25. EXAMPLE PIPELINE
# ============================================================


# ------------------------------------------------------------
# STEP 1 - Setup PostgreSQL + pgvector
# ------------------------------------------------------------

# db_setup = DatabaseSetup()
# db_setup.create_database()
# db_setup.setup_pgvector()


# ------------------------------------------------------------
# STEP 2 - Load dataset
# ------------------------------------------------------------

# df = pd.read_csv("cocktails.csv")


# ------------------------------------------------------------
# STEP 3 - Clean and preprocess data
# ------------------------------------------------------------

# processor = CocktailDataProcessor()
# df = processor.clean_data(df)


# ------------------------------------------------------------
# STEP 4 - Generate embeddings and store cocktails
# ------------------------------------------------------------

# processor.store_cocktails(df)


# ------------------------------------------------------------
# STEP 5 - Create recommendation system
# ------------------------------------------------------------

# recommender = CocktailRecommender()


# ------------------------------------------------------------
# STEP 6 - Recommend by ingredients
# ------------------------------------------------------------

# results = recommender.recommend_by_ingredients(
#     ["gin", "lemon"],
#     limit=5
# )
#
# print(results)


# ------------------------------------------------------------
# STEP 7 - Recommend by style
# ------------------------------------------------------------

# results = recommender.recommend_by_style(
#     ["sweet", "refreshing"],
#     limit=5
# )


# ------------------------------------------------------------
# STEP 8 - Recommend by occasion
# ------------------------------------------------------------

# results = recommender.recommend_by_occasion(
#     "summer party",
#     limit=5
# )


# ------------------------------------------------------------
# STEP 9 - Recommend by mixed preferences
# ------------------------------------------------------------

# results = recommender.recommend_by_mixed_preferences(
#     ingredients=["gin", "lime"],
#     style=["sweet", "refreshing"],
#     occasion="summer party",
#     alcoholic_preference="Alcoholic",
#     limit=5
# )


# ============================================================
# 26. STREAMLIT APP
# ============================================================
#
# The PDF does not provide the Streamlit implementation.
# It only references the complete application source code
# from the external repository.
# ============================================================