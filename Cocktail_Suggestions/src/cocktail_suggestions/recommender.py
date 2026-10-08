from pgvector import Vector
from sentence_transformers import SentenceTransformer

from cocktail_suggestions.config import MODEL_NAME
from cocktail_suggestions.database import DatabaseSetup


class CocktailRecommender:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)
        self.db_setup = DatabaseSetup()

    def get_user_preferences_embedding(self, preferences):
        """Convert user preferences into one embedding."""
        preference_text = " ".join(preferences)
        return self.model.encode([preference_text])[0]

    def search_similar_cocktails(self, query_embedding, limit=10, similarity_threshold=0.3):
        """Search cocktails using cosine similarity."""
        conn = None
        cursor = None

        try:
            conn = self.db_setup.get_connection()
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM cocktails")
            count = cursor.fetchone()[0]

            if count == 0:
                print("No cocktails found in database.")
                return []

            query_vector = Vector(query_embedding)

            cursor.execute("""
                SELECT
                    id,
                    name,
                    ingredients,
                    recipe,
                    glass,
                    category,
                    iba,
                    alcoholic,
                    1 - (embedding <=> %s) AS similarity
                FROM cocktails
                WHERE 1 - (embedding <=> %s) > %s
                ORDER BY similarity DESC
                LIMIT %s
            """, (
                query_vector,
                query_vector,
                similarity_threshold,
                limit
            ))

            return cursor.fetchall()

        except Exception as e:
            print(f"Error searching cocktails: {e}")
            return []

        finally:
            if cursor:
                cursor.close()

            if conn:
                conn.close()

    def recommend_by_ingredients(self, ingredients, limit=10):
        text = f"cocktail with {' and '.join(ingredients)}"
        embedding = self.get_user_preferences_embedding([text])
        return self.search_similar_cocktails(embedding, limit)

    def recommend_by_style(self, style_preferences, limit=10):
        text = f"cocktail that is {' and '.join(style_preferences)}"
        embedding = self.get_user_preferences_embedding([text])
        return self.search_similar_cocktails(embedding, limit)

    def recommend_by_occasion(self, occasion, limit=10):
        text = f"cocktail for {occasion}"
        embedding = self.get_user_preferences_embedding([text])
        return self.search_similar_cocktails(embedding, limit)

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

        embedding = self.get_user_preferences_embedding(preferences)
        return self.search_similar_cocktails(embedding, limit)

    def get_cocktail_by_name(self, name):
        conn = self.db_setup.get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT id, name, ingredients, recipe, glass, category, iba, alcoholic
                FROM cocktails
                WHERE name ILIKE %s
                LIMIT 5
            """, (f"%{name}%",))

            return cursor.fetchall()

        finally:
            cursor.close()
            conn.close()

    def get_random_cocktails(self, limit=5):
        conn = self.db_setup.get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT id, name, ingredients, recipe, glass, category, iba, alcoholic
                FROM cocktails
                ORDER BY RANDOM()
                LIMIT %s
            """, (limit,))

            return cursor.fetchall()

        finally:
            cursor.close()
            conn.close()

    def get_cocktails_by_category(self, category, limit=10):
        conn = self.db_setup.get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT id, name, ingredients, recipe, glass, category, iba, alcoholic
                FROM cocktails
                WHERE category ILIKE %s
                ORDER BY name
                LIMIT %s
            """, (f"%{category}%", limit))

            return cursor.fetchall()

        finally:
            cursor.close()
            conn.close()

    def format_cocktail_result(self, result):
        cocktail = {
            "id": result[0],
            "name": result[1],
            "ingredients": result[2],
            "recipe": result[3],
            "glass": result[4],
            "category": result[5],
            "iba": result[6],
            "alcoholic": result[7]
        }

        if len(result) > 8 and result[8] is not None:
            cocktail["similarity"] = round(float(result[8]) * 100, 1)

        return cocktail