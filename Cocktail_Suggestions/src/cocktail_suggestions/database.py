import psycopg
from psycopg import sql
from pgvector.psycopg import register_vector

from cocktail_suggestions.config import DATABASE_URL, DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME


class DatabaseSetup:
    def _connect(self, dbname=None, autocommit=False):
        if DATABASE_URL:
            return psycopg.connect(DATABASE_URL, autocommit=autocommit)

        return psycopg.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            dbname=dbname or DB_NAME,
            autocommit=autocommit
        )


    def create_database(self):
        """Create database if it does not exist."""
        if DATABASE_URL:
            print("Remote database detected. Skipping database creation.")
            return

        with self._connect(dbname="postgres", autocommit=True) as conn:
            exists = conn.execute(
                "SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s",
                (DB_NAME,)
            ).fetchone()

            if not exists:
                conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(DB_NAME)))
                print(f"Database '{DB_NAME}' created successfully.")
            else:
                print(f"Database '{DB_NAME}' already exists.")


    def get_connection(self):
        """Connect to the application database and register pgvector."""
        conn = self._connect()
        register_vector(conn)
        return conn

    def setup_pgvector(self):
        """Enable pgvector and create cocktails table."""
        conn = None

        try:
            conn = self._connect()

            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.commit()

            register_vector(conn)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS cocktails (
                    id BIGSERIAL PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    ingredients TEXT NOT NULL,
                    recipe TEXT,
                    glass VARCHAR(100),
                    category VARCHAR(100),
                    iba VARCHAR(100),
                    alcoholic VARCHAR(50),
                    embedding vector(384)
                )
            """)

            conn.commit()

            print("Database table and pgvector set up successfully.")

        except Exception as e:
            if conn is not None:
                conn.rollback()
            raise

        finally:
            if conn is not None:
                conn.close()