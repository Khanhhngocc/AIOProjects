import streamlit as st
from html import escape

from cocktail_suggestions.recommender import CocktailRecommender


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Cocktail Suggestions",
    page_icon="🍹",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.html("""
<style>
.main-header {
    font-size: 3rem;
    font-weight: 750;
    text-align: center;
    margin-bottom: 0.2rem;
}

.main-subtitle {
    text-align: center;
    opacity: 0.7;
    font-size: 1.05rem;
    margin-bottom: 2rem;
}

.cocktail-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 22px 24px;
    border-radius: 16px;
    margin: 10px 0 8px 0;
    box-shadow: 0 5px 14px rgba(0, 0, 0, 0.12);
}

.cocktail-card h3 {
    color: white;
    margin: 0 0 12px 0;
}

.cocktail-card p {
    margin: 5px 0;
}

.similarity-score {
    display: inline-block;
    background: rgba(255, 255, 255, 0.22);
    padding: 5px 12px;
    margin-bottom: 12px;
    border-radius: 20px;
    font-size: 0.9rem;
    font-weight: 650;
}

.ingredients-container {
    margin: 8px 0 14px 0;
}

.ingredient-tag {
    display: inline-block;
    background: #ff9800;
    color: white;
    padding: 5px 10px;
    margin: 3px;
    border-radius: 14px;
    font-size: 0.83rem;
    font-weight: 500;
}
</style>
""")


# ============================================================
# RESOURCE
# ============================================================

@st.cache_resource
def get_recommender():
    """Create and cache the recommender."""
    return CocktailRecommender()


# ============================================================
# SESSION STATE
# ============================================================

def init_session_state():
    if "search_results" not in st.session_state:
        st.session_state.search_results = []

    if "last_search_type" not in st.session_state:
        st.session_state.last_search_type = ""

    if "has_searched" not in st.session_state:
        st.session_state.has_searched = False


# ============================================================
# DISPLAY
# ============================================================

def display_cocktail(cocktail):
    """Display one cocktail result."""
    name = escape(str(cocktail.get("name", "")))
    category = escape(str(cocktail.get("category", "")))
    alcoholic = escape(str(cocktail.get("alcoholic", "")))
    glass = escape(str(cocktail.get("glass", "")))

    similarity_html = ""
    if "similarity" in cocktail:
        similarity_html = (
            f'<div class="similarity-score">'
            f'Match: {cocktail["similarity"]}%'
            f'</div>'
        )

    card_html = f"""
<div class="cocktail-card">
    <h3>🍹 {name}</h3>
    {similarity_html}
    <p><strong>Category:</strong> {category or "N/A"}</p>
    <p><strong>Type:</strong> {alcoholic or "N/A"}</p>
    <p><strong>Glass:</strong> {glass or "N/A"}</p>
</div>
"""
    st.html(card_html)

    ingredients_text = cocktail.get("ingredients", "")
    if ingredients_text:
        ingredients = [
            ingredient.strip()
            for ingredient in ingredients_text.split(",")
            if ingredient.strip()
        ]

        ingredient_html = "".join(
            f'<span class="ingredient-tag">{escape(ingredient)}</span>'
            for ingredient in ingredients[:8]
        )

        st.html(f'<div class="ingredients-container">{ingredient_html}</div>')

    recipe = cocktail.get("recipe")
    if recipe:
        with st.expander("📖 View Recipe", expanded=False):
            st.text(recipe)


def display_results(recommender):
    """Display current search results."""
    results = st.session_state.search_results

    if results:
        st.divider()

        count = len(results)
        st.subheader(f"🍹 Found {count} cocktail{'s' if count != 1 else ''}")

        for result in results:
            cocktail = recommender.format_cocktail_result(result)
            display_cocktail(cocktail)

    elif st.session_state.has_searched:
        st.info("No cocktails found matching your criteria. Try adjusting your preferences.")


# ============================================================
# SEARCH FUNCTIONS
# ============================================================

def search_by_name(recommender):
    st.subheader("🔍 Search Cocktails by Name")
    st.caption("Search using a full or partial cocktail name.")

    with st.form("name_search_form"):
        cocktail_name = st.text_input(
            "Cocktail name",
            placeholder="e.g. Margarita, Mojito"
        )
        submitted = st.form_submit_button(
            "Search",
            type="primary",
            use_container_width=True
        )

    if submitted:
        if not cocktail_name.strip():
            st.warning("Please enter a cocktail name.")
            return

        with st.spinner("Searching..."):
            st.session_state.search_results = recommender.get_cocktail_by_name(cocktail_name.strip())

        st.session_state.has_searched = True


def search_by_ingredients(recommender, common_ingredients):
    st.subheader("🥃 Find Cocktails by Ingredients")
    st.caption("Choose common ingredients or enter your own.")

    with st.form("ingredient_search_form"):
        col1, col2 = st.columns(2)

        with col1:
            selected_common = st.multiselect("Quick select", common_ingredients)

        with col2:
            custom_ingredients = st.text_input(
                "Custom ingredients",
                placeholder="e.g. coconut, strawberry"
            )

        submitted = st.form_submit_button(
            "Find Cocktails",
            type="primary",
            use_container_width=True
        )

    if submitted:
        ingredients = selected_common.copy()

        if custom_ingredients:
            ingredients.extend([
                ingredient.strip()
                for ingredient in custom_ingredients.split(",")
                if ingredient.strip()
            ])

        if not ingredients:
            st.warning("Please select or enter at least one ingredient.")
            return

        with st.spinner("Finding perfect matches..."):
            st.session_state.search_results = recommender.recommend_by_ingredients(
                ingredients,
                limit=10
            )

        st.session_state.has_searched = True


def search_by_style(recommender):
    st.subheader("🎭 Find Cocktails by Style / Mood")
    st.caption("Choose one or more characteristics.")

    style_options = [
        "sweet", "sour", "bitter", "strong", "light",
        "fruity", "creamy", "refreshing", "exotic",
        "classic", "tropical"
    ]

    with st.form("style_search_form"):
        selected_styles = st.multiselect(
            "What kind of cocktail do you want?",
            style_options
        )

        submitted = st.form_submit_button(
            "Find Cocktails",
            type="primary",
            use_container_width=True
        )

    if submitted:
        if not selected_styles:
            st.warning("Please select at least one style.")
            return

        with st.spinner("Finding cocktails for your mood..."):
            st.session_state.search_results = recommender.recommend_by_style(
                selected_styles,
                limit=10
            )

        st.session_state.has_searched = True


def search_by_occasion(recommender):
    st.subheader("🎉 Find Cocktails by Occasion")

    occasion_options = [
        "",
        "party",
        "date night",
        "summer evening",
        "winter warmer",
        "brunch",
        "after dinner",
        "celebration",
        "relaxing at home"
    ]

    with st.form("occasion_search_form"):
        occasion = st.selectbox("What's the occasion?", occasion_options)

        submitted = st.form_submit_button(
            "Find Cocktails",
            type="primary",
            use_container_width=True
        )

    if submitted:
        if not occasion:
            st.warning("Please choose an occasion.")
            return

        with st.spinner("Finding suitable cocktails..."):
            st.session_state.search_results = recommender.recommend_by_occasion(
                occasion,
                limit=10
            )

        st.session_state.has_searched = True


def search_mixed_preferences(recommender, common_ingredients, alcoholic_options):
    st.subheader("🎲 Mixed Preferences")
    st.caption("Combine ingredients, style, occasion and alcohol preference.")

    with st.form("mixed_search_form"):
        col1, col2 = st.columns(2)

        with col1:
            ingredients = st.multiselect("Preferred ingredients", common_ingredients)
            styles = st.multiselect(
                "Style preferences",
                ["sweet", "sour", "strong", "light", "fruity", "refreshing"]
            )

        with col2:
            occasion = st.selectbox(
                "Occasion",
                ["", "party", "date night", "summer", "winter", "brunch"]
            )
            alcoholic_pref = st.selectbox(
                "Alcohol preference",
                [""] + alcoholic_options
            )

        submitted = st.form_submit_button(
            "Find My Perfect Cocktail",
            type="primary",
            use_container_width=True
        )

    if submitted:
        if not any([ingredients, styles, occasion, alcoholic_pref]):
            st.warning("Please choose at least one preference.")
            return

        with st.spinner("Analyzing your preferences..."):
            st.session_state.search_results = recommender.recommend_by_mixed_preferences(
                ingredients=ingredients or None,
                style=styles or None,
                occasion=occasion or None,
                alcoholic_preference=alcoholic_pref or None,
                limit=10
            )

        st.session_state.has_searched = True


def search_by_category(recommender):
    st.subheader("📂 Browse by Category")

    category_options = [
        "",
        "Ordinary Drink",
        "Cocktail",
        "Shot",
        "Coffee / Tea",
        "Homemade Liqueur",
        "Punch / Party Drink",
        "Beer",
        "Soft Drink"
    ]

    with st.form("category_search_form"):
        category = st.selectbox("Choose a category", category_options)

        submitted = st.form_submit_button(
            "Browse Cocktails",
            type="primary",
            use_container_width=True
        )

    if submitted:
        if not category:
            st.warning("Please select a category.")
            return

        with st.spinner("Loading category..."):
            st.session_state.search_results = recommender.get_cocktails_by_category(
                category,
                limit=10
            )

        st.session_state.has_searched = True


def random_discovery(recommender):
    st.subheader("🎰 Random Discovery")
    st.write("Not sure what you want? Let the app pick some cocktails for you.")

    if st.button("🎲 Surprise Me!", type="primary", use_container_width=True):
        with st.spinner("Picking cocktails..."):
            st.session_state.search_results = recommender.get_random_cocktails(limit=6)

        st.session_state.has_searched = True


# ============================================================
# MAIN
# ============================================================

def main():
    init_session_state()

    st.html("""
<h1 class="main-header">🍹 AI-Powered Cocktail Suggestions</h1>
<p class="main-subtitle">
    Discover cocktails using semantic search and vector similarity
</p>
""")

    try:
        recommender = get_recommender()
    except Exception as e:
        st.error(f"Could not initialize the recommender: {e}")
        st.info(
            "Make sure Docker/PostgreSQL is running, "
            "pgvector is enabled and the database has been set up."
        )
        return

    common_ingredients = [
        "vodka", "gin", "rum", "whiskey", "tequila", "bourbon",
        "lime", "lemon", "orange", "cranberry", "pineapple",
        "mint", "basil", "simple syrup", "triple sec", "vermouth"
    ]

    alcoholic_options = [
        "Alcoholic",
        "Non alcoholic",
        "Optional alcohol"
    ]

    with st.sidebar:
        st.header("🎯 Your Preferences")

        search_type = st.selectbox(
            "How would you like to find cocktails?",
            [
                "🔍 Search by Name",
                "🥃 By Ingredients",
                "🎭 By Style / Mood",
                "🎉 By Occasion",
                "🎲 Mixed Preferences",
                "📂 By Category",
                "🎰 Random Discovery"
            ]
        )

        st.divider()

        st.caption(
            "Cocktail recommendations use SentenceTransformer embeddings "
            "and pgvector similarity search."
        )

    if st.session_state.last_search_type != search_type:
        st.session_state.search_results = []
        st.session_state.has_searched = False
        st.session_state.last_search_type = search_type

    main_col, info_col = st.columns([2.3, 1], gap="large")

    with main_col:
        if search_type == "🔍 Search by Name":
            search_by_name(recommender)

        elif search_type == "🥃 By Ingredients":
            search_by_ingredients(recommender, common_ingredients)

        elif search_type == "🎭 By Style / Mood":
            search_by_style(recommender)

        elif search_type == "🎉 By Occasion":
            search_by_occasion(recommender)

        elif search_type == "🎲 Mixed Preferences":
            search_mixed_preferences(
                recommender,
                common_ingredients,
                alcoholic_options
            )

        elif search_type == "📂 By Category":
            search_by_category(recommender)

        elif search_type == "🎰 Random Discovery":
            random_discovery(recommender)

        display_results(recommender)

    with info_col:
        st.subheader("💡 Tips")

        with st.container(border=True):
            st.markdown("""
**Better recommendations**

- Be specific about ingredients.
- Combine several style preferences.
- Try different occasions.
- Search using partial cocktail names.
- Use Mixed Preferences for detailed queries.
""")

        st.subheader("🧠 How it works")

        with st.container(border=True):
            st.markdown("""
**1. Your preferences**

Your input is converted into text.

**2. AI Embedding**

`all-MiniLM-L6-v2` converts the text into a vector.

**3. Vector Search**

PostgreSQL + pgvector compares your vector with cocktail vectors.

**4. Results**

The most similar cocktails are returned first.
""")


if __name__ == "__main__":
    main()