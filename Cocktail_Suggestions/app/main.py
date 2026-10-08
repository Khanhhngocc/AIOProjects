from cocktail_suggestions.recommender import CocktailRecommender


def print_results(recommender, results):
    if not results:
        print("\nNo cocktails found.")
        return

    for result in results:
        cocktail = recommender.format_cocktail_result(result)

        print("\n" + "=" * 50)
        print(f"Name: {cocktail['name']}")
        print(f"Ingredients: {cocktail['ingredients']}")
        print(f"Category: {cocktail['category']}")
        print(f"Alcohol: {cocktail['alcoholic']}")

        if "similarity" in cocktail:
            print(f"Match: {cocktail['similarity']}%")


def main():
    recommender = CocktailRecommender()

    while True:
        print("""
================================
COCKTAIL SUGGESTION SYSTEM
================================
1. Search by name
2. Recommend by ingredients
3. Recommend by style
4. Recommend by occasion
5. Mixed preferences
6. Random cocktails
0. Exit
""")

        choice = input("Choose an option: ").strip()

        if choice == "1":
            name = input("Cocktail name: ")
            results = recommender.get_cocktail_by_name(name)

        elif choice == "2":
            text = input("Ingredients (comma separated): ")
            ingredients = [item.strip() for item in text.split(",") if item.strip()]
            results = recommender.recommend_by_ingredients(ingredients)

        elif choice == "3":
            text = input("Style (sweet, sour, refreshing...): ")
            styles = [item.strip() for item in text.split(",") if item.strip()]
            results = recommender.recommend_by_style(styles)

        elif choice == "4":
            occasion = input("Occasion: ")
            results = recommender.recommend_by_occasion(occasion)

        elif choice == "5":
            ingredient_text = input("Ingredients: ")
            style_text = input("Style: ")
            occasion = input("Occasion: ")
            alcoholic = input("Alcohol preference: ")

            ingredients = [x.strip() for x in ingredient_text.split(",") if x.strip()]
            styles = [x.strip() for x in style_text.split(",") if x.strip()]

            results = recommender.recommend_by_mixed_preferences(
                ingredients=ingredients or None,
                style=styles or None,
                occasion=occasion or None,
                alcoholic_preference=alcoholic or None
            )

        elif choice == "6":
            results = recommender.get_random_cocktails()

        elif choice == "0":
            print("Goodbye!")
            break

        else:
            print("Invalid option.")
            continue

        print_results(recommender, results)


if __name__ == "__main__":
    main()