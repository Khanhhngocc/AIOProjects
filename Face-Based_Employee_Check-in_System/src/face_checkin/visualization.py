import matplotlib.pyplot as plt
from PIL import Image


def display_results(query_image_path, results, metric):
    if not results:
        print("No results found.")
        return

    # ========================================================
    # QUERY IMAGE
    # ========================================================

    query_img = Image.open(query_image_path).convert("RGB")

    plt.figure(figsize=(4, 4))
    plt.imshow(query_img)
    plt.title("Query Image")
    plt.axis("off")
    plt.show()

    # ========================================================
    # MATCHING IMAGES
    # ========================================================

    plt.figure(figsize=(3 * len(results), 4))

    for i, result in enumerate(results):
        img = Image.open(result["image_path"]).convert("RGB")
        score = result[metric]

        plt.subplot(1, len(results), i + 1)
        plt.imshow(img)
        plt.title(
            f"{result['label']}\n"
            f"{metric}: {score:.4f}"
        )
        plt.axis("off")

    plt.tight_layout()
    plt.show()