from pathlib import Path

from face_checkin.system import FaceCheckinSystem
from face_checkin.visualization import display_results


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def print_results(results, metric):
    if not results:
        print("No results found.")
        return

    print("\nResults:")

    for rank, result in enumerate(results, start=1):
        print(
            f"{rank}. {result['label']} | "
            f"{metric}={result[metric]:.4f} | "
            f"{result['image_path']}"
        )


def get_query_path():
    query = input("Query image path: ").strip()

    path = Path(query).expanduser()

    if not path.exists():
        raise FileNotFoundError(
            f"Query image not found: {path}"
        )

    return path


# ============================================================
# MENU ACTIONS
# ============================================================

def build_indexes(system):
    stats = system.build_indexes(save=True)

    print("\nIndexes saved.")
    print(f"Raw vectors: {stats['raw_vectors']}")
    print(f"Face embeddings: {stats['face_embeddings']}")


def search_raw(system):
    query_path = get_query_path()

    results = system.search_raw(
        query_path,
        k=5
    )

    print_results(results, metric="distance")
    display_results(query_path, results, metric="distance")


def search_face(system):
    query_path = get_query_path()

    results = system.search_face(
        query_path,
        k=5
    )

    print_results(results, metric="similarity")
    display_results(query_path, results, metric="similarity")


def recognize_face(system):
    query_path = get_query_path()

    threshold = float(
        input("Similarity threshold: ")
    )

    result = system.recognize_face(
        query_path,
        threshold
    )

    if result is None:
        print("Employee not recognized.")
        return

    print("\nRecognized:")
    print(f"Employee: {result['label']}")
    print(f"Similarity: {result['similarity']:.4f}")


def recognize_raw(system):
    query_path = get_query_path()

    threshold = float(input("Distance threshold: "))

    result = system.recognize_raw(query_path, threshold)

    if result is None:
        print("Employee not recognized.")
        return

    print("\nRecognized:")
    print(f"Employee: {result['label']}")
    print(f"Distance: {result['distance']:.4f}")


# ============================================================
# MAIN
# ============================================================

def main():
    system = FaceCheckinSystem()

    while True:
        print("\n==========================================")
        print("FACE-BASED EMPLOYEE CHECK-IN SYSTEM")
        print("==========================================")
        print("1. Build / rebuild indexes")
        print("2. Search using raw pixels")
        print("3. Search using face embeddings")
        print("4. Recognize employee using face")
        print("5. Recognize employee using raw pixels")
        print("0. Exit")

        choice = input("\nChoose an option: ").strip()

        try:
            if choice == "1":
                build_indexes(system)

            elif choice == "2":
                search_raw(system)

            elif choice == "3":
                search_face(system)

            elif choice == "4":
                recognize_face(system)

            elif choice == "5":
                recognize_raw(system)

            elif choice == "0":
                print("Bye!")
                break

            else:
                print("Invalid option.")

        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()