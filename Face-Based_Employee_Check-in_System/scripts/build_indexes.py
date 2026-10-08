from face_checkin.system import FaceCheckinSystem


def main():
    system = FaceCheckinSystem()

    stats = system.build_indexes(save=True)

    print("\n==========================")
    print("BUILD COMPLETE")
    print("==========================")

    print(f"Raw pixel vectors: {stats['raw_vectors']}")
    print(f"Face embeddings: {stats['face_embeddings']}")


if __name__ == "__main__":
    main()