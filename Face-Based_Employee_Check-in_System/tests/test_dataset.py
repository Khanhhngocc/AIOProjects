from face_checkin.dataset import load_dataset


def test_load_dataset(tmp_path):
    image_1 = tmp_path / "Avatar_Alice.jpg"
    image_2 = tmp_path / "Bob.png"
    text_file = tmp_path / "notes.txt"

    image_1.touch()
    image_2.touch()
    text_file.touch()

    records = load_dataset(tmp_path)

    assert len(records) == 2

    labels = [
        record["label"]
        for record in records
    ]

    assert "Alice" in labels
    assert "Bob" in labels