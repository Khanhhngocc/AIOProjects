from spam_classifier.classifier import majority_vote


def test_majority_vote_spam():
    predictions = [
        "spam",
        "spam",
        "ham"
    ]

    prediction, distribution = majority_vote(
        predictions
    )

    assert prediction == "spam"

    assert distribution == {
        "ham": 1,
        "spam": 2
    }


def test_majority_vote_ham():
    predictions = [
        "ham",
        "ham",
        "spam"
    ]

    prediction, distribution = majority_vote(
        predictions
    )

    assert prediction == "ham"

    assert distribution == {
        "ham": 2,
        "spam": 1
    }