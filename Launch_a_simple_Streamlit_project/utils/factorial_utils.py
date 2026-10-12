def factorial(number: int) -> int:
    """Calculate the factorial of a non-negative integer."""

    if number < 0:
        raise ValueError("Factorial is only defined for non-negative integers.")

    result = 1

    for value in range(2, number + 1):
        result *= value

    return result