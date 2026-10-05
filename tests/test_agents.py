import pytest

from app.agents import AgentInputError, arithmetic_agent, weekday_agent


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("Calcula: 12 * (3 + 2)", 60),
        ("Calculate: 12 * (3 + 2)", 60),
        ("-2.5 + 4 / 2", -0.5),
        ("What is 2 + 2?", 4),
    ],
)
def test_arithmetic(message, expected):
    assert arithmetic_agent(message) == expected


@pytest.mark.parametrize(
    "message",
    [
        "__import__('os').system('ls')",
        "2 ** 1000000",
        "True + 1",
        "2 + 3 and 4 * 5",
        "1 / 0",
        "1e300 * 1e300",
        "1000000000001",
        "1 + " * 60 + "1",
        "(" * 300 + "1" + ")" * 300,
    ],
)
def test_arithmetic_rejects_unsafe_or_unsupported_input(message):
    with pytest.raises(AgentInputError):
        arithmetic_agent(message)


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("What day was 2024-01-01?", "Monday"),
        ("Date: 2024-01-02", "Tuesday"),
        ("Date: 2024-01-03", "Wednesday"),
        ("Date: 2024-02-29", "Thursday"),
        ("Date: 2024-01-05", "Friday"),
        ("Date: 2024-01-06", "Saturday"),
        ("Date: 2024-01-07", "Sunday"),
    ],
)
def test_weekday(message, expected):
    assert weekday_agent(message) == expected


@pytest.mark.parametrize("message", ["2024-02-30", "tomorrow", "2024-01-01 and 2024-01-02"])
def test_weekday_requires_one_valid_iso_date(message):
    with pytest.raises(AgentInputError):
        weekday_agent(message)
