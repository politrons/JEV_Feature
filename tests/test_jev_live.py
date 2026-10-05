import asyncio

import pytest
from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

from app.jev import classify
from app.main import Settings
from app.models import Intent


@pytest.mark.live
@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("Translate into English: Buenos dias, como estas?", Intent.TRANSLATE_ES_EN),
        ("Calculate: 12 * (3 + 2)", Intent.ARITHMETIC),
        ("What day of the week was 2024-01-01?", Intent.WEEKDAY),
        ("Write a poem about the sea.", Intent.OUT_OF_SCOPE),
        ("Help me.", Intent.NEEDS_CLARIFICATION),
        ("Translate into English: El lunes calculamos dos mas dos.", Intent.TRANSLATE_ES_EN),
        ("Translate hola into English and calculate 2 + 2.", Intent.NEEDS_CLARIFICATION),
    ],
)
def test_real_openjev_classification(message, expected):
    settings = Settings()

    async def run():
        async with AsyncTypeSafeClient(
            api_key="local-openjev",
            base_url=str(settings.openjev_base_url),
            model=settings.openjev_model,
            timeout=60,
            retry=RetryPolicy(max_retries=0),
        ) as client:
            decision = await classify(client, message)
        assert decision.intent == expected
        assert set(decision.probabilities) == set(Intent)

    asyncio.run(run())
