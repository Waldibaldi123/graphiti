from datetime import datetime, timezone
from typing import AsyncGenerator

import pytest
import pytest_asyncio

from pkms.pkms import PKMS, Task, Person
from graphiti_core.nodes import EpisodeType


async def setup_module():
    async with PKMS() as pkms:
        await pkms._clear_database()


import asyncio
asyncio.run(setup_module())


@pytest_asyncio.fixture
async def pkms() -> AsyncGenerator[PKMS, None]:
    async with PKMS() as pkms:
        # Disable LLM client caching to see all requests
        pkms.graphiti.llm_client.cache_enabled = False
        yield pkms


@pytest.mark.parametrize("input_text", [
    "I need to water Stephan Walder's plants tonight.",
    "I watered Stephan Walder's plants.",
    # "I need to text Leila Zahlut back about our breakfast next week.",
    # "Remind me to text back Felix about the ice bath event.",
    # "I need to respond to Nicola Stern's email regarding the new contract for BTV by today.",
    # "I have my weekly syncup with Scott from Globalstar on Wednesday. I need to bring up that I currently cannot login into my Vanguard account.",
])
@pytest.mark.asyncio
async def test_add_todo(
    pkms: PKMS,
    input_text: str,
):
    ts = datetime.now(timezone.utc)
    await pkms.graphiti.add_episode(
        name=f"{ts}_todo",
        episode_body=input_text,
        source=EpisodeType.text,
        group_id="test",
        source_description="TODO Note",
        reference_time=ts,
        entity_types={"person": Person},
        edge_types={"TASK_OR_NOT": Task},
    )
    assert True

