import os
from typing import Any
from dotenv import load_dotenv

from pydantic import BaseModel, Field

from graphiti_core.graphiti import Graphiti
from graphiti_core.driver.neo4j_driver import Neo4jDriver

load_dotenv()

NEO4J_URI = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
NEO4J_USER = os.getenv('NEO4J_USER', 'neo4j')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD', 'test')


class Person(BaseModel):
    """A Person represents an individual mentioned in conversations or notes.

    Instructions for identifying and extracting people:
    - Look for explicit names ("Nicola Stern", "Felix", "Scott from Globalstar")
    - IMPORTANT: When you see pronouns "I", "me", "my", "myself" - create a Person entity with name "Daniel Walder"
    - Create separate entities for each distinct person mentioned
    - Use full names when provided, otherwise use the name as stated
    - Always create "Daniel Walder" entity when first-person pronouns are used
    """

    company: str = Field(
        ...,
        description='The company the person works at.',
    )
    description: str = Field(
        ...,
        description='Brief description of the person. Only use information mentioned in the context to write this description.',
    )


class Task(BaseModel):
    """A Task represents an action item or TODO that needs to be completed by Daniel Walder.

    Look for phrases like:
    - I need to...
    - Reminde me to...
    - I have to...
    """

    need_to_be_done_by: str = Field(
        ...,
        description='The time by which this task needs to be done. Use ISO 8601 format (YYYY-MM-DDTHH:MM:SS.SSSSSSZ).',
    )
    done_by: str = Field(
        ...,
        description='The time by which this task was done. Use ISO 8601 format (YYYY-MM-DDTHH:MM:SS.SSSSSSZ). This should be the reference time if the new information indicates that the task has now been completed. For example, if the task was to paint the house and the new information states that the house has been painted, then the done_by attribute should be populated with the reference time of the new information.',
    )
    cancelled_by: str = Field(
        ...,
        description='The time by which this task was cancelled. Use ISO 8601 format (YYYY-MM-DDTHH:MM:SS.SSSSSSZ).',
    )


class PKMS:
    def __init__(self):
        driver = Neo4jDriver(
            uri=NEO4J_URI,
            user=NEO4J_USER,
            password=NEO4J_PASSWORD,
        )
        self.graphiti = Graphiti(graph_driver=driver)

    async def __aenter__(self) -> "PKMS":
        await self.graphiti.build_indices_and_constraints()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ):
        # await self._clear_database()
        await self.graphiti.close()

    async def _clear_database(self):
        search_results = await self.graphiti.search_(
            query='*',
            group_ids=["test"],
        )
        for node in search_results.nodes:
            await node.delete(self.graphiti.driver)
        for episode in search_results.episodes:
            await episode.delete(self.graphiti.driver)

