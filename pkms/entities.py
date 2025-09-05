from datetime import datetime
from typing import Union, Literal, Annotated
from uuid import uuid4

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from .common import AgentDeps, generate_embedding


class BaseEntity(BaseModel):
    uuid: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    entity_type: str


class PersonEntity(BaseEntity):
    """A person or individual mentioned in the text.
    If there are first person personal pronouns such as "I", "me" or "myself",
    then ALWAYS extract a person with the name "Daniel"."""
    entity_type: Literal["person"] = "person"


class LocationEntity(BaseEntity):
    """A location or place mentioned in the text."""
    entity_type: Literal["location"] = "location"


class CompanyEntity(BaseEntity):
    """A company or organization mentioned in the text."""
    entity_type: Literal["company"] = "company"


class TaskEntity(BaseEntity):
    """The abstract concept of a task or the need to do something.
    It's naming should be brief with a maximum of 3 words
    and should capture the core meaning of the task.
    We will capture other information like who is evolved in a given task
    later via edges."""
    entity_type: Literal["task"] = "task"


class MeetingEntity(BaseEntity):
    """The abstract concept of a meeting or a get-together.
    It's naming should be brief with a maximum of 3 words
    and should capture the core meaning of the task.
    We will capture other information like who is evolved in a given task
    later via edges."""
    entity_type: Literal["meeting"] = "meeting"


class DayEntity(BaseEntity):
    """A day or date mentioned in the text. Any relative day should be resolved
    relative to the in the prompt given REFERENCE TIME. A day entity should always have a
    name in the format YYYY-MM-DD.
    """
    entity_type: Literal["day"] = "day"


ENTITY_EXTRACTION_AGENT_INSTRUCTIONS = f"""
<ENTITY TYPES>
Person: 
  {PersonEntity.__doc__}
Location:
  {LocationEntity.__doc__}
Company:
  {CompanyEntity.__doc__}
Task:
  {TaskEntity.__doc__}
Meeting:
  {MeetingEntity.__doc__}
Day:
  {DayEntity.__doc__}
</ENTITY TYPES>

Extract all entities mentioned in the given TEXT based on the provided ENTITY TYPES.
For each entity extracted, determine its entity type by name. Then, use the add_entities_to_graph
tool to add the extracted entities to the graph.

Instructions:
1. Extract all distinct entities mentioned in the text
2. Only extract entities that match the provided entity types
"""


Entity = Annotated[Union[PersonEntity, LocationEntity, CompanyEntity, TaskEntity, MeetingEntity, DayEntity], Field(discriminator='entity_type')]


class ExtractedEntities(BaseModel):
    entities: list[Entity]


def create_entity_extraction_agent() -> Agent[AgentDeps, ExtractedEntities]:
    """Factory function to create entity extraction agent with tools."""

    agent = Agent(
        model='gpt-4.1',
        output_type=ExtractedEntities,
        instructions=ENTITY_EXTRACTION_AGENT_INSTRUCTIONS,
        deps_type=AgentDeps,
    )

    @agent.tool
    async def add_entities_to_graph(
        ctx: RunContext[AgentDeps],
        entities: ExtractedEntities,
    ) -> None:
        """Generate embeddings and add entities to Neo4j graph."""
        driver = ctx.deps.neo4j_driver
        openai_client = ctx.deps.openai_client
        for entity in entities.entities:
            name_embedding = await generate_embedding(entity.name, openai_client)
            await driver.execute_query(
                f"MERGE (e:{entity.entity_type} {{uuid: $uuid, name: $name, name_embedding: $name_embedding}})",
                uuid=entity.uuid,
                name=entity.name,
                name_embedding=name_embedding
            )
    return agent


async def extract_entities(
    input_text: str, 
    agent: Agent[AgentDeps, ExtractedEntities], 
    deps: AgentDeps
) -> ExtractedEntities:
    """Extract entities from input text using the provided agent."""
    formatted_input_text = (
        f"<REFERENCE TIME>\n{datetime.now()}\n</REFERENCE TIME>\n\n"
        f"<TEXT>\n{input_text}\n</TEXT>"
    )
    r = await agent.run(formatted_input_text, deps=deps)
    return r.output

