from typing import Union, Literal, Annotated
from uuid import uuid4

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from .common import AgentDeps, generate_embedding


ENTITY_EXTRACTION_AGENT_INSTRUCTIONS = """
<ENTITY TYPES>
Person: 
  A person or individual mentioned in the text.

  If there are first person personal pronouns such as "I", "me" or "myself", then ALWAYS extract a person with the name "Daniel".

Location:
  A place mentioned in the text.

Task:
    The abstract concept of a task or the need to do something. It's naming should be brief with a maximum of 3 words
    and should capture the core meaning of the task. We will capture other information like who is evolved in a given task
    later via edges.
</ENTITY TYPES>

Extract all entities mentioned in the below TEXT based on the provided ENTITY TYPES.
For each entity extracted, determine its entity type by name. Then, use the add_entities_to_graph
tool to add the extracted entities to the graph.

Instructions:
1. Extract all distinct entities mentioned in the text
2. Only extract entities that match the provided entity types

<TEXT>
"""


class BaseEntity(BaseModel):
    uuid: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    entity_type: str


class PersonEntity(BaseEntity):
    entity_type: Literal["Person"] = "Person"


class LocationEntity(BaseEntity):
    entity_type: Literal["Location"] = "Location"


class TaskEntity(BaseEntity):
    entity_type: Literal["Task"] = "Task"


Entity = Annotated[Union[PersonEntity, LocationEntity, TaskEntity], Field(discriminator='entity_type')]


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
    r = await agent.run(input_text, deps=deps)
    return r.output

