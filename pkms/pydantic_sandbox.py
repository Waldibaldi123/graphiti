import asyncio
from dataclasses import dataclass
from typing import Any

import logfire
from dotenv import load_dotenv
from neo4j import AsyncGraphDatabase, AsyncDriver
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import EqualsExpected

load_dotenv()


logfire.configure(
    send_to_logfire='if-token-present',  
    environment='development',  
    service_name='evals',
)
logfire.instrument_pydantic_ai()


ENTITY_EXTRACTION_AGENT_INSTRUCTIONS = """
<ENTITY TYPES>
Person: 
  A person or individual mentioned in the text.

  If there are first person personal pronouns such as "I", "me" or "myself", then ALWAYS extract a person with the name "Daniel".

Location:
  A place mentioned in the text.
</ENTITY TYPES>

Extract all entities mentioned in the below TEXT based on the provided ENTITY TYPES.
For each entity extracted, determine its entity type by name. Then, use the add_entities_to_graph
tool to add the extracted entities to the graph.

Instructions:
1. Extract all distinct entities mentioned in the text
2. Only extract entities that match the provided entity types

<TEXT>
"""


class ExtractedEntities(BaseModel):
    persons: list[str]
    locations: list[str]


@dataclass
class AgentDeps:
    neo4j_driver: AsyncDriver


entity_extraction_agent = Agent(
    model='gpt-4.1',
    output_type=ExtractedEntities,
    instructions=ENTITY_EXTRACTION_AGENT_INSTRUCTIONS,
    deps_type=AgentDeps,
)


async def extract_entities(input_text: str, deps: AgentDeps) -> ExtractedEntities:
    r = await entity_extraction_agent.run(input_text, deps=deps)
    return r.output


@entity_extraction_agent.tool
async def add_entities_to_graph(
    ctx: RunContext[AgentDeps],
    entities: ExtractedEntities,
) -> None:
    driver = ctx.deps.neo4j_driver
    for person in entities.persons:
        await driver.execute_query(
            "MERGE (p:Person {name: $name})",
            name=person
        )
    for location in entities.locations:
        await driver.execute_query(
            "MERGE (l:Location {name: $name})",
            name=location
        )


async def clear_graph(driver: AsyncDriver):
    async with driver.session() as session:
        await session.execute_write(
            lambda tx: tx.run("MATCH (n) DETACH DELETE n")
        )


async def main():
    async with AsyncGraphDatabase.driver(
        uri='bolt://localhost:7687',
        auth=('neo4j', 'password')
    ) as driver:
        await clear_graph(driver)
        deps = AgentDeps(driver)
        result = await extract_entities("I meet with Felix today in Vienna.", deps=deps)
        print(result)


if __name__ == '__main__':
    asyncio.run(main())


# Eval logic
# --------------------------------------------
# extract_entities_dataset = Dataset[str, ExtractedEntities, Any](
#     cases=[
#         Case(
#             name='meeting_with_location',
#             inputs='I meet with Felix today in Vienna.',
#             expected_output=ExtractedEntities(
#                 persons=['Daniel', 'Felix'],
#                 locations=['Vienna'],
#             ),
#             metadata={'difficulty': 'easy'},
#         )
#     ],
#     evaluators=[EqualsExpected()],
# )
#
# report = extract_entities_dataset.evaluate_sync(extract_entities)
# print(report)
