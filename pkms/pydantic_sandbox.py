import asyncio
from dataclasses import dataclass
from typing import Any, Union, Literal, Annotated
from uuid import uuid4

import logfire
from dotenv import load_dotenv
from neo4j import AsyncGraphDatabase, AsyncDriver
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import EqualsExpected

load_dotenv()


INPUT_TEXTS = [
    "I need to text TaoTao back about planning our lunch in Vienna.",
    # "I need to text Leila about planning our lunch in Vienna."
]

QUERIES = [
    "With whom do I have a lunch planned?",
]


logfire.configure(
    send_to_logfire='if-token-present',  
    environment='development',  
    service_name='evals',
)
logfire.instrument_pydantic_ai()


import openai

@dataclass
class AgentDeps:
    neo4j_driver: AsyncDriver
    openai_client: openai.AsyncOpenAI


async def generate_embedding(input: str, openai_client: openai.AsyncOpenAI) -> list[float]:
    """Generate embedding for entity name using OpenAI API."""
    response = await openai_client.embeddings.create(
        input=input,
        model='text-embedding-3-small',
    )
    return response.data[0].embedding


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


class Person(BaseEntity):
    entity_type: Literal["Person"] = "Person"


class Location(BaseEntity):
    entity_type: Literal["Location"] = "Location"


class Task(BaseEntity):
    entity_type: Literal["Task"] = "Task"


Entity = Annotated[Union[Person, Location, Task], Field(discriminator='entity_type')]


class ExtractedEntities(BaseModel):
    entities: list[Entity]


entity_extraction_agent = Agent(
    model='gpt-4.1',
    output_type=ExtractedEntities,
    instructions=ENTITY_EXTRACTION_AGENT_INSTRUCTIONS,
    deps_type=AgentDeps,
)


@entity_extraction_agent.tool
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


async def extract_entities(input_text: str, deps: AgentDeps) -> ExtractedEntities:
    r = await entity_extraction_agent.run(input_text, deps=deps)
    return r.output


# ENTITY_DEDUPLICATION_AGENT_INSTRUCTIONS = """
# Given the below EXISTING ENTITIES and their attributes, MESSAGE, and PREVIOUS MESSAGES; Determine if the NEW ENTITY extracted from the conversation
# is a duplicate entity of one of the EXISTING ENTITIES.
#
# Entities should only be considered duplicates if they refer to the *same real-world object or concept*.
#
# Do NOT mark entities as duplicates if:
# - They are related but distinct.
# - They have similar names or purposes but refer to separate instances or concepts.
#
#  TASK:
#  1. Compare `new_entity` against each item in `existing_entities`.
#  2. If it refers to the same real‐world object or concept, collect its index.
#  3. Let `duplicate_idx` = the *first* collected index, or –1 if none.
#  4. Let `duplicates` = the list of *all* collected indices (empty list if none).
#
# Also return the full name of the NEW ENTITY (whether it is the name of the NEW ENTITY, a node it
# is a duplicate of, or a combination of the two).
# """
#
#
# entity_deduplication_agent = Agent(
#     model='gpt-4.1',
#     output_type=Union[Person, Location, Task],
#     instructions=ENTITY_DEDUPLICATION_AGENT_INSTRUCTIONS,
#     deps_type=AgentDeps,
# )
#
#
# async def dedupe_entities(
#     extracted_entities: ExtractedEntities,
#     existing_entities: list[Entity],
#     deps: AgentDeps
# ) -> ExtractedEntities:
#     """Deduplicate extracted entities against existing entities."""
#     pass
#

EDGE_EXTRACTION_AGENT_INSTRUCTIONS = """
<FACT TYPES>
RELATES_TO:
    Person --> Person
    Person --> Location
    Task --> Person
    Task --> Location
</FACT TYPES>

Extract all factual relationships between the given ENTITIES based on the CURRENT MESSAGE.
Then, use the add_edges_to_graph tool to add the extracted edges to the graph.
Only extract facts that:
- involve two DISTINCT ENTITIES from the ENTITIES list,
- are clearly stated or unambiguously implied in the CURRENT MESSAGE,
    and can be represented as edges in a knowledge graph.
- Facts should include entity names rather than pronouns whenever possible.
- The FACT TYPES each contain their own mapping which represents the subject and object entity types
  the edge is allowed to connect.

# EXTRACTION RULES

1. Only emit facts where both the subject and object match entities in ENTITIES.
2. Each fact must involve two **distinct** entities.
3. Do not emit duplicate or semantically redundant facts.
4. The `fact` should quote or closely paraphrase the original source sentence(s).
5. Only create edges that obey the rules of FACT TYPES.
6. For edges that involve tasks, make sure to create an edge for each person or Location involved.
"""


class Edge(BaseModel):
    edge_type: str
    subject: str
    object: str 
    fact: str

class ExtractedEdges(BaseModel):
    edges: list[Edge]


edge_extraction_agent = Agent(
    model='gpt-4.1',
    output_type=ExtractedEdges,
    instructions=EDGE_EXTRACTION_AGENT_INSTRUCTIONS,
    deps_type=AgentDeps,
)


@edge_extraction_agent.tool
async def add_edges_to_graph(
    ctx: RunContext[AgentDeps],
    edges: ExtractedEdges,
) -> None:
    """Add extracted edges to Neo4j graph."""
    driver = ctx.deps.neo4j_driver
    openai_client = ctx.deps.openai_client
    for edge in edges.edges:
        # TODO: Use entity uuid instead of name for source/target
        fact_embedding = await generate_embedding(edge.fact, openai_client)
        await driver.execute_query(
            """
            MATCH (source {name: $source_name})
            MATCH (target {name: $target_name})
            MERGE (source)-[r:RELATES_TO {fact: $fact, fact_embedding: $fact_embedding}]->(target)
            """,
            source_name=edge.subject,
            target_name=edge.object,
            fact=edge.fact,
            fact_embedding=fact_embedding,
        )


async def extract_edges(input_text: str, entities: list[str], deps: AgentDeps) -> ExtractedEdges:
    prompt = f"""
    <ENTITIES>
    {', '.join(entities)}
    </ENTITIES>

    <CURRENT MESSAGE>
    {input_text}
    </CURRENT MESSAGE>
    """
    r = await edge_extraction_agent.run(prompt, deps=deps)
    return r.output


SEARCH_AGENT_INSTRUCTIONS = """
You are a helpful assistant that answers questions about entities and their relationships in a knowledge graph.

Use the semantic_search tool to find relevant entities and relationships based on the user's query, then provide a clear and helpful answer based on the search results.
"""


class EntityResult(BaseModel):
    entity_name: str
    entity_labels: list[str]
    similarity_score: float


class EdgeResult(BaseModel):
    source_name: str
    target_name: str
    fact: str
    similarity_score: float


class SearchResults(BaseModel):
    entities: list[EntityResult]
    edges: list[EdgeResult]


search_agent = Agent(
    model='gpt-4.1',
    instructions=SEARCH_AGENT_INSTRUCTIONS,
    deps_type=AgentDeps,
)


@search_agent.tool
async def semantic_search(
    ctx: RunContext[AgentDeps],
    query: str,
    limit: int = 10,
    similarity_threshold: float = 0.7
) -> SearchResults:
    """Search for entities and edges using semantic similarity."""
    driver = ctx.deps.neo4j_driver
    openai_client = ctx.deps.openai_client
    query_embedding = await generate_embedding(query, openai_client)
    entity_records, _, _ = await driver.execute_query(
        """
        MATCH (n)
        WHERE n.name_embedding IS NOT NULL
        WITH n, vector.similarity.cosine(n.name_embedding, $query_embedding) AS similarity
        WHERE similarity >= $threshold
        RETURN n.name AS entity_name, labels(n) AS entity_labels, similarity
        ORDER BY similarity DESC
        LIMIT $limit
        """,
        query_embedding=query_embedding,
        threshold=similarity_threshold,
        limit=limit
    )
    edge_records, _, _ = await driver.execute_query(
        """
        MATCH (source)-[r:RELATES_TO]->(target)
        WHERE r.fact_embedding IS NOT NULL
        WITH source, target, r, vector.similarity.cosine(r.fact_embedding, $query_embedding) AS similarity
        WHERE similarity >= $threshold
        RETURN source.name AS source_name, target.name AS target_name, r.fact AS fact, similarity
        ORDER BY similarity DESC
        LIMIT $limit
        """,
        query_embedding=query_embedding,
        threshold=similarity_threshold,
        limit=limit
    )
    entities = [
        EntityResult(
            entity_name=record['entity_name'],
            entity_labels=record['entity_labels'],
            similarity_score=record['similarity']
        )
        for record in entity_records
    ]
    
    edges = [
        EdgeResult(
            source_name=record['source_name'],
            target_name=record['target_name'],
            fact=record['fact'],
            similarity_score=record['similarity']
        )
        for record in edge_records
    ]
    return SearchResults(entities=entities, edges=edges)


async def search_graph(query: str, deps: AgentDeps) -> str:
    """Search the graph and get an answer to the query."""
    response = await search_agent.run(query, deps=deps)
    return response.output


async def clear_graph(driver: AsyncDriver):
    async with driver.session() as session:
        await session.execute_write(
            lambda tx: tx.run("MATCH (n) DETACH DELETE n")
        )


async def main():
    openai_client = openai.AsyncOpenAI()
    async with AsyncGraphDatabase.driver(
        uri='bolt://localhost:7687',
        auth=('neo4j', 'password')
    ) as driver:
        await clear_graph(driver)
        deps = AgentDeps(driver, openai_client)

        for input_text in INPUT_TEXTS:
            print("Input:", input_text)
            entities_result = await extract_entities(input_text, deps=deps)
            print("Entities:", entities_result)

            # TODO: per entity, run dedupe agent
            #       append returned entity to resolved entity

            # TODO: pass in resolved entities
            all_entities = [entity.name for entity in entities_result.entities]
            edges_result = await extract_edges(input_text, all_entities, deps=deps)
            print("Edges:", edges_result)

        print("=" * 50)
        print("SEARCHING THE GRAPH")
        print("=" * 50)
        for query in QUERIES:
            print(f"Query: {query}")
            answer = await search_graph(query, deps=deps)
            print(f"Answer: {answer}")


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
