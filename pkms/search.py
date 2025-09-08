from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.settings import ModelSettings

from .common import AgentDeps, generate_embedding


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


SEARCH_AGENT_INSTRUCTIONS = """
You are a helpful assistant that answers questions about entities and their relationships in a knowledge graph.

Use the semantic_search tool to find relevant entities and relationships based on the user's query, then provide a clear and helpful answer based on the search results. Use the default limit and threshold parameters.
First person personal pronouns such as "I", "me" or "myself" refer to a person named "Daniel Walder".
"""


def create_search_agent() -> Agent[AgentDeps, str]:
    """Factory function to create search agent with tools."""

    agent = Agent(
        model='gpt-4.1',
        instructions=SEARCH_AGENT_INSTRUCTIONS,
        deps_type=AgentDeps,
        model_settings=ModelSettings(temperature=0.0),
    )

    @agent.tool
    async def semantic_search(
        ctx: RunContext[AgentDeps],
        query: str,
        limit: int = 10,
        similarity_threshold: float = 0.5
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
            MATCH (source)-[r]->(target)
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

    return agent


async def search_graph(
    query: str, 
    agent: Agent[AgentDeps, str], 
    deps: AgentDeps
) -> str:
    """Search the graph and get an answer to the query."""
    response = await agent.run(query, deps=deps)
    return response.output

