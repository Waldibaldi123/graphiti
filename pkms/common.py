import openai
from dataclasses import dataclass

from neo4j import AsyncDriver


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


async def clear_graph(driver: AsyncDriver):
    async with driver.session() as session:
        await session.execute_write(
            lambda tx: tx.run("MATCH (n) DETACH DELETE n")
        )

