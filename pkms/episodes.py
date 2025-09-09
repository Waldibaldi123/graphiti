from datetime import datetime
from typing import List

from pydantic import BaseModel

from .common import AgentDeps
from .entities import Entity


class Episode(BaseModel):
    content: str
    timestamp: datetime
    episode_type: str = "user_input"


async def add_episode_to_graph(
    episode: Episode,
    deps: AgentDeps,
) -> str:
    driver = deps.neo4j_driver
    episode_name = f"episode_{episode.timestamp.strftime('%Y%m%d_%H%M%S')}"
    await driver.execute_query(
        """
        MERGE (e:episode {name: $name})
        SET e.content = $content,
            e.timestamp = $timestamp,
            e.episode_type = $episode_type
        """,
        name=episode_name,
        content=episode.content,
        timestamp=episode.timestamp.isoformat(),
        episode_type=episode.episode_type,
    )
    return episode_name


async def connect_episode_to_entities(
    episode_name: str,
    entities: List[Entity],
    deps: AgentDeps,
) -> None:
    driver = deps.neo4j_driver
    for entity in entities:
        await driver.execute_query(
            """
            MATCH (episode:episode {name: $episode_name})
            MATCH (entity {name: $entity_name})
            MERGE (episode)-[r:MENTIONS]->(entity)
            """,
            episode_name=episode_name,
            entity_name=entity.name
        )


async def create_episode_with_entities(
    content: str,
    entities: List[Entity],
    deps: AgentDeps,
    timestamp: datetime | None = None
) -> str:
    episode_timestamp = timestamp or datetime.now()
    episode = Episode(
        content=content,
        timestamp=episode_timestamp
    )
    episode_name = await add_episode_to_graph(episode, deps)
    await connect_episode_to_entities(episode_name, entities, deps)
    return episode_name

