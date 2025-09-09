import logging
import uvicorn
from datetime import datetime

import openai
from fastapi import FastAPI
from neo4j import AsyncGraphDatabase
from dotenv import load_dotenv

from pkms.common import AgentDeps
from pkms.entities import create_entity_extraction_agent, extract_entities
from pkms.edges import create_edge_extraction_agent, extract_edges
from pkms.search import create_search_agent, search_graph
from pkms.episodes import create_episode_with_entities

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI(title="PKMS Server", description="Personal Knowledge Management System")

driver = AsyncGraphDatabase.driver(
    uri='bolt://localhost:7687',
    auth=('neo4j', 'password')
)
openai_client = openai.AsyncOpenAI()
deps = AgentDeps(driver, openai_client)
entity_agent = create_entity_extraction_agent()
edge_agent = create_edge_extraction_agent()
search_agent = create_search_agent()


@app.get("/query_graph", response_model=str)
async def query_grap(query: str) -> str:
    return await search_graph(query, search_agent, deps)


@app.post("/modify_graph", response_model=str)
async def modify_graph(query: str) -> str:
    reference_time = datetime.now()
    entities_result = await extract_entities(
        query, 
        entity_agent, 
        deps, 
        reference_time,
    )
    edges_result = await extract_edges(
        query,
        entities_result.entities,
        edge_agent,
        deps,
        reference_time,
    )
    episode_name = await create_episode_with_entities(
        query,
        entities_result.entities,
        deps,
        reference_time,
    )
    entity_names = [entity.name for entity in entities_result.entities]
    edge_descriptions = [f"{edge.subject} -> {edge.object}" for edge in edges_result.edges]
    return f'Episode: {episode_name}\nEntities: {", ".join(entity_names)}\nEdges: {", ".join(edge_descriptions)}'


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

