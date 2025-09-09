from datetime import datetime
from typing import Union, Literal, Annotated

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.settings import ModelSettings

from .common import AgentDeps, generate_embedding
from .entities import Entity


class BaseEdge(BaseModel):
    subject: str
    object: str
    fact: str = ''
    edge_type: str


class RelatesToEdge(BaseEdge):
    """A general relationship between two entities.
    Used for person-to-person, person-to-location, task-to-person,
    task-to-meeting, meeting-to-task, and task-to-location relationships.
    Do not create a person-to-person relationship if the meaning is already
    captured in another edge."""
    edge_type: Literal["relates_to"] = "relates_to"


class ScheduledForEdge(BaseEdge):
    """A relationship indicating when something is scheduled for a specific day.
    Used for task-to-day and meeting-to-day relationships."""
    edge_type: Literal["scheduled_for"] = "scheduled_for"


class AttendsEdge(BaseEdge):
    """A relationship indicating a person attends a meeting.
    Used for person-to-meeting relationships."""
    edge_type: Literal["attends"] = "attends"


class LocatedAtEdge(BaseEdge):
    """A relationship indicating where something takes place.
    Used for meeting-to-location relationships."""
    edge_type: Literal["located_at"] = "located_at"


EDGE_EXTRACTION_AGENT_INSTRUCTIONS = f"""
<EDGE TYPES>
RelatesTo:
  {RelatesToEdge.__doc__}
ScheduledFor:
  {ScheduledForEdge.__doc__}
Attends:
  {AttendsEdge.__doc__}
LocatedAt:
  {LocatedAtEdge.__doc__}
</EDGE TYPES>

Extract all factual relationships between the given ENTITIES based on the CURRENT MESSAGE.
For each relationship extracted, determine its edge type by the relationship semantics. Then, use the add_edges_to_graph
tool to add the extracted edges to the graph.

Instructions:
1. Extract all distinct relationships mentioned in the text
2. Only extract relationships that match the provided edge types and their allowed connections
3. Facts should include entity names rather than pronouns whenever possible
4. First person personal pronouns such as "I", "me" or "myself" refer to an entity named "Daniel Walder"
5. Only emit facts where both the subject and object match entities in ENTITIES
6. Each fact must involve two **distinct** entities
7. Do not emit duplicate or semantically redundant facts
8. The `fact` should quote or closely paraphrase the original source sentence(s)
"""


Edge = Annotated[Union[RelatesToEdge, ScheduledForEdge, AttendsEdge, LocatedAtEdge], Field(discriminator='edge_type')]


class ExtractedEdges(BaseModel):
    edges: list[Edge]


def create_edge_extraction_agent() -> Agent[AgentDeps, ExtractedEdges]:
    """Factory function to create edge extraction agent with tools."""
    agent = Agent(
        model='gpt-4.1',
        output_type=ExtractedEdges,
        instructions=EDGE_EXTRACTION_AGENT_INSTRUCTIONS,
        deps_type=AgentDeps,
        model_settings=ModelSettings(temperature=0.0),
    )

    @agent.tool
    async def add_edges_to_graph(
        ctx: RunContext[AgentDeps],
        edges: ExtractedEdges,
    ) -> None:
        """Generate embeddings and add edges to Neo4j graph."""
        driver = ctx.deps.neo4j_driver
        openai_client = ctx.deps.openai_client
        for edge in edges.edges:
            fact_embedding = await generate_embedding(edge.fact, openai_client)
            await driver.execute_query(
                f"""
                MATCH (source {{name: $source_name}})
                MATCH (target {{name: $target_name}})
                MERGE (source)-[r:{edge.edge_type.upper()} {{fact: $fact}}]->(target)
                SET r.fact_embedding = $fact_embedding
                """,
                source_name=edge.subject,
                target_name=edge.object,
                fact=edge.fact,
                fact_embedding=fact_embedding,
            )
    return agent


async def extract_edges(
    input_text: str, 
    entities: list[Entity], 
    agent: Agent[AgentDeps, ExtractedEdges], 
    deps: AgentDeps,
    reference_time: datetime | None = None
) -> ExtractedEdges:
    """Extract edges from input text using the provided agent."""
    ref_time = reference_time or datetime.now()
    formatted_input_text = (
        f"<REFERENCE TIME>\n{ref_time}\n</REFERENCE TIME>\n\n"
        f"<ENTITIES>\n{', '.join(str(e.model_dump()) for e in entities)}\n</ENTITIES>\n\n"
        f"<CURRENT MESSAGE>\n{input_text}\n</CURRENT MESSAGE>"
    )
    r = await agent.run(formatted_input_text, deps=deps)
    return r.output

