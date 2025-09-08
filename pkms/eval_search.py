from typing import Any
import yaml

import logfire
import openai
from dataclasses import dataclass
from dotenv import load_dotenv
from neo4j import AsyncGraphDatabase
from pydantic_evals import Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext, EvaluationReason

from .common import AgentDeps, clear_graph
from .search import create_search_agent, search_graph
from .entities import add_entities_to_graph, ExtractedEntities

load_dotenv()
logfire.configure(
    send_to_logfire='if-token-present',
    environment='development',
    service_name='evals',
)
logfire.instrument_pydantic_ai()


@dataclass
class SearchEvaluator(Evaluator[dict, str]):
    async def evaluate(self, ctx: EvaluatorContext[dict, str]) -> EvaluationReason:
        score = 1.0
        reasons = []
        inputs = ctx.inputs
        output = ctx.output.lower()
        if 'expected_keywords' in inputs:
            missing_keywords = []
            for keyword in inputs['expected_keywords']:
                if keyword.lower() not in output:
                    missing_keywords.append(keyword)
                    score -= 0.3
            if missing_keywords:
                reasons.append(f'Missing keywords: {", ".join(missing_keywords)}')
        if 'expected_entities' in inputs:
            missing_entities = []
            for entity in inputs['expected_entities']:
                if entity.lower() not in output:
                    missing_entities.append(entity)
                    score -= 0.2
            if missing_entities:
                reasons.append(f'Missing entity mentions: {", ".join(missing_entities)}')
        if reasons:
            reason_text = '\n  '.join(reasons)
        else:
            reason_text = ''
        return EvaluationReason(value=max(0.0, score), reason=reason_text)


search_agent = create_search_agent()


search_dataset = Dataset[dict, str, Any].from_file(
    'search_tests.yaml',
    custom_evaluator_types=(SearchEvaluator,)
)


async def setup_search_graph(driver, openai_client):
    """Set up the graph with test data from search_graph_setup.yaml"""
    with open('search_graph_setup.yaml', 'r') as f:
        graph_data = yaml.safe_load(f)
    deps = AgentDeps(driver, openai_client)
    entities_data = graph_data['entities']
    entities = []
    for entity_data in entities_data:
        if entity_data['entity_type'] == 'person':
            from .entities import PersonEntity
            entities.append(PersonEntity(**entity_data))
        elif entity_data['entity_type'] == 'location':
            from .entities import LocationEntity
            entities.append(LocationEntity(**entity_data))
        elif entity_data['entity_type'] == 'task':
            from .entities import TaskEntity
            entities.append(TaskEntity(**entity_data))
        elif entity_data['entity_type'] == 'day':
            from .entities import DayEntity
            entities.append(DayEntity(**entity_data))
        elif entity_data['entity_type'] == 'meeting':
            from .entities import MeetingEntity
            entities.append(MeetingEntity(**entity_data))
        elif entity_data['entity_type'] == 'company':
            from .entities import CompanyEntity
            entities.append(CompanyEntity(**entity_data))
    entities_result = ExtractedEntities(entities=entities)
    await add_entities_to_graph(entities_result, deps)
    edges_data = graph_data['edges']
    edges = []
    for edge_data in edges_data:
        if edge_data['edge_type'] == 'relates_to':
            from .edges import RelatesToEdge
            edges.append(RelatesToEdge(**edge_data))
        elif edge_data['edge_type'] == 'scheduled_for':
            from .edges import ScheduledForEdge
            edges.append(ScheduledForEdge(**edge_data))
        elif edge_data['edge_type'] == 'attends':
            from .edges import AttendsEdge
            edges.append(AttendsEdge(**edge_data))
        elif edge_data['edge_type'] == 'located_at':
            from .edges import LocatedAtEdge
            edges.append(LocatedAtEdge(**edge_data))
    from .common import generate_embedding
    for edge in edges:
        fact_embedding = await generate_embedding(edge.fact, openai_client)
        await driver.execute_query(
            f"""
            MATCH (source {{name: $source_name}})
            MATCH (target {{name: $target_name}})
            MERGE (source)-[r:{edge.edge_type.upper()} {{uuid: $uuid, fact: $fact, fact_embedding: $fact_embedding}}]->(target)
            """,
            source_name=edge.subject,
            target_name=edge.object,
            uuid=edge.uuid,
            fact=edge.fact,
            fact_embedding=fact_embedding,
        )


# Global flag to track if graph has been set up
_graph_setup_done = False


async def _eval_search(inputs: dict) -> str:
    global _graph_setup_done
    openai_client = openai.AsyncOpenAI()
    async with AsyncGraphDatabase.driver(
        uri='bolt://localhost:7687',
        auth=('neo4j', 'password')
    ) as driver:
        # Disabled graph setup to save money and time
        # if not _graph_setup_done:
        #     await clear_graph(driver)
        #     await setup_search_graph(driver, openai_client)
        #     _graph_setup_done = True
        deps = AgentDeps(driver, openai_client)
        result = await search_graph(
            query=inputs['query'],
            agent=search_agent,
            deps=deps
        )
        return result


def eval_search():
    global _graph_setup_done
    _graph_setup_done = False  # Reset for each evaluation run
    search_dataset.cases = [search_dataset.cases[0]]
    report = search_dataset.evaluate_sync(_eval_search)
    report.print(include_reasons=True)

