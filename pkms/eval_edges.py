from typing import Any

import logfire
import openai
from dataclasses import dataclass
from dotenv import load_dotenv
from neo4j import AsyncGraphDatabase
from pydantic_evals import Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext, EvaluationReason

from .common import AgentDeps, clear_graph
from .edges import (
    ExtractedEdges,
    create_edge_extraction_agent,
    extract_edges,
)
from .entities import ExtractedEntities, add_entities_to_graph

load_dotenv()
logfire.configure(
    send_to_logfire='if-token-present',
    environment='development',
    service_name='evals',
)
logfire.instrument_pydantic_ai()


@dataclass
class ExtractedEdgesEvaluator(Evaluator[dict, ExtractedEdges]):
    async def evaluate(self, ctx: EvaluatorContext[dict, ExtractedEdges]) -> EvaluationReason:
        score = 1.0
        reasons = []
        output_edge_keys = {(e.subject.lower(), e.object.lower(), e.edge_type) for e in ctx.output.edges}
        expected_edge_keys = {(e.subject.lower(), e.object.lower(), e.edge_type) for e in ctx.expected_output.edges}
        missing_edges = expected_edge_keys - output_edge_keys
        for subject, obj, edge_type in missing_edges:
            score -= 0.15
            reasons.append(f'Missing {edge_type} edge: "{subject}" -> "{obj}"')
        additional_edges = output_edge_keys - expected_edge_keys
        for subject, obj, edge_type in additional_edges:
            score -= 0.1
            reasons.append(f'Unexpected {edge_type} edge: "{subject}" -> "{obj}"')
        # TODO: semantic comparison of facts
        if reasons:
            reason_text = '\n  '.join(reasons)
        else:
            reason_text = ''
        return EvaluationReason(value=max(0.0, score), reason=reason_text)


edge_extraction_agent = create_edge_extraction_agent()


extract_edges_dataset = Dataset[dict, ExtractedEdges, Any].from_file(
    'edge_extraction_tests.yaml',
    custom_evaluator_types=(ExtractedEdgesEvaluator,)
)


async def _eval_extract_edges(inputs: dict) -> ExtractedEdges:
    openai_client = openai.AsyncOpenAI()
    async with AsyncGraphDatabase.driver(
        uri='bolt://localhost:7687',
        auth=('neo4j', 'password')
    ) as driver:
        await clear_graph(driver)
        deps = AgentDeps(driver, openai_client)
        entities = []
        for entity_data in inputs['entities']:
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

        edges_result = await extract_edges(
            input_text=inputs['text'],
            entities=entities,
            agent=edge_extraction_agent,
            deps=deps,
            reference_time=inputs['reference_time'],
        )
        return edges_result


def eval_extract_edges():
    extract_edges_dataset.cases = [extract_edges_dataset.cases[0]]
    report = extract_edges_dataset.evaluate_sync(_eval_extract_edges)
    report.print(include_reasons=True)

