from typing import Any

import logfire
import openai
from dataclasses import dataclass
from dotenv import load_dotenv
from neo4j import AsyncGraphDatabase
from pydantic_evals import Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext, EvaluationReason

from .common import AgentDeps, clear_graph
from .entities import (
    ExtractedEntities,
    create_entity_extraction_agent,
    extract_entities,
)

load_dotenv()
logfire.configure(
    send_to_logfire='if-token-present',
    environment='development',
    service_name='evals',
)
logfire.instrument_pydantic_ai()


@dataclass
class ExtractedEntitiesEvaluator(Evaluator[dict, ExtractedEntities]):
    async def evaluate(self, ctx: EvaluatorContext[dict, ExtractedEntities]) -> EvaluationReason:  
        score = 1.0
        reasons = []
        output_entities = {e.name.lower(): e.entity_type for e in ctx.output.entities}
        expected_entities = {e.name.lower(): e.entity_type for e in ctx.expected_output.entities}
        missing_entity_names = set(expected_entities.keys()) - set(output_entities.keys())
        for missing_name in missing_entity_names:
            score -= 0.1
            reasons.append(f'Missing {expected_entities[missing_name]} "{missing_name}"')
        additional_entity_names = set(output_entities.keys()) - set(expected_entities.keys())
        for additional_name in additional_entity_names:
            score -= 0.1
            reasons.append(f'Unexpected {output_entities[additional_name]} "{additional_name}"')
        common_names = set(output_entities.keys()) & set(expected_entities.keys())
        for name in common_names:
            if output_entities[name] != expected_entities[name]:
                score -= 0.1
                reasons.append(f'Entity "{name}" has wrong type: expected {expected_entities[name]}, got {output_entities[name]}')
        if reasons:
            reason_text = '\n  '.join(reasons)
        else:
            reason_text = ''
        return EvaluationReason(value=max(0.0, score), reason=reason_text)


entity_extraction_agent = create_entity_extraction_agent()


extract_entities_dataset = Dataset[dict, ExtractedEntities, Any].from_file(
    'entity_extraction_tests.yaml',
    custom_evaluator_types=(ExtractedEntitiesEvaluator,)
)


async def _eval_extract_entities(inputs: dict) -> ExtractedEntities:
    openai_client = openai.AsyncOpenAI()
    async with AsyncGraphDatabase.driver(
        uri='bolt://localhost:7687',
        auth=('neo4j', 'password')
    ) as driver:
        await clear_graph(driver)
        deps = AgentDeps(driver, openai_client)
        r = await extract_entities(
            input_text=inputs['text'],
            agent=entity_extraction_agent,
            deps=deps,
            reference_time=inputs['reference_time'],
        )
        return r


def eval_extract_entities():
    extract_entities_dataset.cases = [extract_entities_dataset.cases[0]]
    report = extract_entities_dataset.evaluate_sync(_eval_extract_entities)
    report.print(include_reasons=True)

