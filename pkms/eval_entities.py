from typing import Any

import logfire
import openai
from dataclasses import dataclass
from dotenv import load_dotenv
from neo4j import AsyncGraphDatabase
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext, IsInstance

from .common import AgentDeps, clear_graph
from .entities import (
    ExtractedEntities,
    LocationEntity,
    PersonEntity,
    TaskEntity,
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


# TODO: check if score is correctly calculated
@dataclass
class ExtractedEntitiesEvaluator(Evaluator[str, ExtractedEntities]):
    async def evaluate(self, ctx: EvaluatorContext[str, ExtractedEntities]) -> float:  
        if ctx.expected_output is None:
            return 0.0
        score = 1.0
        output = {e.name for e in ctx.output.entities}
        for expected_identity in ctx.expected_output.entities:
            if expected_identity.name not in output:
                score =- 0.1
        score =- abs(len(ctx.output.entities) - len(ctx.expected_output.entities))
        return score


entity_extraction_agent = create_entity_extraction_agent()
extract_entities_dataset = Dataset[str, ExtractedEntities, Any](
    cases=[
        Case(
            name='remind_dinner',
            inputs='Remind me to plan a dinner with Wolfgang Schneider in Vienna.',
            expected_output=ExtractedEntities(
                entities=[
                    PersonEntity(name='Daniel'),
                    PersonEntity(name='Wolfgang Schneider'),
                    LocationEntity(name='Vienna'),
                    TaskEntity(name='Plan Dinner'),
                ],
            ),
        ),
    ],
    evaluators=[IsInstance(type_name='ExtractedEntities'), ExtractedEntitiesEvaluator()],
)


async def eval_extract_entities(input_text: str) -> ExtractedEntities:
    openai_client = openai.AsyncOpenAI()
    async with AsyncGraphDatabase.driver(
        uri='bolt://localhost:7687',
        auth=('neo4j', 'password')
    ) as driver:
        await clear_graph(driver)
        deps = AgentDeps(driver, openai_client)
        r = await extract_entities(
            input_text=input_text,
            agent=entity_extraction_agent,
            deps=deps,
        )
        return r

report = extract_entities_dataset.evaluate_sync(eval_extract_entities)
print(report)
