import os
import asyncio
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
from neo4j import AsyncGraphDatabase

load_dotenv()

INPUTS = [
    "I need to write Felix about the money I owe him today.",
    "I have a meeting with Leila and Marion at 9am next Thursday in Cafe Ritter.",
    "Sarah called me about the project deadline."
]

class Person(BaseModel):
    """
    A person or individual mentioned in the text.

    If there are first person personal pronouns such as "I", "me" or "myself", then ALWAYS extract a person with the name "Daniel".
    """
    name: str = Field(..., description='Name of the person')

class Todo(BaseModel):
    """
    The abstract concept of a task or the need to do something.

    Examples patterns:
    "I need to VERB Felix about OBJECT today" --> "VERB about OBJECT"
    """
    name: str = Field(..., description='What the todo is about. Keep it a short description.')

class Meeting(BaseModel):
    """The abstract concept of getting together to do something."""
    name: str = Field(..., description='What the meeting is about. Keep it a short description. Leave out person names, they should be separate entities.')

class ExtractedEntity(BaseModel):
    name: str = Field(..., description='Name of the extracted entity')
    entity_type: str = Field(..., description='Type of the entity')

class ExtractedEntities(BaseModel):
    extracted_entities: list[ExtractedEntity] = Field(..., description='List of extracted entities')

def format_entity_types(entity_models: list[type[BaseModel]]) -> str:
    entity_types = []
    for i, model in enumerate(entity_models):
        # Get docstring for additional context
        docstring = model.__doc__ or "No description available"
        
        fields_desc = []
        for field_name, field_info in model.model_fields.items():
            desc = field_info.description or "No description"
            fields_desc.append(f"  - {field_name}: {desc}")
        
        entity_types.append(f"{i}: {model.__name__}\nDescription: {docstring}\nFields:\n" + "\n".join(fields_desc))
    return "\n\n".join(entity_types)

EXTRACTION_PROMPT = """
<ENTITY TYPES>
{entity_types}
</ENTITY TYPES>

<TEXT>
{text}
</TEXT>

Extract all entities mentioned in the TEXT based on the provided ENTITY TYPES.
For each entity extracted, determine its entity type and name from the provided types.

Instructions:
1. Extract all distinct entities mentioned in the text
2. Only extract entities that match the provided entity types
"""

async def extract_entities(text: str, entity_models: list[type[BaseModel]]) -> ExtractedEntities:
    client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    entity_types_str = format_entity_types(entity_models)
    content = EXTRACTION_PROMPT.format(text=text, entity_types=entity_types_str)
    print(content)
    
    response = await client.beta.chat.completions.parse(
        model="gpt-4.1",
        messages=[
            {"role": "user", "content": content}
        ],
        response_format=ExtractedEntities,
        temperature=0
    )
    
    return response.choices[0].message.parsed

async def write_nodes_to_neo4j(entities: list[ExtractedEntity]):
    driver = AsyncGraphDatabase.driver(
        "neo4j://127.0.0.1:7687",
        auth=("neo4j", "password")
    )
    
    async with driver.session() as session:
        # Clear all existing data
        await session.execute_write(
            lambda tx: tx.run("MATCH (n) DETACH DELETE n")
        )
        
        for entity in entities:
            await session.execute_write(
                lambda tx: tx.run(
                    f"MERGE (e:{entity.entity_type} {{name: $name}})",
                    name=entity.name
                )
            )
    
    await driver.close()

async def main():
    entity_models = [Person, Meeting, Todo]
    all_extracted_entities = {}
    for i, input_text in enumerate(INPUTS):
        print(f"Processing input {i+1}: {input_text}")
        extracted_entities = await extract_entities(input_text, entity_models)
        all_extracted_entities[input_text] = extracted_entities.extracted_entities
        await write_nodes_to_neo4j(extracted_entities.extracted_entities)

    print("\n=== All Extracted Entities ===")
    for input_text, entities in all_extracted_entities.items():
        print(f"Input: {input_text}")
        for entity in entities:
            print(f"  - {entity.name} ({entity.entity_type})")
        print()

if __name__ == "__main__":
    asyncio.run(main())

