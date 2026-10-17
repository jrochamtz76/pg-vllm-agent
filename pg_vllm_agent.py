import os
import asyncpg
from agentfield import Agent, AIConfig
from pydantic import BaseModel

# --- vLLM Configuration (OpenAI-compatible endpoint) ---
ai_config = AIConfig(
    model="tiny-pickle-v3-coder",   # Model ID served by vLLM
    api_base="http://mylab.ai:1234/v1",    # vLLM's /v1 endpoint (adjust for networking)
    api_key="dummy-key",
)

app = Agent(
    node_id="pg-vllm-agent",
    ai_config=ai_config,
    # The control plane URL is injected automatically when run in-container;
    # this default is harmless for local reference.
    agentfield_server=os.getenv("AGENTFIELD_SERVER", "http://mylab.ai:8010"),
)

# Read the DSN from an environment variable. The control plane's secret store
# injects this at runtime—never hardcode credentials.
PG_DSN = os.environ["PG_DSN"]

async def fetch_from_postgres(query: str) -> list[dict]:
    conn = await asyncpg.connect(PG_DSN)
    try:
        records = await conn.fetch(query)
        return [dict(r) for r in records]
    finally:
        await conn.close()

class Answer(BaseModel):
    summary: str
    reasoning: str

@app.reasoner()
async def query_with_vllm(question: str) -> dict:
    rows = await fetch_from_postgres("SELECT * FROM articles LIMIT 10;")

    prompt = f"""
    Based on these PostgreSQL results, answer the user's question.

    Question: {question}
    Database Rows: {rows}

    Provide a summary and reasoning.
    """
    response = await app.ai(
        system="You are a data analyst assistant.",
        user=prompt,
        schema=Answer,
    )
    return response.model_dump()