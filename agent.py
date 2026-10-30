from deepagents import create_deep_agent
from deepagents.middleware.subagents import SubAgent
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from prompt import (
    CHUNK_ANALYST_INSTRUCTIONS,
    RAG_WORKFLOW_INSTRUCTIONS,
    SUBAGENT_DELEGATION_INSTRUCTIONS,
)
from tools import backend, search_documents

max_concurrent_analysts = 3

INSTRUCTIONS = (
    RAG_WORKFLOW_INSTRUCTIONS
    + "\n\n"
    + "=" * 80
    + "\n\n"
    + SUBAGENT_DELEGATION_INSTRUCTIONS.format(
        max_concurrent_analysts=max_concurrent_analysts
    )
)

chunk_analyst_subagent: SubAgent = {
    "name": "chunk_analyst",
    "description": (
        "analyse one retrieved documentation chunk file"
        "pass the user question and a single file path under /retrieved"
    ),
    "system_prompt": CHUNK_ANALYST_INSTRUCTIONS,
}

model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")

agent = create_deep_agent(
    model=model,
    tools=[search_documents],
    backend=backend,
    system_prompt=INSTRUCTIONS,
    subagents=[chunk_analyst_subagent],
)

example_query = "What are the cause of climate changes? "

result = agent.invoke({"messages": [HumanMessage(content=example_query)]})

for block in result['messages'][-1].content:
    if block["type"] == "text":
        print(block["text"])