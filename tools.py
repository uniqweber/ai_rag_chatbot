import uuid

from deepagents.backends import StateBackend
from langchain.tools import tool

from indexing import vector_store

backend = StateBackend()


def search_documents(query: str):
    """
    Search langchain documentation and save matching chunks to the agent filesystem

    Args:
        query: Natural language search query

    Returns:
        File paths where retrieved chunks were saved under /retrieved
    """

    retrieved_docs = vector_store.similarity_search(query=query, k=4)

    batch_id = uuid.uuid4().hex[:8]
    uploads: list[tuple[str, bytes]] = []
    saved_paths: list[str] = []

    for index, doc in enumerate(retrieved_docs, start=1):
        path = f"/retrieved/{batch_id}/chunk_{index}.md"
        content = f"# Source: {doc.metadata.get('source', 'unknown')} \n"


search_documents("hello")
