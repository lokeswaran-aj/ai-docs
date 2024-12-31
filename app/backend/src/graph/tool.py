from langchain.tools.retriever import create_retriever_tool
from langchain_openai import OpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector

from .configuration import get_config


def setup_vector_store() -> PGVector:
    """Initialize and return the vector store."""
    settings = get_config()

    return PGVector(
        embeddings=OpenAIEmbeddings(model=settings.embedding_model),
        collection_name="nextjs_docs",
        connection=settings.database_url + "?sslmode=" + settings.database_sslmode,
        use_jsonb=True,
    )


def create_retrieval_tool() -> dict:
    """Create a retrieval tool with the given retriever."""
    settings = get_config()
    vector_store = setup_vector_store()
    retriever = vector_store.as_retriever(
        search_kwargs={"k": settings.top_k_docs},
    )
    return create_retriever_tool(
        retriever,
        "retrieve_documentation",
        "Search and retrieve relevant documentation based on the query. Returns documentation snippets that best match the question.",
    )


retrieval_tool = create_retrieval_tool()
TOOLS = [retrieval_tool]
