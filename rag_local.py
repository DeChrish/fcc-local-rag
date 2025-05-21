from rag_chain import create_rag_chain
from rag_embedding import get_embedding_function
from rag_load_doc import load_documents
from rag_split_doc import split_documents
from rag_vector_store import get_vector_store, index_documents  

from langchain_chroma import Chroma

CHROMA_PATH = "chroma_db" # Directory to store ChromaDB data


def query_rag(chain, question):
    """Queries the RAG chain and prints the response."""
    print("\nQuerying RAG chain...")
    print(f"Question: {question}")
    response = chain.invoke(question)
    print("\nResponse:")
    print(response)


# --- Main Execution ---
if __name__ == "__main__":

    # 1. Load Documents
    docs = load_documents()

    # 2. Split Documents
    chunks = split_documents(docs)

    # 3. Get Embedding Function
    embedding_function = get_embedding_function() # Using Ollama nomic-embed-text
    #vector_store = get_vector_store(embedding_function)
    # vector_store = index_documents(chunks, embedding_function)

    vector_store = Chroma(persist_directory="chroma_db", embedding_function=embedding_function)

    # 4. Create RAG Chain
    rag_chain = create_rag_chain(vector_store)

    # 6. Query
    query_question = "What is the main topic of the document?" # Replace with a specific question
    query_rag(rag_chain, query_question)

    query_question_2 = "Summarize the introduction section." # Another example
    query_rag(rag_chain, query_question_2)