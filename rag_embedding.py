from langchain_ollama import OllamaEmbeddings



def get_embedding_function(model_name="nomic-embed-text"):
    embeddings = OllamaEmbeddings(model=model_name)
    print(f"Initialized Ollama embeddings with model: {model_name}")
    return embeddings