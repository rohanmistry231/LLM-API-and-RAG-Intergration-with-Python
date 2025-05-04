# 🔍 Embeddings and Vector Stores

## 📖 Introduction

**Embeddings and Vector Stores** leverages OpenAI’s Embeddings API and vector databases (e.g., FAISS) to enable semantic search and data storage, critical for RAG systems. This guide provides Python examples, focusing on the AI-driven era (May 3, 2025).

## 🌟 Key Concepts

- **Embeddings API**: Generating text vectors.
- **Vector Stores**: Storing and querying embeddings (e.g., FAISS).
- **Semantic Search**: Retrieving relevant texts by meaning.
- **RAG Preparation**: Building retrieval components.

## 🛠️ Practical Example

```python
%% embeddings_vector_stores.py
# Setup: pip install openai faiss-cpu requests matplotlib pandas nltk numpy
import os
import openai
import faiss
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

def run_embeddings_vector_stores_demo():
    # Synthetic Document Data
    documents = [
        "Machine learning is a subset of AI focusing on data-driven models.",
        "Deep learning uses neural networks for complex pattern recognition.",
        "Natural language processing enables computers to understand text.",
        "AI is transforming industries with automation and insights."
    ]
    query = "What is machine learning?"
    print("Synthetic Data: Documents and query created")
    print(f"Documents: {documents}")
    print(f"Query: {query}")

    # OpenAI API Configuration
    openai.api_key = os.getenv("OPENAI_API_KEY", "your-openai-api-key")

    # Generate Embeddings
    embeddings = []
    for text in documents + [query]:
        try:
            response = openai.Embedding.create(
                model="text-embedding-ada-002",
                input=text
            )
            embeddings.append(response['data'][0]['embedding'])
        except openai.error.OpenAIError as e:
            print(f"Error for {text}: {e}")
            return

    # Store in FAISS
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings[:-1]).astype('float32'))

    # Search with Query
    query_embedding = np.array([embeddings[-1]]).astype('float32')
    distances, indices = index.search(query_embedding, len(documents))
    similarities = 1 - distances[0] / 2  # Approximate cosine similarity
    print("Similarities:", similarities)

    # Visualization
    plt.figure(figsize=(8, 4))
    plt.bar(range(1, len(documents) + 1), similarities, color='purple')
    plt.title("Semantic Search Similarity Scores")
    plt.xlabel("Document")
    plt.ylabel("Similarity")
    plt.savefig("embeddings_vector_stores_output.png")
    print("Visualization: Similarity scores saved as embeddings_vector_stores_output.png")

# Execute the demo
if __name__ == "__main__":
    run_embeddings_vector_stores_demo()
```

## 📊 Visualization Output

The code generates a bar chart (`embeddings_vector_stores_output.png`) showing similarity scores between the query and documents, illustrating retrieval relevance.

## 💡 Applications

- **RAG Systems**: Retrieve context for LLM responses.
- **Search Engines**: Implement semantic search for documents.
- **Recommendations**: Suggest similar content based on embeddings.

## 🏆 Practical Tasks

1. Generate embeddings for documents.
2. Store embeddings in FAISS and perform semantic search.
3. Visualize similarity scores for search results.

## 💡 Interview Scenarios

**Question**: How do embeddings and vector stores enable RAG?  
**Answer**: Embeddings convert text to vectors; vector stores like FAISS retrieve relevant documents for LLMs.  
**Key**: Enhances context-aware responses.  
**Example**: `openai.Embedding.create` and `faiss.IndexFlatL2`

**Coding Task**: Implement semantic search with FAISS.  
**Tip**: Use `Embedding.create` and FAISS index.

## 📚 Resources

- [OpenAI API Documentation](https://platform.openai.com/docs/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)

## 🤝 Contributions

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/embeddings-vector-stores`).
3. Commit changes (`git commit -m 'Add embeddings and vector stores content'`).
4. Push to the branch (`git push origin feature/embeddings-vector-stores`).
5. Open a Pull Request.