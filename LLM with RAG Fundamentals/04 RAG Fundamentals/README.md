# 📚 RAG Fundamentals

## 📖 Introduction

**RAG Fundamentals** introduces Retrieval-Augmented Generation, combining OpenAI’s LLM with external knowledge bases for context-aware responses. This guide provides Python examples using LangChain and FAISS, focusing on the AI-driven era (May 3, 2025).

## 🌟 Key Concepts

- **RAG Overview**: Integrating retrieval with generation.
- **LangChain**: Building RAG pipelines with OpenAI and vector stores.
- **Retrieval**: Fetching relevant documents for LLM context.
- **Evaluation**: Assessing retrieval and response quality.

## 🛠️ Practical Example

```python
%% rag_fundamentals.py
# Setup: pip install openai langchain faiss-cpu requests matplotlib pandas nltk
import os
import openai
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import matplotlib.pyplot as plt
import numpy as np

def run_rag_fundamentals_demo():
    # Synthetic Document Data
    documents = [
        "AI is transforming healthcare with predictive diagnostics.",
        "Machine learning models require large datasets for training.",
        "Deep learning excels in image and speech recognition.",
        "Natural language processing powers chatbots and translation."
    ]
    queries = [
        "How is AI used in healthcare?",
        "What is needed for machine learning?",
        "What does deep learning do?"
    ]
    print("Synthetic Data: Documents and queries created")
    print(f"Documents: {documents}")
    print(f"Queries: {queries}")

    # OpenAI API Configuration
    openai.api_key = os.getenv("OPENAI_API_KEY", "your-openai-api-key")

    # Create Vector Store
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(documents, embeddings)

    # RAG Pipeline
    llm = ChatOpenAI(model="gpt-3.5-turbo", max_tokens=150)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever()
    )

    # Run Queries
    responses = []
    for query in queries:
        try:
            response = qa_chain.run(query)
            responses.append(response)
            print(f"Query: {query}")
            print(f"Response: {response.strip()}")
        except Exception as e:
            print(f"Error for {query}: {e}")

    # Visualization (Simulated Retrieval Accuracy)
    accuracy_scores = [0.9, 0.85, 0.95]  # Simulated for demo
    plt.figure(figsize=(8, 4))
    plt.bar(range(1, len(queries) + 1), accuracy_scores, color='green')
    plt.title("Simulated Retrieval Accuracy")
    plt.xlabel("Query")
    plt.ylabel("Accuracy")
    plt.savefig("rag_fundamentals_output.png")
    print("Visualization: Retrieval accuracy saved as rag_fundamentals_output.png")

# Execute the demo
if __name__ == "__main__":
    nltk.download('punkt', quiet=True)
    run_rag_fundamentals_demo()
```

## 📊 Visualization Output

The code generates a bar chart (`rag_fundamentals_output.png`) showing simulated retrieval accuracy scores, illustrating RAG performance.

## 💡 Applications

- **Question Answering**: Enhance LLM responses with external data.
- **Chatbots**: Provide context-aware conversational agents.
- **Knowledge Management**: Query internal documents intelligently.

## 🏆 Practical Tasks

1. Build a RAG pipeline with LangChain and FAISS.
2. Test RAG with sample queries and documents.
3. Visualize retrieval accuracy or response metrics.

## 💡 Interview Scenarios

**Question**: What is Retrieval-Augmented Generation?  
**Answer**: RAG combines LLMs with retrieval from vector stores to provide context-aware responses.  
**Key**: Improves accuracy over standalone LLMs.  
**Example**: `RetrievalQA.from_chain_type` with FAISS retriever.

**Coding Task**: Build a basic RAG pipeline.  
**Tip**: Use `LangChain` with `OpenAIEmbeddings` and `FAISS`.

## 📚 Resources

- [LangChain Documentation](https://python.langchain.com/docs/)
- [OpenAI API Documentation](https://platform.openai.com/docs/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)

## 🤝 Contributions

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/rag-fundamentals`).
3. Commit changes (`git commit -m 'Add RAG fundamentals content'`).
4. Push to the branch (`git push origin feature/rag-fundamentals`).
5. Open a Pull Request.