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