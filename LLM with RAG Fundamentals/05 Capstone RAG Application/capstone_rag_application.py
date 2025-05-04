# Setup: pip install openai langchain faiss-cpu requests matplotlib pandas nltk
import os
import openai
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import matplotlib.pyplot as plt
import numpy as np
import time

def run_capstone_rag_application_demo():
    # Synthetic Knowledge Base
    knowledge_base = [
        "AI in healthcare improves diagnostics with predictive models.",
        "Machine learning requires large, clean datasets for effective training.",
        "Deep learning uses neural networks for tasks like image recognition.",
        "NLP enables chatbots to understand and respond to human language.",
        "Cloud computing provides scalable infrastructure for AI applications."
    ]
    queries = [
        "How does AI improve healthcare?",
        "What are the requirements for machine learning?",
        "What is deep learning used for?",
        "How does NLP work in chatbots?"
    ]
    print("Synthetic Data: Knowledge base and queries created")
    print(f"Knowledge Base: {knowledge_base}")
    print(f"Queries: {queries}")

    # OpenAI API Configuration
    openai.api_key = os.getenv("OPENAI_API_KEY", "your-openai-api-key")

    # Create Vector Store
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(knowledge_base, embeddings)

    # RAG Pipeline
    llm = ChatOpenAI(model="gpt-3.5-turbo", max_tokens=200)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 2})
    )

    # Run Queries and Measure Performance
    responses = []
    latencies = []
    for query in queries:
        start_time = time.time()
        try:
            response = qa_chain.run(query)
            responses.append(response)
            latencies.append(time.time() - start_time)
            print(f"Query: {query}")
            print(f"Response: {response.strip()}")
        except Exception as e:
            print(f"Error for {query}: {e}")

    # Visualization
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(queries) + 1), latencies, marker='o', color='blue')
    plt.title("RAG Application Query Latencies")
    plt.xlabel("Query")
    plt.ylabel("Latency (seconds)")
    plt.savefig("capstone_rag_application_output.png")
    print("Visualization: Latencies saved as capstone_rag_application_output.png")

# Execute the demo
if __name__ == "__main__":
    nltk.download('punkt', quiet=True)
    run_capstone_rag_application_demo()