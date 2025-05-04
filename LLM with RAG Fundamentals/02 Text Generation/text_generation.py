# Setup: pip install openai requests matplotlib pandas nltk
import os
import openai
import matplotlib.pyplot as plt
import nltk

def run_text_generation_demo():
    # Synthetic Query Data
    queries = [
        "Create a chatbot response for 'What is machine learning?'",
        "Write a 50-word sci-fi story about AI.",
        "Summarize blockchain technology in 3 sentences."
    ]
    print("Synthetic Data: Queries created")
    print(f"Queries: {queries}")

    # OpenAI API Configuration
    openai.api_key = os.getenv("OPENAI_API_KEY", "your-openai-api-key")

    responses = []
    for query in queries:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": query}],
                max_tokens=150
            )
            text = response.choices[0].message.content.strip()
            responses.append(text)
            print(f"Query: {query}")
            print(f"Response: {text}")
        except openai.error.OpenAIError as e:
            print(f"Error for {query}: {e}")

    # Visualization
    response_lengths = [len(nltk.word_tokenize(resp)) for resp in responses]
    plt.figure(figsize=(8, 4))
    plt.bar(range(1, len(queries) + 1), response_lengths, color='blue')
    plt.title("Text Generation Response Lengths")
    plt.xlabel("Query")
    plt.ylabel("Word Count")
    plt.savefig("text_generation_output.png")
    print("Visualization: Response lengths saved as text_generation_output.png")

# Execute the demo
if __name__ == "__main__":
    nltk.download('punkt', quiet=True)
    run_text_generation_demo()