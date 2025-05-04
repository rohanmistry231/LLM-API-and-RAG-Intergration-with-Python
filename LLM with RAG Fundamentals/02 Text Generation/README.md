# 📝 Text Generation

## 📖 Introduction

**Text Generation** uses OpenAI’s Chat and Completion APIs to create conversational and creative text, a core component for RAG applications. This guide provides Python examples, focusing on the AI-driven era (May 3, 2025).

## 🌟 Key Concepts

- **Chat API**: Generating conversational responses.
- **Completion API**: Creating structured or creative text.
- **Prompt Design**: Crafting prompts for quality outputs.
- **Response Metrics**: Evaluating length and relevance.

## 🛠️ Practical Example

```python
%% text_generation.py
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
```

## 📊 Visualization Output

The code generates a bar chart (`text_generation_output.png`) showing response word counts, illustrating output consistency.

## 💡 Applications

- **Chatbots**: Build conversational interfaces for RAG systems.
- **Content Creation**: Generate summaries or stories.
- **RAG Integration**: Provide LLM responses for retrieved data.

## 🏆 Practical Tasks

1. Build a chatbot prompt for user queries.
2. Generate creative text (e.g., story, summary).
3. Visualize response lengths for different queries.

## 💡 Interview Scenarios

**Question**: How do you use OpenAI’s Chat API for text generation?  
**Answer**: The Chat API generates text using a model and messages array, guided by prompts.  
**Key**: Prompt design ensures relevant outputs.  
**Example**: `openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": query}])`

**Coding Task**: Create a chatbot response using OpenAI’s Chat API.  
**Tip**: Use `ChatCompletion.create` with a user prompt.

## 📚 Resources

- [OpenAI API Documentation](https://platform.openai.com/docs/)
- [NLTK Documentation](https://www.nltk.org/)

## 🤝 Contributions

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/text-generation`).
3. Commit changes (`git commit -m 'Add text generation content'`).
4. Push to the branch (`git push origin feature/text-generation`).
5. Open a Pull Request.