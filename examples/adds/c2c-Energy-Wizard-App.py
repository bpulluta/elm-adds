import streamlit as st
import os
import numpy as np
import pandas as pd
import json
from glob import glob
from openai import AzureOpenAI
from sklearn.metrics.pairwise import cosine_similarity

# Azure OpenAI configuration
client = AzureOpenAI(
    azure_endpoint="https://ewiz-openai-gpt4.openai.azure.com/",
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2024-02-15-preview"
)

SYSTEM_MESSAGE = """
You are an advanced assistant specializing in the Communities to Clean (C2C) program, 
a technical assistance initiative for cities across the United States. Your primary role 
is to help internal technical assistants and company users understand the intricacies 
of the work done in various cities as part of this program.

Your knowledge base includes detailed fact sheets about each project and a history of 
questions and answers from previous interactions. When answering questions, always provide 
specific references to the relevant projects and cities. Your goal is to offer comprehensive 
insights into:

1. The scope and nature of work done in each city
2. Key stakeholders and experts involved in each project
3. Significant results, outcomes, and impacts of the C2C initiatives
4. Challenges faced and solutions implemented
5. Innovative approaches and technologies utilized
6. Lessons learned and best practices identified
7. Common questions and themes that emerge across multiple user interactions

When providing information, always cite the specific city and any relevant details 
from the fact sheets or previous conversations. If the information isn't available in 
your knowledge base, clearly state that you don't have that specific information.

Remember, your responses should be tailored to assist technical professionals in 
understanding and leveraging the insights from these projects for future initiatives. 
Also, consider patterns in user questions to identify and address common areas of interest 
or confusion about the C2C program.
"""

# Function to load pre-generated embeddings
@st.cache_data
def get_corpus():
    """Get the corpus of text data with embeddings from the existing JSON files."""
    corpus_files = sorted(glob('./embed/*.json'))

    if not corpus_files:
        st.error("No JSON embedding files found in './embed/'.")
        raise FileNotFoundError("No JSON embedding files found in './embed/'. Please make sure the files are generated correctly.")

    corpus_list = []

    # Load each JSON file into a DataFrame
    for fp in corpus_files:
        try:
            data = pd.read_json(fp)
            data['filename'] = os.path.basename(fp).replace('.json', '.docx')
            corpus_list.append(data)
        except ValueError as e:
            st.error(f"Error reading {fp}: {e}")
            continue

    # Concatenate all individual DataFrames into a single DataFrame
    corpus = pd.concat(corpus_list, ignore_index=True)

    # Load metadata
    meta = pd.read_csv('meta.csv')
    meta['filename'] = meta['filename'].astype(str)
    filtered_meta = meta[meta['filename'].isin(corpus['filename'])]
    corpus = corpus.set_index('filename').join(filtered_meta.set_index('filename'), how='left')

    # Ensure metadata is complete
    missing_meta = corpus[corpus.isna().any(axis=1)]
    if not missing_meta.empty:
        st.error(f"Missing metadata for: {missing_meta.index.tolist()}")
        raise ValueError("Missing metadata for some files.")

    corpus['ref'] = [f"{filename} ({row['txt_fp']})" for filename, row in corpus.iterrows()]
    return corpus

# Function to calculate cosine similarity
def get_similar_docs(query_embedding, corpus_embeddings, top_k=5):
    """Find the top_k documents most similar to the query."""
    similarities = cosine_similarity([query_embedding], corpus_embeddings)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return top_indices, similarities[top_indices]

# Function to handle the chat response with document retrieval
def handle_chat_with_corpus(prompt, corpus, chat_history_corpus):
    # Get the user's query embedding from Azure
    try:
        embedding_response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=prompt
        )
        
        query_embedding = embedding_response.data[0].embedding
        
    except Exception as e:
        return f"Error generating query embedding: {e}"

    # Combine corpus and chat history corpus
    combined_corpus = pd.concat([corpus, chat_history_corpus], ignore_index=True)
    combined_embeddings = np.vstack([np.stack(corpus['embedding'].values), np.stack(chat_history_corpus['embedding'].values)])

    # Find the most similar documents
    top_indices, similarities = get_similar_docs(query_embedding, combined_embeddings)

    # Retrieve the top matching documents
    top_docs = combined_corpus.iloc[top_indices]

    # Combine the contents of the top matching documents with their references
    context = "\n\n".join([f"From {row['ref']}:\n{row['text']}" for _, row in top_docs.iterrows()])

    # Prepare the messages for the API call
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": prompt},
        {"role": "system", "content": f"Here is relevant context from the C2C fact sheets and previous conversations:\n\n{context}"}
    ]

    # Send the query to Azure OpenAI along with the relevant context
    try:
        completion = client.chat.completions.create(
            model="ewiz-gpt-4",
            messages=messages,
            temperature=0.7,
            max_tokens=800,
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0,
            stop=None
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error generating completion: {e}"

# Function to load chat history
def load_chat_history():
    try:
        with open('chat_history.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []

# Function to save chat history
def save_chat_history(messages):
    with open('chat_history.json', 'w') as f:
        json.dump(messages, f)

# Function to create embeddings for chat history
def create_chat_history_corpus(chat_history):
    chat_corpus = []
    for i, message in enumerate(chat_history):
        if message['role'] == 'user':
            question = message['content']
            answer = chat_history[i+1]['content'] if i+1 < len(chat_history) and chat_history[i+1]['role'] == 'assistant' else ""
            
            try:
                embedding_response = client.embeddings.create(
                    model="text-embedding-ada-002",
                    input=question
                )
                embedding = embedding_response.data[0].embedding
                
                chat_corpus.append({
                    'text': f"Q: {question}\nA: {answer}",
                    'embedding': embedding,
                    'ref': f"Chat History Entry #{i//2 + 1}"
                })
            except Exception as e:
                st.error(f"Error generating embedding for chat history entry: {e}")
    
    return pd.DataFrame(chat_corpus)

# Main function to launch the Streamlit app
def main():
    st.title("Communities to Clean (C2C) Program Assistant")
    st.write(
"""Welcome to the C2C Program Assistant! This tool is designed to help internal technical 
assistants and company users understand the work done in various cities as part of the 
Communities to Clean program. You can ask questions about specific projects, outcomes, 
challenges, and best practices. The assistant will provide detailed answers with 
references to the relevant fact sheets and previous conversations. Some example questions you can ask:

- What were the main objectives and outcomes of the C2C project in [city name]?
- Who were the key stakeholders involved in the [city name] project?
- What innovative technologies or approaches were used in [city name]'s C2C initiative?
- Can you compare the energy efficiency improvements achieved in [city1] and [city2]?
- What were the common challenges faced across multiple C2C projects, and how were they addressed?
- What best practices for community engagement emerged from the C2C program?
- What are the most frequently asked questions about the C2C program?

Feel free to ask about specific cities, compare different projects, or inquire about 
overall trends and insights from the C2C program. Your questions help improve the 
assistant's knowledge base for future interactions!
"""
    )

    # Load chat history
    if "messages" not in st.session_state:
        st.session_state.messages = load_chat_history()

    # Load the corpus with embeddings
    try:
        corpus = get_corpus()
    except Exception as e:
        st.error(f"Error loading corpus: {e}")
        return

    # Create chat history corpus
    chat_history_corpus = create_chat_history_corpus(st.session_state.messages)

    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])

    if prompt := st.chat_input("Type your question here"):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = handle_chat_with_corpus(prompt, corpus, chat_history_corpus)
            message_placeholder.markdown(full_response)

            st.session_state.messages.append({"role": "assistant", "content": full_response})

        # Save updated chat history
        save_chat_history(st.session_state.messages)

        # Update chat history corpus with the new Q&A pair
        new_chat_corpus = create_chat_history_corpus(st.session_state.messages[-2:])
        chat_history_corpus = pd.concat([chat_history_corpus, new_chat_corpus], ignore_index=True)

if __name__ == '__main__':
    main()