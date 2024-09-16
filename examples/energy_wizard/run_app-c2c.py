"""This script launches a streamlit app for the Energy Wizard"""
import streamlit as st
import os
import openai
from glob import glob
import pandas as pd
import sys

from elm import EnergyWizard


model = 'gpt-4'

# NREL-Azure endpoint. You can also use just the openai endpoint.
# NOTE: embedding values are different between OpenAI and Azure models!
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_key = os.getenv("AZURE_OPENAI_KEY")
openai.api_type = 'azure'
openai.api_version = os.getenv('AZURE_OPENAI_VERSION')

EnergyWizard.EMBEDDING_MODEL = 'text-embedding-ada-002-2'
EnergyWizard.EMBEDDING_URL = ('https://stratus-embeddings-south-central.'
                              'openai.azure.com/openai/deployments/'
                              'text-embedding-ada-002-2/embeddings?'
                              f'api-version={openai.api_version}')
EnergyWizard.URL = ('https://stratus-embeddings-south-central.'
                    'openai.azure.com/openai/deployments/'
                    f'{model}/chat/completions?'
                    f'api-version={openai.api_version}')
EnergyWizard.HEADERS = {"Content-Type": "application/json",
                        "Authorization": f"Bearer {openai.api_key}",
                        "api-key": f"{openai.api_key}"}

EnergyWizard.MODEL_ROLE = (
    "You are an assistant specializing in the C2C (Communities to Clean) program. "
    "Your goal is to help the user understand the work being done within the program, "
    "focusing on summarizing key aspects of the fact sheets and explaining the results of the work. "
    "The fact sheets describe the work done in various cities, and you should use these materials "
    "to answer questions, summarize the findings, and help the user understand the results behind the projects. "
    "If the fact sheets do not provide enough information to answer the question, you should respond with 'I do not know.' "
    "Be clear, concise, and informative, guiding the user through the important points of the C2C program."
)
EnergyWizard.MODEL_INSTRUCTION = EnergyWizard.MODEL_ROLE


# Load pre-generated embeddings
@st.cache_data
def get_corpus():
    """Get the corpus of text data with embeddings from the existing JSON files."""
    corpus_files = sorted(glob('./embed/*.json'))

    if not corpus_files:
        st.error("No JSON embedding files found in './embed/'.")
        raise FileNotFoundError("No JSON embedding files found in './embed/'. Please make sure the files are generated correctly.")

    st.write(f"Found {len(corpus_files)} JSON embedding files.")

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

    if not corpus_list:
        st.error("No valid JSON files found.")
        raise ValueError("No valid JSON files found in './embed/'. Please check the format of the JSON files.")

    # Concatenate all individual DataFrames into a single DataFrame
    corpus = pd.concat(corpus_list, ignore_index=True)

    # Load metadata
    meta = pd.read_csv('meta.csv')
    st.write(f"Successfully loaded 'meta.csv' with {len(meta)} entries.")

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

@st.cache_resource
def get_wizard():
    """Load the energy wizard with pre-existing embeddings."""
    try:
        corpus = get_corpus()
    except Exception as e:
        st.error(f"Error loading corpus: {e}")
        return None

    if corpus is not None:
        wizard = EnergyWizard(corpus, ref_col='ref', model='gpt-4')
        return wizard
    else:
        return None

if __name__ == '__main__':
    wizard = get_wizard()

    msg = """Hello!\nI am your assistant for the Communities to Clean (C2C) program. 
    I have access to fact sheets that describe the work done in various cities as part of the program. 
    You can ask me questions about these projects, and I will help you understand the work, summarize key points, 
    and explain the results where available. Here are some examples of questions you can ask me:
    \n - Can you summarize the work done in [city name] as part of the C2C program?
    \n - What were the key results of the C2C project in [city name]?
    \n - What challenges were addressed in the C2C program for [city name]?
    \n - How has the C2C program impacted energy use in [city name]?
    \n - What renewable energy solutions were implemented as part of the C2C work in [city name]?
    \n - Can you explain the main findings from the C2C project in [city name]?
    \nIf the information is available in the fact sheets, I'll provide detailed insights to help you better understand the work done in each community."""
    
    st.title(msg)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    msg = "Type your question here"
    if prompt := st.chat_input(msg):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):

            message_placeholder = st.empty()
            full_response = ""

            out = wizard.chat(prompt,
                              debug=True, stream=True, token_budget=6000,
                              temperature=0.0, print_references=True,
                              convo=False, return_chat_obj=True)
            references = out[-1]

            for response in out[0]:
                full_response += response.choices[0].delta.content or ""
                message_placeholder.markdown(full_response + "▌")

            ref_msg = ('\n\nThe wizard was provided with the '
                       'following documents to support its answer:')
            ref_msg += '\n - ' + '\n - '.join(references)
            full_response += ref_msg

            message_placeholder.markdown(full_response)

        st.session_state.messages.append({"role": "assistant",
                                          "content": full_response})
