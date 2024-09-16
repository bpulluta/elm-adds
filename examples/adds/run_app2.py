"""This script launches a streamlit app for the Energy Wizard"""
import streamlit as st
import os
import openai
from glob import glob
import pandas as pd
import sys

from elm import EnergyWizard


model = 'ewiz-gpt-4'

# NREL-Azure endpoint. You can also use just the openai endpoint.
# NOTE: embedding values are different between OpenAI and Azure models!
openai.api_base = 'https://stratus-embeddings-south-central.openai.azure.com/'
openai.api_key = '720df3d64998425ab3a454902c77d9b1'
openai.api_type = 'azure'
openai.api_version = '2024-02-15-preview'

EnergyWizard.EMBEDDING_MODEL = 'ewiz-gpt-4'
EnergyWizard.EMBEDDING_URL = ('https://stratus-embeddings-south-central.'
                              'openai.azure.com/openai/deployments/'
                              'ewiz-gpt-4/embeddings?'
                              f'api-version={openai.api_version}')
EnergyWizard.URL = ('https://stratus-embeddings-south-central.'
                    'openai.azure.com/openai/deployments/'
                    f'{model}/chat/completions?'
                    f'api-version={openai.api_version}')
EnergyWizard.HEADERS = {"Content-Type": "application/json",
                        "Authorization": f"Bearer {openai.api_key}",
                        "api-key": f"{openai.api_key}"}

EnergyWizard.MODEL_ROLE = ('You are an energy research assistant focused on '
                           'understanding the user’s knowledge and identifying '
                           'gaps in their understanding. Your goal is to learn '
                           'what the user knows and does not know about energy research, '
                           'and to help them clarify the key question they are trying '
                           'to answer. Use the provided articles to address the user’s '
                           'specific inquiries, guiding them to deeper understanding. '
                           'If the articles do not provide enough information, say "I do not know."')
EnergyWizard.MODEL_INSTRUCTION = EnergyWizard.MODEL_ROLE


@st.cache_data
def get_corpus():
    """Get the corpus of text data with embeddings."""
    corpus = sorted(glob('./embed/*.json'))
    corpus = [pd.read_json(fp) for fp in corpus]
    corpus = pd.concat(corpus, ignore_index=True)
    meta = pd.read_csv('./meta.csv')

    corpus['osti_id'] = corpus['osti_id'].astype(str)
    meta['osti_id'] = meta['osti_id'].astype(str)
    corpus = corpus.set_index('osti_id')
    meta = meta.set_index('osti_id')

    corpus = corpus.join(meta, on='osti_id', rsuffix='_record', how='left')

    ref = [f"{row['title']} ({row['doi']})" for _, row in corpus.iterrows()]
    corpus['ref'] = ref

    return corpus


@st.cache_resource
def get_wizard():
    """Get the energy wizard object."""

    # Getting Corpus of data. If no corpus throw error for user.
    try:
        corpus = get_corpus()
    except Exception:
        print("Error: Have you run 'retrieve_docs.py'?")
        st.header("Error")
        st.write("Error: Have you run 'retrieve_docs.py'?")
        sys.exit(0)

    wizard = EnergyWizard(corpus, ref_col='ref', model=model)
    return wizard


if __name__ == '__main__':
    wizard = get_wizard()

    msg = """Hello!\nI am the Energy Wizard. I have access to all NREL
    technical reports from 1-1-2022 to present. Note that each question you ask
    is independent. I am not fully conversational yet like ChatGPT is. Here
    are some examples of questions you can ask me:
    \n - What are some of the key takeaways from the LA100 study?
    \n - What kind of work does NREL do on energy security and resilience?
    \n - Who is working on the reV model?
    \n - Who at NREL has published on capacity expansion analysis?
    \n - Can you teach me the basics of grid inertia versus
        inverter based resources?
    \n - What are some of the unique cyber security challenges facing
    renewables?
    \n - Can you give me some ideas for follow-on research related to climate
    change adaptation with renewables?
    """

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
