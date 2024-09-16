import streamlit as st
import os
import json

from openai import AzureOpenAI

# Azure OpenAI configuration
client = AzureOpenAI(
    azure_endpoint="https://ewiz-openai-gpt4.openai.azure.com/",
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2024-02-15-preview"
)

# Define the persona for user profile builder
SYSTEM_MESSAGE_PROFILE_BUILDER = (
    "You are a renewable energy expert focused on underserved communities. "
    "Your primary goal is to gather as much information about the user's background and experience "
    "so that this information can later be passed on to a more technical analysis persona. "
    "You aim to collect data about the user's role, familiarity with renewable energy technologies, "
    "their long-term energy-related goals, and the challenges they are facing. "
    "Focus on basic information, avoiding technical questions at this stage."
)

# Function to store user data in JSON format
def save_data_to_json(data):
    json_data = json.dumps(data, indent=4)
    with open("user_profile_data2.json", "w") as f:
        f.write(json_data)

def user_profile_builder():
    st.title("ADDS Community Support - User Profile Builder Persona")
    st.write(
        """Hello! I am a renewable energy expert focused on learning more about you and your background, so I can better assist you. 
        Please share some information about your experience and role, which will help me guide you toward sustainable energy solutions. 
        Here’s what would be helpful to know:
        \n - What is your role in the community (e.g., utility manager, policymaker, citizen)?
        \n - How familiar are you with renewable energy technologies (e.g., solar, wind, storage)?
        \n - Have you worked on any energy-related projects in the past?
        \n - What are your long-term goals when it comes to energy use in your community?
        \n - What challenges or concerns do you personally have when it comes to energy solutions?
        """
    )

    # Initialize session state if not already done
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Initialize a default JSON structure if it doesn't exist
    if "profile_json" not in st.session_state:
        st.session_state.profile_json = {
            "role": None,
            "familiarity": None,
            "past_projects": None,
            "long_term_goals": None,
            "challenges": None,
        }

    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])

    # Gather user inputs and store in JSON structure
    if prompt := st.chat_input("Type your response here"):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            try:
                completion = client.chat.completions.create(
                    model="ewiz-gpt-4",
                    messages=[
                        {"role": "system", "content": SYSTEM_MESSAGE_PROFILE_BUILDER},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=600,
                    top_p=0.9,
                    frequency_penalty=0.1,
                    presence_penalty=0,
                    stop=None
                )
                full_response = completion.choices[0].message.content

                # Simulate parsing and storing data in JSON
                if st.session_state.profile_json["role"] is None:
                    st.session_state.profile_json["role"] = prompt
                elif st.session_state.profile_json["familiarity"] is None:
                    st.session_state.profile_json["familiarity"] = prompt
                elif st.session_state.profile_json["past_projects"] is None:
                    st.session_state.profile_json["past_projects"] = prompt
                elif st.session_state.profile_json["long_term_goals"] is None:
                    st.session_state.profile_json["long_term_goals"] = prompt
                elif st.session_state.profile_json["challenges"] is None:
                    st.session_state.profile_json["challenges"] = prompt

                # Store the updated data in a JSON file
                save_data_to_json(st.session_state.profile_json)

                message_placeholder.markdown(full_response)

            except Exception as e:
                full_response = f"Error: {e}"
                message_placeholder.markdown(full_response)

            st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == '__main__':
    user_profile_builder()
