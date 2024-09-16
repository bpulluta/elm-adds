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

# Define the persona for the modeling persona
SYSTEM_MESSAGE_MODELING_PERSONA = (
    "You are an advanced renewable energy modeling expert. "
    "You are tasked with gathering technical details necessary to run energy models and simulations. "
    "You know the user’s background from previous interactions, including their role, familiarity with renewable energy technologies, "
    "and long-term goals. Use this knowledge to guide your questions and collect specific inputs such as their community's energy costs, "
    "current energy sources, renewable energy initiatives, and energy-related goals."
)

# Function to load user data from JSON file
def load_data_from_json(file_name="user_profile_data.json"):
    try:
        with open(file_name, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("User profile data not found. Please complete the profile builder first.")
        return None

# Function to save technical data to JSON file
def save_data_to_json(data, file_name="user_profile_data.json"):
    try:
        with open(file_name, "w") as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        st.error(f"Error saving data: {e}")

def modeling_persona():
    st.title("ADDS Community Support - Modeling Persona")

    # Load the user profile data from JSON file
    profile_data = load_data_from_json()

    if profile_data:
        st.write(f"Based on the information you provided earlier, I understand the following:")
        st.write(f"- **Role**: {profile_data.get('role', 'Unknown')}")
        st.write(f"- **Familiarity with Renewable Energy**: {profile_data.get('familiarity', 'Unknown')}")
        st.write(f"- **Past Projects**: {profile_data.get('past_projects', 'Unknown')}")
        st.write(f"- **Long-term Goals**: {profile_data.get('long_term_goals', 'Unknown')}")
        st.write(f"- **Challenges**: {profile_data.get('challenges', 'Unknown')}")

        st.write(
            """Now, let's dive into the specifics of your community's energy situation. Could you provide the following details?
            \n - What is the average cost of electricity in your area?
            \n - What energy sources are currently being used (e.g., fossil fuels, solar, wind)?
            \n - Are there any renewable energy initiatives or programs already in place in your area?
            \n - What specific energy-related challenges is your community facing (e.g., high costs, unreliable access)?
            \n - What are your community’s energy-related goals (e.g., transitioning to 100% renewables)?
            """
        )

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            st.chat_message(message["role"]).markdown(message["content"])

        # Gather user inputs for technical details
        if prompt := st.chat_input("Type your answer here"):
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""

                try:
                    completion = client.chat.completions.create(
                        model="ewiz-gpt-4",
                        messages=[
                            {"role": "system", "content": SYSTEM_MESSAGE_MODELING_PERSONA},
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

                    # Simulate parsing and storing the new technical data
                    profile_data["technical_details"] = {
                        "average_electricity_cost": prompt,
                    }

                    # Save the updated data in the JSON file
                    save_data_to_json(profile_data)

                    message_placeholder.markdown(full_response)

                except Exception as e:
                    full_response = f"Error: {e}"
                    message_placeholder.markdown(full_response)

                st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == '__main__':
    modeling_persona()
