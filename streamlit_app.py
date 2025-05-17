import streamlit as st
import time
import openai
from io import StringIO
import os
import fitz
from typing import Optional
from langchain_openai import ChatOpenAI
from pydantic import Field, SecretStr

client = openai.OpenAI(api_key = st.secrets["API_KEY"], base_url = st.secrets["BASE_URL"])
if "files" not in st.session_state:
    st.session_state.files = []

class ChatOpenRouter(ChatOpenAI):
    openai_api_key: Optional[SecretStr] = Field(
        alias="api_key", default_factory=st.secrets["API_KEY"]
    )
    @property
    def lc_secrets(self) -> dict[str, str]:
        return {"openai_api_key": st.secrets["API_KEY"]}

    def __init__(self, openai_api_key: Optional[str] = None, **kwargs):
        openai_api_key = openai_api_key or st.secrets["API_KEY"]
        super().__init__(base_url=st.secrets["BASE_URL"], openai_api_key=openai_api_key, **kwargs)

with st.sidebar:
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        st.session_state.files.append(uploaded_file)

    for file in st.session_state.files:
        # To read file as bytes:
        bytes_data = file.getvalue()
        if file.type == "application/pdf":
            string_data = load_pdf(bytes_data)
        else:
            stringio = StringIO(file.getvalue().decode("utf-8"))
            string_data = stringio.read()

        st.write(string_data)

    if st.button("Clear"):
        st.session_state.files = []
        uploaded_file = None

st.write("Streamlit loves LLMs! 🤖 [Build your own chat app](https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps) in minutes, then make it powerful by adding images, dataframes, or even input widgets to the chat.")

st.caption("Note that this demo app isn't actually connected to any LLMs. Those are expensive ;)")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Let's start chatting! 👇"}]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        assistant_response = client.chat.completions.create(model = st.secrets["MODEL"], messages = st.session_state.messages)
        # Simulate stream of response with milliseconds delay
        for chunk in assistant_response.choices[0].message.content.split():
            full_response += chunk + " "
            time.sleep(0.05)
            # Add a blinking cursor to simulate typing
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})