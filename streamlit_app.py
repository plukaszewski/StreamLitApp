import streamlit as st
import time
from io import StringIO
import os
from typing import Optional
from langchain_openai import ChatOpenAI
from pydantic import Field, SecretStr
import faiss
import numpy as np
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


###########LLM###########

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

template = """
Answer questions with given template.
Question: {question}
Context: {context}
Answer:
"""

selected_model = "minstralai/minstral-7b-instruct:free"

def answer_question(question, documents, model):
    context = "\n\n".join([doc["text"] for doc in documents])
    prompt = ChatOpenAI.ChatPromptTemplate.from_template(template)
    chain = prompt | model
    return chain.invoke({"question": question, "context": context})

if "query" not in st.session_state:
    st.session_state.query = ""

if "answer" not in st.session_state:
    st.session_state.answer = ""

if "files" not in st.session_state:
    st.session_state.files = []

class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.model = ChatOpenRouter(model_name = selected_model)

    def connect_to_server(self, server_script_path: str):
        server_params = StdioServerParameters(
            command="python", args=[server_script_path], env=None
        )

        stdio_transport = self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self.stdio, self.write = stdio_transport
        self.session = self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )

        self.session.initialize()

        # List available tools
        response = self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    def process_query(self, query: str) -> str:
        """Process a query using Claude and available tools"""
        messages = [
            {
                "role": "user",
                "content": query
            }
        ]

        response = self.session.list_tools()
        available_tools = [{
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.inputSchema
        } for tool in response.tools]

        # Initial Claude API call
        response = self.anthropic.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            messages=messages,
            tools=available_tools
        )

        # Process response and handle tool calls
        final_text = []

        assistant_message_content = []
        for content in response.content:
            if content.type == 'text':
                final_text.append(content.text)
                assistant_message_content.append(content)
            elif content.type == 'tool_use':
                tool_name = content.name
                tool_args = content.input

                # Execute tool call
                result = self.session.call_tool(tool_name, tool_args)
                final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")

                assistant_message_content.append(content)
                messages.append({
                    "role": "assistant",
                    "content": assistant_message_content
                })
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": content.id,
                            "content": result.content
                        }
                    ]
                })

                # Get next response from Claude
                response = self.anthropic.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1000,
                    messages=messages,
                    tools=available_tools
                )

                final_text.append(response.content[0].text)

        return "\n".join(final_text)

    def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == 'quit':
                    break

                response = self.process_query(query)
                print("\n" + response)

            except Exception as e:
                print(f"\nError: {str(e)}")

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

client = MCPClient()
try:
    client.connect_to_server("mcp_server.py")
    client.chat_loop()
finally:
    client.cleanup()

with st.sidebar:
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        st.session_state.files.append(uploaded_file)

    for file in st.session_state.files:
        # To read file as bytes:
        bytes_data = file.getvalue()
        stringio = StringIO(file.getvalue().decode("utf-8"))
        string_data = stringio.read()

        st.write(string_data)

    if st.button("Clear"):
        st.session_state.files = []
        uploaded_file = None

st.caption("RAG")

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
        assistant_response = answer_question(prompt, [], model)
        #assistant_response = model.chat.completions.create(model = st.secrets["MODEL"], messages = st.session_state.messages)
        # Simulate stream of response with milliseconds delay
        for chunk in assistant_response.choices[0].message.content.split():
            full_response += chunk + " "
            time.sleep(0.05)
            # Add a blinking cursor to simulate typing
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})