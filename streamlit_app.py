import streamlit as st
import time
from io import StringIO
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import Field, SecretStr
from fastmcp import Client, FastMCP
import asyncio
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain.agents import AgentExecutor, create_react_agent

async def main():
    ###########MCP###########

    mcp = FastMCP("Image Handler")

    def format_response(response: str) -> str:
        return f"Response: {response}"

    @mcp.tool()
    def test(text: str) -> str:
        """Test the availabilty of the Image Hander service"""
        return f"TEST SUCCESSFUL: {text}"
    
    @mcp.tool()
    def test2(text: str) -> str:
        """Test the correctness of the setup"""
        return f"TEST2 SUCCESSFUL: {text}"

    #########################


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
    You are an assistant for question-answering tasks. You have a set of tools to your disposal.
    If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
    Question: {question} 
    Answer:
    """

    selected_model = "meta-llama/llama-3.3-8b-instruct:free"

    model = ChatOpenRouter(model_name = selected_model)

    tools = await client.list_tools()

    # Create and run the agent
    agent = create_react_agent(model, tools)
    agent_response = await agent.ainvoke({"messages": "Test the availability of Image Handler"})

    def answer_question(question, model):
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | model
        return chain.invoke({"question": question})

    if "query" not in st.session_state:
        st.session_state.query = ""

    if "answer" not in st.session_state:
        st.session_state.answer = ""

    if "files" not in st.session_state:
        st.session_state.files = []

    async with Client(mcp) as client:
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

            for tool in await client.list_tools():
                st.text(tool.name)

            st.text(agent_response.text)

            st.text(await client.call_tool("test", {"text": "test message"}))

        st.caption("MCP")

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
                assistant_response = answer_question(prompt, model)
                #assistant_response = model.chat.completions.create(model = st.secrets["MODEL"], messages = st.session_state.messages)
                # Simulate stream of response with milliseconds delay
                for chunk in assistant_response.content.split():
                    full_response += chunk + " "
                    time.sleep(0.05)
                    # Add a blinking cursor to simulate typing
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})

asyncio.run(main())