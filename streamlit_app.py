import streamlit as st
import asyncio
import time
from io import StringIO
from typing import Optional, Any, List, Union
import langchain

from langchain_openai import ChatOpenAI
from langchain_core.tools import StructuredTool
from pydantic import Field, SecretStr
from fastmcp import Client, FastMCP
from mcp.types import (
    EmbeddedResource,
    ImageContent,
)
NonTextContent = ImageContent | EmbeddedResource


from langchain_mcp_adapters.tools import _convert_call_tool_result

from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate

from langchain import OpenAI, LLMChain
from langchain.schema import AgentAction, AgentFinish
from langchain.tools import DuckDuckGoSearchRun
import re


async def main():
    ###########MCP###########

    mcp = FastMCP("Image Handler")

    def format_response(response: str) -> str:
        return f"Response: {response}"

    @mcp.tool()
    def test() -> str:
        """Returns the configuration of the Image Hander service"""
        st.session_state.tested += 1
        return f"TEST SUCCESSFUL"
    
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

    def convert_tool(client, mcptool):

        async def call_tool(
            **arguments: dict[str, Any],
        ) -> tuple[str | list[str], list[NonTextContent] | None]:
            call_tool_result = await client.call_tool(mcptool.name, arguments)
            return _convert_call_tool_result(call_tool_result)

        return StructuredTool(
        name=mcptool.name,
        description=mcptool.description or "",
        args_schema=mcptool.inputSchema,
        coroutine=call_tool,
        response_format="content_and_artifact",
        metadata=mcptool.annotations.model_dump() if mcptool.annotations else None,
    )


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

    if "init" not in st.session_state:
        st.session_state.init = True

    if "tested" not in st.session_state:
        st.session_state.tested = 0

    async with Client(mcp) as client:
        search = DuckDuckGoSearchRun()

        def duck_wrapper(input_text):
            search_results = search.run(f"site:webmd.com {input_text}")
            return search_results

        def test():
            res = client.call_tool("test")
            return res

        tools = [
            Tool(
                name = "Search WebMD",
                func=duck_wrapper,
                description="useful for when you need to answer medical and pharmalogical questions"
            ),
            Tool(
                name = "Test Image Handler",
                func=test,
                description="returns configuration of Image Handler",
                return_direct=True
            )
        ]

        template = """You are personal assistant. You have access to the following tools:

        {tools}

        Use the following format:

        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question

        Begin!

        Question: {input}
        {agent_scratchpad}"""

        class CustomPromptTemplate(StringPromptTemplate):
            # The template to use
            template: str
            # The list of tools available
            tools: List[Tool]

            def format(self, **kwargs) -> str:
                # Get the intermediate steps (AgentAction, Observation tuples)
                # Format them in a particular way
                intermediate_steps = kwargs.pop("intermediate_steps")
                thoughts = ""
                for action, observation in intermediate_steps:
                    thoughts += action.log
                    thoughts += f"\nObservation: {observation}\nThought: "
                # Set the agent_scratchpad variable to that value
                kwargs["agent_scratchpad"] = thoughts
                # Create a tools variable from the list of tools provided
                kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
                # Create a list of tool names for the tools provided
                kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
                return self.template.format(**kwargs)

        prompt = CustomPromptTemplate(
            template=template,
            tools=tools,
            # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
            # This includes the `intermediate_steps` variable because that is needed
            input_variables=["input", "intermediate_steps"]
        )

        class CustomOutputParser(AgentOutputParser):
            def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
                # Check if agent should finish
                if "Final Answer:" in llm_output:
                    return AgentFinish(
                        # Return values is generally always a dictionary with a single `output` key
                        # It is not recommended to try anything else at the moment :)
                        return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                        log=llm_output,
                    )
                # Parse out the action and action input
                regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
                match = re.search(regex, llm_output, re.DOTALL)
                if not match:
                    raise ValueError(f"Could not parse LLM output: `{llm_output}`")
                action = match.group(1).strip()
                action_input = match.group(2)
                # Return the action and action input
                return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)

        output_parser = CustomOutputParser()

        llm_chain = LLMChain(llm=model, prompt=prompt)

        tool_names = [tool.name for tool in tools]

        agent = LLMSingleActionAgent(
            llm_chain=llm_chain,
            output_parser=output_parser,
            stop=["\nObservation:"],
            allowed_tools=tool_names
        )

        agent_executor = AgentExecutor.from_agent_and_tools(agent=agent,
                                                        tools=tools,
                                                        verbose=True)
    
    

    

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

            st.text(await client.call_tool("test"))
            st.text(st.session_state.tested)

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

            #st.text(agent_executor.run("How can I treat a spained ankle?"))
            st.text(agent_executor.run("Get the configuration of Image Handler"))

            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""

                #assistant_response = answer_question(prompt, model)

                assistant_response = ["XD"]

                #assistant_response = model.chat.completions.create(model = st.secrets["MODEL"], messages = st.session_state.messages)
                # Simulate stream of response with milliseconds delay
                #for chunk in assistant_response.content.split():
                for chunk in assistant_response:
                    full_response += chunk + " "
                    time.sleep(0.05)
                    # Add a blinking cursor to simulate typing
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    asyncio.run(main())